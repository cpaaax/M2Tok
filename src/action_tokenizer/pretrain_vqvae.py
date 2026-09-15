import os
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from vqvae_new.mgpt_vq import VQVae
import wandb
from load_data.dataset import ActionDataset

os.environ["WANDB_MODE"] = "offline"
os.environ["NCCL_P2P_DISABLE"] = "1"

import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def main(rank, cfg, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


    if rank==0:
        os.environ["WANDB_API_KEY"] = cfg.wandb.api_key
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity", None),
            name=os.path.abspath(__file__).split('/')[-2]
        )

    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)


    vqvae_model = VQVae(
        input_emb_width=cfg.vqvae_model.action_dim,
        codebook_size=cfg.vqvae_model.codebook_size,
        codebook_dim=cfg.vqvae_model.codebook_dim,
        n_latent_dims=cfg.vqvae_model.n_latent_dims,
        down_t=cfg.vqvae_model.down_t,
        output_emb_width=512,
        depth=3,
        dilation_growth_rate=3,
        num_codebooks=cfg.vqvae_model.num_codebooks,
    )
    vqvae_model.to(rank).train()

    vqvae_model = DDP(vqvae_model, device_ids=[rank])


    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    train_data = ActionDataset(cfg.train_data_path, data_scale=cfg.data_scale)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)


    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.batch_size, pin_memory=False, num_workers=8, sampler=train_sampler
    )

    # set the optimizer

    params = list(vqvae_model.parameters())
    vqvae_optimizer = torch.optim.AdamW(
        params, lr=cfg.optim.lr,  # weight_decay=cfg.optim.weight_decay
    )

    global_step = 0
    for epoch in tqdm.trange(cfg.epochs):
        train_sampler.set_epoch(epoch)  # ensure that the random order is different for each epoch
        for data in tqdm.tqdm(train_loader):
            act = data.to(rank)
            recon_out, vq_loss, usages = vqvae_model(act)
            recon_loss = torch.nn.MSELoss()(recon_out, act)

            loss = recon_loss * cfg.vqvae_model.recon_loss_weight + vq_loss
            vqvae_optimizer.zero_grad()
            loss.backward()

            total_grad_sum = 0.0

            for param in vqvae_model.parameters():
                if param.grad is not None:
                    total_grad_sum += param.grad.sum().item()


            torch.nn.utils.clip_grad_norm_(vqvae_model.parameters(), cfg.max_grad_norm)

            vqvae_optimizer.step()

            logs = {
                "pretrain/code_usages": usages.item(),
                "pretrain/vq_loss": vq_loss.item(),
                "pretrain/recon_loss": recon_loss.item(),
                "pretrain/total_loss": loss.item(),
                "pretrain/total_grad": total_grad_sum
            }
            global_step += 1
            if rank==0:
                wandb.log(logs, step=global_step)

        if epoch > 0 and epoch % cfg.save_every_epoch == 0:
            if rank==0:
                state_dict = vqvae_model.module.state_dict()
                torch.save(state_dict, os.path.join(cfg.save_path, f"trained_vqvae_{epoch}.pt"))
    if rank==0:
        state_dict = vqvae_model.module.state_dict()
        torch.save(state_dict, os.path.join(cfg.save_path, f"trained_vqvae_{epoch}.pt"))
    # Tear down the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    cfg = get_config()
    assert cfg.device_num <=torch.cuda.device_count()

    world_size = cfg.device_num
    rank = int(os.environ['RANK'])
    main(rank, cfg, world_size)
