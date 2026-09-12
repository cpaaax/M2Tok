"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import HfFileSystem, hf_hub_download

from prismatic.conf import ModelConfig
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.vlas import OpenVLA
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
# from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
HF_HUB_REPO = "TRI-ML/prismatic-vlms"
VLA_HF_HUB_REPO = "openvla/openvla-dev"

from transformers import AutoTokenizer

from prismatic.models import Showo, VQ_models
import torch



# === Available Models ===
def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> List[str]:
    return list(GLOBAL_REGISTRY.items())


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===
def load(
    vlm_path: Union[str, Path],
    vq_model_path: Union[str, Path],
    load_for_training: bool = True,
    codebook_size: int = 32768
) -> PrismaticVLM:
    overwatch.info(f"Load local tokenizer from {vlm_path}")
    unify_tokenizer = AutoTokenizer.from_pretrained(vlm_path,
                                              padding_side="left")

    # 添加task start and end
    unify_tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>"})

    # 根据codebook_size添加图像相关的tokens, <img_0> ~ <img_codebook_size-1>
    img_tokens = [f"<img_{i}>" for i in range(codebook_size)]
    unify_tokenizer.add_special_tokens({'additional_special_tokens': ["<soa>", "<eoa>"]})
    act_tokens = [f"<act_{i}>" for i in range(256)]
    img_tokens.extend(act_tokens)
    unify_tokenizer.add_tokens(img_tokens)


    # create and load model
    vq_model = VQ_models["TokenFlow"](
        codebook_size=32768,
        codebook_embed_dim=8,
        semantic_code_dim=32,
        commit_loss_beta=0.25,
        entropy_loss_ratio=0,
        dropout_p=0,
        kmeans=False,
        teacher="siglip_224",
        enhanced_decoder=True,
    )

    vq_model_checkpoint = torch.load(
        vq_model_path,
        map_location="cpu")
    if "ema" in vq_model_checkpoint:  # ema
        model_weight = vq_model_checkpoint["ema"]
    elif "model" in vq_model_checkpoint:  # ddp
        model_weight = vq_model_checkpoint["model"]
    elif "state_dict" in vq_model_checkpoint:
        model_weight = vq_model_checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")

    vq_model.load_state_dict(model_weight)

    vq_model.eval().to(dtype=torch.bfloat16)
    vq_model.requires_grad_(False)


    vlm = Showo.from_pretrained(vlm_path)
    vlm.to(dtype=torch.bfloat16)
    if not load_for_training:
        vlm.requires_grad_(False)
        vlm.eval()
    else:
        vlm.requires_grad_(True)

    return vlm, vq_model, unify_tokenizer


# === Load Pretrained VLA Model ===
def load_vla(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    step_to_load: Optional[int] = None,
    model_type: str = "pretrained",
) -> OpenVLA:
    """Loads a pretrained OpenVLA from either local disk or the HuggingFace Hub."""

    # TODO (siddk, moojink) :: Unify semantics with `load()` above; right now, `load_vla()` assumes path points to
    #   checkpoint `.pt` file, rather than the top-level run directory!
    if os.path.isfile(model_id_or_path):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(model_id_or_path))}`")

        # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
        assert (checkpoint_pt.suffix == ".pt") and (checkpoint_pt.parent.name == "checkpoints"), "Invalid checkpoint!"
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"

    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`VLA_HF_HUB_REPO`)
    else:
        # Search HF Hub Repo via fsspec API
        overwatch.info(f"Checking HF for `{(hf_path := str(Path(VLA_HF_HUB_REPO) / model_type / model_id_or_path))}`")
        if not (tmpfs := HfFileSystem()).exists(hf_path):
            raise ValueError(f"Couldn't find valid HF Hub Path `{hf_path = }`")

        # Identify Checkpoint to Load (via `step_to_load`)
        step_to_load = f"{step_to_load:06d}" if step_to_load is not None else None
        valid_ckpts = tmpfs.glob(f"{hf_path}/checkpoints/step-{step_to_load if step_to_load is not None else ''}*.pt")
        if (len(valid_ckpts) == 0) or (step_to_load is not None and len(valid_ckpts) != 1):
            raise ValueError(f"Couldn't find a valid checkpoint to load from HF Hub Path `{hf_path}/checkpoints/")

        # Call to `glob` will sort steps in ascending order (if `step_to_load` is None); just grab last element
        target_ckpt = Path(valid_ckpts[-1]).name

        overwatch.info(f"Downloading Model `{model_id_or_path}` Config & Checkpoint `{target_ckpt}`")
        with overwatch.local_zero_first():
            relpath = Path(model_type) / model_id_or_path
            config_json = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO, filename=f"{(relpath / 'config.json')!s}", cache_dir=cache_dir
            )
            dataset_statistics_json = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO, filename=f"{(relpath / 'dataset_statistics.json')!s}", cache_dir=cache_dir
            )
            checkpoint_pt = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO, filename=f"{(relpath / 'checkpoints' / target_ckpt)!s}", cache_dir=cache_dir
            )

    # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
    with open(config_json, "r") as f:
        vla_cfg = json.load(f)["vla"]
        model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])()

    # Load Dataset Statistics for Action Denormalization
    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)

    # = Load Individual Components necessary for Instantiating a VLA (via base VLM components) =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg.model_id}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg.vision_backbone_id}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg.llm_backbone_id}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg.arch_specifier}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg.vision_backbone_id}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg.vision_backbone_id,
        model_cfg.image_resize_strategy,
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg.llm_backbone_id,
        llm_max_length=model_cfg.llm_max_length,
        hf_token=hf_token,
        inference_mode=not load_for_training,
    )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(llm_backbone.get_tokenizer())

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLA [bold blue]{model_cfg.model_id}[/] from Checkpoint")
    vla = OpenVLA.from_pretrained(
        checkpoint_pt,
        model_cfg.model_id,
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg.arch_specifier,
        freeze_weights=not load_for_training,
        norm_stats=norm_stats,
        action_tokenizer=action_tokenizer,
    )

    return vla
