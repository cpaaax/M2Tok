# coding=utf-8
# Copyright 2024 NUS Show Lab, HuggingFace.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F


class VLA_Model(torch.nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
            self,
            vlm,
            w_clip_vit=False,
    ):
        super().__init__()
        self.w_clip_vit = w_clip_vit
        self.model = vlm


    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def forward(
            self,
            input_ids=None,
            input_embeddings=None,
            attention_mask=None,
            labels=None,
            vocab_size=-1,
            **kwargs,
    ):
        if input_embeddings is None:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)['logits']
        else:
            logits = self.model(inputs_embeds=input_embeddings, attention_mask=attention_mask)['logits']

        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, vocab_size),
                labels[:, 1:].contiguous().view(-1), ignore_index=-100,
            )

            # return logits, loss_t2i, loss_lm, loss_mmu
            return logits, loss


        return logits


    @torch.no_grad()
    def mmu_generate_img_actions(
            self,
            idx=None,
            input_embeddings=None,
            attention_mask=None,
            max_new_tokens=100,
            temperature=1.0,
            top_k=None,
            eot_token=None,
            img_token_start_idx=None,
            img_token_end_idx=None,
            img_embedding_weight=None,
            action_vqvae_model=None,
            act_token_start_idx=None,
            act_token_end_idx=None,
            action_num_codebooks=None,
            action_projector=None,
            force_decode=True,  # Force tokens at each step to be within the corresponding codebook indices
            special_act_start_end_token_ids=None,
            head_num=None,
            codebook_size=None
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        cur_action_idx = 0

        for cur_token_num in range(max_new_tokens):

            logits = self(idx, input_embeddings=input_embeddings, attention_mask=attention_mask)

            attention_mask = torch.cat((
                attention_mask,
                torch.ones((1, 1)).to(device)
            ), dim=1)


            logits = logits[:, -1, :]
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)


            idx_next = torch.argmax(probs)

            # left_start … left_end | left_start … left_end | right_start … right_end | right_start … right_end | eos
            # 0               9           10         19           20             29           30         39

            #         special_act_start_end_token_ids = [
            #             self.tokenizer.left_arm_soa_token_id,
            #             self.tokenizer.left_arm_eoa_token_id,
            #             self.tokenizer.right_arm_soa_token_id,
            #             self.tokenizer.right_arm_eoa_token_id
            #             ]
            left_start_1 = 0
            left_start_2 = head_num+2

            left_end_1 = head_num+1
            left_end_2 = head_num*2+3

            right_start_1 = head_num*2+4
            right_start_2 = head_num*3+6

            right_end_1 = head_num*3+5
            right_end_2 = head_num*4+7


            if cur_token_num in [left_start_1, left_start_2]:
                idx_next = torch.tensor(special_act_start_end_token_ids[0]).to(device)
                idx_next_embeddings = self.model.model.embed_tokens(idx_next)
            elif cur_token_num in [left_end_1, left_end_2]:
                idx_next = torch.tensor(special_act_start_end_token_ids[1]).to(device)
                idx_next_embeddings = self.model.model.embed_tokens(idx_next)
            elif cur_token_num in [right_start_1, right_start_2]:
                idx_next = torch.tensor(special_act_start_end_token_ids[2]).to(device)
                idx_next_embeddings = self.model.model.embed_tokens(idx_next)
            elif cur_token_num in [right_end_1, right_end_2]:
                idx_next = torch.tensor(special_act_start_end_token_ids[3]).to(device)
                idx_next_embeddings = self.model.model.embed_tokens(idx_next)
            elif (cur_token_num in range(left_start_1+1, left_end_1)
                  or cur_token_num in range(left_start_2+1, left_end_2)
                  or cur_token_num in range(right_start_1+1, right_end_1)
                  or cur_token_num in range(right_start_2+1, right_end_2)):
                if force_decode:
                    cur_token_id_scope = [cur_action_idx * (codebook_size//head_num) + act_token_start_idx,
                                          (cur_action_idx + 1) * (codebook_size//head_num) + act_token_start_idx]
                    if idx_next >= cur_token_id_scope[1] or idx_next < cur_token_id_scope[
                        0]:  # Recompute idx_next if it is out of the expected range
                        valid_logits = probs[:, cur_token_id_scope[0]:cur_token_id_scope[1]]  # Get logits within the desired range
                        idx_next = torch.argmax(valid_logits) + cur_token_id_scope[0]


                # If current token is an action token, track its position within the 8 (num_head) action tokens to get the corresponding codebook's embedding.
                act_feat = action_vqvae_model.quantizer.idx_to_specifical_codebook_f(cur_action_idx,
                                                                                     idx_next - act_token_start_idx)
                idx_next_embeddings = action_projector(act_feat)
                cur_action_idx += 1
                cur_action_idx = cur_action_idx % head_num

            else:
                    # print(idx_next.item())
                    # print(eot_token)
                    # print(special_act_start_end_token_ids)
                    idx_next = torch.tensor(eot_token[0]).to(device)
                    assert idx_next.item() in eot_token
                    idx_next_embeddings = self.model.model.embed_tokens(idx_next)



            result.append(idx_next)
            input_embeddings = torch.cat([input_embeddings, idx_next_embeddings.view(1, 1, -1)], dim=1)
            # print(idx_next.cpu())
            if eot_token is not None and idx_next.item() in eot_token:
                break

        return result


    @torch.no_grad()
    def mmu_generate_img_actions_parallel_decoding(
            self,
            idx=None,
            input_embeddings=None,
            attention_mask=None,
            max_new_tokens=100,
            temperature=1.0,
            top_k=None,
            eot_token=None,
            img_token_start_idx=None,
            img_token_end_idx=None,
            img_embedding_weight=None,
            action_vqvae_model=None,
            act_token_start_idx=None,
            act_token_end_idx=None,
            action_num_codebooks=None,
            action_projector=None,
            force_decode=True,
            special_act_start_end_token_ids=None,
            action_token_length=None,
            head_num=None,
            codebook_size=None

    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        cur_action_idx = 0

        logits = self(idx, input_embeddings=input_embeddings, attention_mask=attention_mask)
        logits = logits[0, -(action_token_length+1):-1, :]

        probs = F.softmax(logits, dim=-1)

        for cur_token_num in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            # logits, _ = self(idx_cond)

            idx_next = torch.argmax(probs[cur_token_num])
            # left_start … left_end | left_start … left_end | right_start … right_end | right_start … right_end | eos
            # 0               9           10         19           20             29           30         39

            #         special_act_start_end_token_ids = [
            #             self.tokenizer.left_arm_soa_token_id,
            #             self.tokenizer.left_arm_eoa_token_id,
            #             self.tokenizer.right_arm_soa_token_id,
            #             self.tokenizer.right_arm_eoa_token_id
            #             ]
            left_start_1 = 0
            left_start_2 = head_num+2

            left_end_1 = head_num+1
            left_end_2 = head_num*2+3

            right_start_1 = head_num*2+4
            right_start_2 = head_num*3+6

            right_end_1 = head_num*3+5
            right_end_2 = head_num*4+7


            # 让 arm soa eoa的顺序也确定
            if cur_token_num in [left_start_1, left_start_2]:
                idx_next = torch.tensor(special_act_start_end_token_ids[0]).to(device)
            elif cur_token_num in [left_end_1, left_end_2]:
                idx_next = torch.tensor(special_act_start_end_token_ids[1]).to(device)
            elif cur_token_num in [right_start_1, right_start_2]:
                idx_next = torch.tensor(special_act_start_end_token_ids[2]).to(device)
            elif cur_token_num in [right_end_1, right_end_2]:
                idx_next = torch.tensor(special_act_start_end_token_ids[3]).to(device)
            elif (cur_token_num in range(left_start_1+1, left_end_1)
                  or cur_token_num in range(left_start_2+1, left_end_2)
                  or cur_token_num in range(right_start_1+1, right_end_1)
                  or cur_token_num in range(right_start_2+1, right_end_2)):
                if force_decode:
                    cur_token_id_scope = [cur_action_idx * (codebook_size//head_num) + act_token_start_idx,
                                          (cur_action_idx + 1) * (codebook_size//head_num) + act_token_start_idx]
                    if idx_next >= cur_token_id_scope[1] or idx_next < cur_token_id_scope[
                        0]:
                        valid_logits = probs[cur_token_num,
                                       cur_token_id_scope[0]:cur_token_id_scope[1]]
                        idx_next = torch.argmax(valid_logits) + cur_token_id_scope[0]


                cur_action_idx += 1
                cur_action_idx = cur_action_idx % 8

            else:
                # print(idx_next.item())
                # print(eot_token)
                # print(special_act_start_end_token_ids)
                idx_next = torch.tensor(eot_token[0]).to(device)
                assert idx_next.item() in eot_token
            result.append(idx_next)

            if eot_token is not None and idx_next.item() in eot_token:
                break

        return result

class ImageProjector(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImageProjector, self).__init__()
        self.image_projector = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, output_dim)
        )
    def forward(self, x):
        return self.image_projector(x)






class StateProjector(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StateProjector, self).__init__()
        self.state_projector = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, output_dim)
        )
    def forward(self, x):
        return self.state_projector(x)

class ActionProjector(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActionProjector, self).__init__()
        self.action_projector = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, output_dim)
        )
    def forward(self, x):
        return self.action_projector(x)




class UnitModel(torch.nn.Module):
    def __init__(self, vla, action_projector, image_projector):
        super().__init__()
        self.vla = vla
        self.action_projector = action_projector
        self.image_projector = image_projector



