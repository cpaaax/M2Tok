from typing import List, Optional, Union

from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from .resnet_mine import Resnet1D
from .quant import VectorQuantizerM

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention


from transformers.activations import ACT2FN
from transformers.integrations import use_kernel_forward_from_hub


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"





def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    # **kwargs,
):
    # key_states = repeat_kv(key, module.num_key_value_groups)
    # value_states = repeat_kv(value, module.num_key_value_groups)
    key_states = key
    value_states = value
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    # if attention_mask is not None:
    #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    #     attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights



class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.gate_proj = nn.Linear(self.hidden_size, self.out_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.out_size, bias=False)
        self.down_proj = nn.Linear(self.out_size, self.out_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,
                 hidden_size,
                 out_size,
                 num_attention_heads
                 ):
        super().__init__()
        # self.config = config
        # self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = int(out_size/num_attention_heads)

        # self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = True

        self.q_proj = nn.Linear(
            self.hidden_size, out_size, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, out_size, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, out_size, bias=False
        )
        self.o_proj = nn.Linear(
            out_size, out_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        # position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        # attention_mask: Optional[torch.Tensor],
        # past_key_value: Optional[Cache] = None,
        # cache_position: Optional[torch.LongTensor] = None,
        # **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

class ProcessLayer(nn.Module):
    def __init__(self, hidden_size, out_size, num_attention_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_attention_heads = num_attention_heads

        self.attn = LlamaAttention(self.hidden_size, self.out_size, self.num_attention_heads)
        self.mlp_input = LlamaMLP(self.hidden_size, out_size)
        self.mlp_hidden = LlamaMLP(out_size, out_size)

        self.input_layernorm = LlamaRMSNorm(self.hidden_size)
        self.post_attention_layernorm = LlamaRMSNorm(self.out_size)

    def forward(self, hidden_states):

        hidden_states = self.input_layernorm(hidden_states)
        residual = hidden_states
        # Self Attention
        hidden_states, self_attn_weights = self.attn(
            hidden_states=hidden_states,
        )
        hidden_states = self.mlp_input(residual) + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp_hidden(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class VQVae(nn.Module):

    def __init__(self,
                 input_emb_width: int,
                 codebook_size=512,
                 codebook_dim=512,
                 output_emb_width=512,
                 n_latent_dims=512,
                 down_t=2,
                 depth=3,
                 dilation_growth_rate=3,
                 num_codebooks=2,
                 **kwargs) -> None:
        super().__init__()

        self.codebook_dim = codebook_dim

        self.encoder = Encoder(input_emb_width,
                               output_emb_width,
                               n_latent_dims,
                               down_t,
                               depth,
                               dilation_growth_rate,
                               )

        self.decoder = Decoder(input_emb_width,
                               output_emb_width,
                               n_latent_dims,
                               down_t,
                               depth,
                               dilation_growth_rate,
                               )

        self.quantizer = VectorQuantizerM(
            vocab_size=codebook_size,
            vocab_width=codebook_dim,
            beta=0.25,
            num_codebooks=num_codebooks,
        )

        self.quant_proj = nn.Sequential(
            ProcessLayer(n_latent_dims, codebook_dim, num_codebooks),
            ProcessLayer(codebook_dim, codebook_dim, num_codebooks),

        )
        self.post_quant_proj = nn.Sequential(
            ProcessLayer(codebook_dim, codebook_dim, num_codebooks),
            ProcessLayer(codebook_dim, n_latent_dims, num_codebooks),

        )


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: Tensor):
        # Preprocess
        x_in = self.preprocess(features) # [batch, action_chunk, action_dim]->[batch, action_dim, action_chunk]

        # Encode
        x_encoder = self.encoder(x_in)  # [batch, n_latent_dims, action_chunk/4]
        x_encoder = x_encoder.permute(0, 2, 1)  # [batch, action_chunk/4, n_latent_dims]
        x_encoder_proj = self.quant_proj(x_encoder)  # [batch, action_chunk/4, code_dim]

        # quantization
        x_quantized, vq_loss, entropy_loss, usages = self.quantizer(x_encoder_proj)

        x_quantized_proj = self.post_quant_proj(x_quantized) # [batch, action_chunk/4, n_latent_dims]

        x_quantized_proj = x_quantized_proj.permute(0, 2, 1) # [batch, n_latent_dims, action_chunk/4]
        # decoder
        x_decoder = self.decoder(x_quantized_proj) # [batch, action_dim, action_chunk]
        x_out = self.postprocess(x_decoder) # [batch, action_chunk, action_dim]

        return x_out, vq_loss, usages

    def action_to_idx(self, action):
        action = self.preprocess(action) # [batch,7,chunk_size]
        features = self.encoder(action).permute(0, 2, 1) # [batch, 512, chunk_size/4]->[batch, chunk_size/4, 512]
        features = self.quant_proj(features) # [batch, chunk_size/4, 512]
        return self.quantizer.f_to_idx(features)

    def idx_to_action(self, indices):
        features = self.quantizer.idx_to_f(indices)  # indices: [batch,chunk_size/4,8], features: [batch,chunk_size/4,2048]
        features = self.post_quant_proj(features).permute(0, 2, 1)  # features: [batch,chunk_size/4,512]->[batch,512,batch,chunk_size/4]
        action = self.decoder(features) # [batch,7,chunk_size]
        action = self.postprocess(action) # [batch,chunk_size,7]
        return action


class Encoder(nn.Module):

    def __init__(self,
                 input_emb_width,
                 output_emb_width,
                 n_latent_dims=512,
                 down_t=3,
                 depth=3,
                 dilation_growth_rate=3,
                 ):
        super().__init__()

        modules = []
        modules.append(nn.Sequential(nn.Conv1d(input_emb_width, output_emb_width, 3, 1, 1),
                       nn.ReLU()))

        for _ in range(down_t):
            block = nn.Sequential(
                nn.Conv1d(output_emb_width, output_emb_width, 4, 2, 1),
                Resnet1D(output_emb_width,
                         depth,
                         dilation_growth_rate,
                         ),
            )
            modules.append(block)
        modules.append(nn.Conv1d(output_emb_width, n_latent_dims, 3, 1, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)



class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=7,
                 output_emb_width=512,
                 n_latent_dims=512,
                 down_t=2,
                 depth=3,
                 dilation_growth_rate=3,
                 ):
        super().__init__()
        blocks = []

        modules = []
        modules.append(nn.Sequential(nn.Conv1d(n_latent_dims, output_emb_width, 3, 1, 1),
                                     nn.ReLU()))
        for _ in range(down_t):
            block = nn.Sequential(
                Resnet1D(output_emb_width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         ), nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(output_emb_width, output_emb_width, 3, 1, 1))
            blocks.append(block)



        blocks.append(nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
