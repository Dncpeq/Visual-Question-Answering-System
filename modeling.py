import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, List, Dict, Any, Union, Iterable
from timm.layers import DropPath
from dataclasses import dataclass
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.utils import logging
from transformers.cache_utils import Cache, StaticCache, DynamicCache, SlidingWindowCache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration, 
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLModel,
    Qwen2_5_VLConfig
)

from config import *

logger = logging.get_logger(__name__)

class MLP(nn.Module):
    def __init__(self, in_feature, intermediate_feature = None, out_feature = None, bias = True, act_fn = None):
        super().__init__()
        self.in_feature = in_feature
        self.intermediate_feature = intermediate_feature or in_feature
        self.out_feature = out_feature or in_feature
        self.gate_proj = nn.Linear(self.in_feature, self.intermediate_feature, bias=bias)
        self.up_proj = nn.Linear(self.in_feature, self.intermediate_feature, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_feature, self.out_feature, bias=bias)
        self.act_fn = act_fn or nn.GELU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, act_fn = None, bias = False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.act = act_fn or nn.GELU()

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
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




class DeepSeekMoE(nn.Module):
    def __init__(
        self,
        config: DeepSeekMoEConfig
    ):
        '''
        alpha: Balance factor for auxiliary loss
        gamma: Bias adjustment rate
        comp_seq_wise_auxiliary_loss: set to use Complementary Sequence-Wise Auxiliary Loss in training
        '''
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_shared = config.num_shared_experts
        self.num_routed = config.num_routed_experts
        self.top_k = config.top_k
        self.drop_out = config.drop_out
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.comp_seq_wise_auxiliary_loss = config.comp_seq_wise_auxiliary_loss
        self.act_fn = config.act_fn
        self.bias = config.bias
        self.variance = config.variance

        # Set intermediate sizes with defaults
        self.shared_expert_intermediate_size = config.shared_expert_intermediate_size or self.hidden_size
        self.routed_expert_intermediate_size = config.routed_expert_intermediate_size or self.hidden_size // 4

        # Shared experts: MLPs applied to all tokens
        self.shared_experts = nn.ModuleList([
            MLP(self.hidden_size, self.shared_expert_intermediate_size, act_fn=self.act_fn, bias=self.bias)
            for _ in range(self.num_shared)
        ])

        # Routed experts: MLPs selectively activated
        self.routed_experts = nn.ModuleList([
            MLP(self.hidden_size, self.routed_expert_intermediate_size, act_fn=self.act_fn, bias=self.bias)
            for _ in range(self.num_routed)
        ])

        # Learnable centroids for gating (shape: num_routed_experts x hidden_size)
        _t = torch.empty(self.num_routed, self.hidden_size)
        nn.init.normal_(
            _t, 
            mean=0.0, 
            std=math.sqrt(self.variance)
        )
        self.expert_centroids = nn.Parameter(_t)

        # Biases for load balancing (adjusted programmatic, not via gradients)
        self.expert_biases = nn.Parameter(torch.zeros(self.num_routed), requires_grad=False)


    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        total_tokens = batch_size * seq_len

        # **Shared Experts**
        shared_output = torch.zeros_like(x) # (batch_size, seq_len, hidden_size)
        for expert in self.shared_experts:  
            shared_output += expert(x)

        # **Routed Experts**

        # **Gating Mechanism**
        # Compute affinity scores: sigmoid(u_t^T e_i)
        affinity_scores = torch.sigmoid(x @ self.expert_centroids.t())  # (batch_size, seq_len, num_routed_experts)

        # Add biases for routing
        routing_scores = affinity_scores + self.expert_biases[None, None, :]  # (batch_size, seq_len, num_routed_experts)

        # Select top-K experts
        selected_indices = torch.topk(routing_scores, k=self.top_k, dim=2).indices  # (batch_size, seq_len, top_k)
        selected_affinity_scores = torch.gather(affinity_scores, dim=2, index=selected_indices)  # (batch_size, seq_len, top_k)

        # Compute gating values
        _d = selected_affinity_scores.sum(dim=2, keepdim=True)
        gating_values = selected_affinity_scores / _d # (batch_size, seq_len, top_k)

        x_flat = x.reshape(total_tokens, -1)  # (total_tokens, hidden_size)
        selected_indices_flat = selected_indices.reshape(total_tokens, self.top_k)  # (total_tokens, top_k)
        gating_values_flat = gating_values.reshape(total_tokens, self.top_k)  # (total_tokens, top_k)

        routed_output = torch.zeros_like(x_flat)  # (total_tokens, hidden_size)
        token_indices = torch.arange(total_tokens, device=x.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1) # (total_tokens * top_k)
        expert_indices = selected_indices_flat.reshape(-1)  # (total_tokens * top_k)
        gating_vals = gating_values_flat.reshape(-1)  # (total_tokens * top_k)

        for expert in range(self.num_routed):
            mask = (expert_indices == expert)
            if mask.any():
                selected_token_idxs = token_indices[mask]
                selected_gating_vals = gating_vals[mask]
                expert_input = x_flat[selected_token_idxs]
                expert_output = self.routed_experts[expert](expert_input)
                weighted_output = expert_output * selected_gating_vals.unsqueeze(1)
                routed_output[selected_token_idxs] += weighted_output

        # Combine outputs
        output = shared_output + routed_output.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, hidden_size)

        aux_loss = None

        if not self.training:
            return output, aux_loss

        # Expert load per sequence
        expert_load_per_seq = torch.zeros(batch_size, self.num_routed, device=x.device)
        selected_indices_batch = selected_indices.view(batch_size, -1)  # (batch_size, seq_len * top_k)
        expert_load_per_seq.scatter_add_(
            1, selected_indices_batch, torch.ones(batch_size, seq_len * self.top_k, device=x.device)
        )  # (batch_size, num_routed_experts)


        # **Complementary Sequence-Wise Auxiliary Loss**
        if self.comp_seq_wise_auxiliary_loss:
            # Normalized affinity scores
            normalized_affinity_scores = affinity_scores / affinity_scores.sum(dim=2, keepdim=True)  # (batch_size, seq_len, num_routed_experts)
            P = normalized_affinity_scores.mean(dim=1)  # (batch_size, num_routed_experts), P_i per sequence
            # Compute f_i = (N_r / (K_r * T)) * sum_t 1{s_i,t in Topk}
            f = (self.num_routed / (self.top_k * seq_len)) * expert_load_per_seq  # (batch_size, num_routed_experts)
            # Loss per sequence: alpha * sum_i (f_i * P_i)
            loss_per_seq = self.alpha * (f * P).sum(dim=1)  # (batch_size,)
            aux_loss = loss_per_seq.mean()  # Scalar


        # **Auxiliary-Loss-Free Load Balancing**
        # Total load across batch
        total_load = expert_load_per_seq.sum(dim=0)  # (num_routed_experts)
        expected_load = (self.top_k * total_tokens) / self.num_routed  # Expected selections per expert

        # Adjust biases directly at end of forward pass
        with torch.no_grad():
            bias_adjustments = torch.where(
                total_load > expected_load,
                -self.gamma,  # Decrease bias for overloaded experts
                self.gamma    # Increase bias for underloaded experts
            )
            self.expert_biases += bias_adjustments

        return output, aux_loss


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, args):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)
    


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_mrope_section(cos: torch.Tensor, sin: torch.Tensor, mrope_section: List[int] = None):
    """
    input:
        cos & sin shape: [num_modalities, bsz, seq, head_dim].
        mrope_section List[int]: NOTE: the sum of mrope_section should be half of head_dim.
    output cos & sin shape: [bsz, seq, head_dim].
    """
    if mrope_section is None:
        mrope_section = [cos.shape[-1] // 2]

    n = len(mrope_section)
    mrope_section = mrope_section * 2

    # mcos, msin = [], []
    # for i, (mc, ms) in enumerate(zip(cos.split(mrope_section, dim=-1), sin.split(mrope_section, dim=-1))):
    #     mcos.append(mc[i % n])
    #     msin.append(ms[i % n])

    # cos = torch.cat(mcos, dim=-1)
    # sin = torch.cat(msin, dim=-1)

    cos = torch.cat([m[i % n] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
    sin = torch.cat([m[i % n] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)

    return cos, sin


# modified from transformers
class RotaryEmbedding(nn.Module):
    def __init__(
        self, 
        head_dim,
        rope_theta = 10000.0,
        max_position_embeddings = 10000,
        dynamic = True,
        mrope_section = None,
        device = None
    ):
        super().__init__()
        self.dim = head_dim
        self.rope_theta = rope_theta
        self.dynamic = dynamic
        self.mrope = mrope_section is not None
        self.mrope_section = mrope_section

        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
        
        inv_freq, self.attention_scaling = self._compute_default_rope_parameters(head_dim, rope_theta, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def _compute_default_rope_parameters(dim, base, seq_len = None, device = None):
        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        attention_scaling = 1.0
        return inv_freq, attention_scaling
    
    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self._compute_default_rope_parameters(self.dim, self.rope_theta, seq_len=seq_len, device=device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.dynamic:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, Qwen2_5_VL has different position ids for the grids
        # So we expand the inv_freq to shape (D, ...)
        # mrope
        if self.mrope:
            inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(len(self.mrope_section), position_ids.shape[1], -1, 1)
            position_ids_expanded = position_ids[:, :, None, :].float()  # shape (D, bs, 1, positions)
        # 1D rope
        else:
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):

            if self.mrope:
                # mrope
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            else:
                # 1D rope
                freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        if self.mrope:
            cos, sin = apply_mrope_section(cos, sin, self.mrope_section)

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)



class MultiheadLatentAttention(nn.Module):
    def __init__(self, config: MultiheadLatentAttentionConfig, index: int = None):

        assert config.hidden_size % config.num_heads == 0

        super(MultiheadLatentAttention, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        head_dim = self.hidden_size // self.num_heads
        self.nope_qk_head_dim = config.nope_qk_head_dim or head_dim
        self.nope_v_head_dim = config.nope_v_head_dim or head_dim
        self.c_kv_dim = config.c_kv_dim                    # KV compression dimension
        self.c_q_dim = config.c_q_dim                      # query compression dimension
        self.rope_head_dim = config.rope_head_dim          # RoPE head dimension
        self.scaling = config.scaling or 1 / math.sqrt(self.nope_qk_head_dim + self.rope_head_dim)
        self.attn_drop = config.attn_drop
        self.layer_idx = index
        self.rms_norm_eps = config.rms_norm_eps
        self.bias = config.bias

        # return_weights can't use flash-attn and torch sdpa
        # flash-attn and torch sdpa can not use the same time 
        assert not (config.return_weights and (config.use_flash_attn or config.use_torch_sdpa)), \
            "Cannot use flash_attn or torch_sdpa when return_weights is True"

        # Enforce Constraint 2: Cannot use both accelerated attention methods simultaneously.
        assert not (config.use_flash_attn and config.use_torch_sdpa), \
            "Cannot use both flash_attn and torch_sdpa at the same time"

        if config.return_weights:
            self.sdpa_func = self._sdpa
        elif config.use_flash_attn:
            self.sdpa_func = self._flash_attn_sdpa
        elif config.use_torch_sdpa:
            self.sdpa_func = self._torch_sdpa
        else:
            self.sdpa_func = self._torch_sdpa

        # down projection (low rank compression)
        self.Wd_q = nn.Linear(self.hidden_size, self.c_q_dim, bias=self.bias)
        self.Wd_kv_kr = nn.Linear(self.hidden_size, self.c_kv_dim + self.rope_head_dim, bias=self.bias)

        # up projection
        self.Wu_q_qr = nn.Linear(self.c_q_dim, self.num_heads * (self.nope_qk_head_dim + self.rope_head_dim), bias=self.bias)
        self.Wu_kv = nn.Linear(self.c_kv_dim, self.num_heads * (self.nope_qk_head_dim + self.nope_v_head_dim), bias=self.bias) # k_head_dim + v_head_dim

        # output projection
        self.W_o = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)

        self.cq_norm = RMSNorm(self.c_q_dim, eps=self.rms_norm_eps)
        self.ckv_norm = RMSNorm(self.c_kv_dim, eps=self.rms_norm_eps)
    
    def _sdpa(self, q, k, v, attn_mask):
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # [bsz, heads, q_seq_len, kv_seq_len]
        if attn_mask is not None:
            if attn_mask.dtype is not torch.bool:
                attn_mask = attn_mask.to(torch.bool)
            attn_weights = torch.masked_fill(attn_weights, attn_mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attn_drop, training=self.training)
        attn_output = attn_weights @ v
        return attn_output, attn_weights

    def _torch_sdpa(self, q, k, v, attn_mask):
        if attn_mask is not None:
            if attn_mask.dtype is not torch.bool:
                attn_mask = attn_mask.to(torch.bool)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop, scale=self.scaling), None
    
    def _flash_attn_sdpa(self, q, k, v, attn_mask):
        raise NotImplementedError()
        return
    
    def forward(
            self, 
            hidden_states, 
            position_embeddings: Tuple[torch.Tensor, torch.Tensor], 
            attn_mask: torch.Tensor = None,
            past_kv_kr: Union[Cache, List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            cache_position: torch.Tensor = None,
            use_cache: bool = False,
        ):
        '''
        args:
            attn_mask [torch.Tensor]: dtype torch.bool, shape: [bsz, heads, q_seq_len, kv_seq_len]
        '''

        bsz, q_seq_len, _ = hidden_states.size()

        # Project the query and key/value inputs.
        c_q = self.Wd_q(hidden_states)
        c_kv_kr = self.Wd_kv_kr(hidden_states)  # [bsz, q_seq_len, c_kv_dim + rope_head_dim]

        # Process query: normalize, project, reshape, then split into q and its rotary part (qr) via slicing.
        q_qr = self.Wu_q_qr(self.cq_norm(c_q))
        q_qr = q_qr.view(bsz, q_seq_len, self.num_heads, self.nope_qk_head_dim + self.rope_head_dim).transpose(1, 2)
        q = q_qr[..., :self.nope_qk_head_dim]
        qr = q_qr[..., self.nope_qk_head_dim:]

        # Process key/value: split into c_kv and its rotary part (kr) via slicing, then normalize c_kv.
        c_kv = c_kv_kr[..., :self.c_kv_dim] # [bsz, q_seq_len, c_kv_dim]
        kr = c_kv_kr[..., self.c_kv_dim:]
        c_kv = self.ckv_norm(c_kv)
        # expand shared kr to all key heads. (kr is shared by all heads)
        kr = kr.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # Apply RoPE to qr & kr.
        cos, sin = position_embeddings

        qr_rot, kr_rot = apply_rotary_pos_emb(qr, kr, cos, sin, unsqueeze_dim=1)  # (bsz, num_heads, q_seq_len, rope_head_dim)

        # Update cache if required.
        if use_cache and past_kv_kr is not None:
            assert self.layer_idx is not None, "Cannot cache without layer index"
            shared_kr_rot = kr_rot[:, 0:1, :, :]
            if isinstance(past_kv_kr, list):
                past_c_kv, past_kr_rot = past_kv_kr[self.layer_idx]
                c_kv = torch.cat((past_c_kv, c_kv), dim=-2)
                kr_rot = torch.cat((past_kr_rot, shared_kr_rot), dim=-2)
                past_kv_kr[self.layer_idx] = (c_kv, kr_rot)
            else:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                c_kv, kr_rot = past_kv_kr.update(c_kv, shared_kr_rot, self.layer_idx, cache_kwargs)
            kr_rot = kr_rot.expand(-1, self.num_heads, -1, -1)

        kv_seq_len = c_kv.shape[1]

        # Compute keys and values, then split using slicing.
        # up project compressed kv history context, this would introduce a performance overhead.
        kv = self.Wu_kv(c_kv)   
        kv = kv.view(bsz, kv_seq_len, self.num_heads, self.nope_qk_head_dim + self.nope_v_head_dim).transpose(1, 2)
        k = kv[..., :self.nope_qk_head_dim]
        v = kv[..., self.nope_qk_head_dim:]

        # compose q, k
        q = torch.cat([q, qr_rot], dim=-1)
        k = torch.cat([k, kr_rot], dim=-1)
        
        # Scaled dot-product attention.
        attn_output, attn_weights = self.sdpa_func(q, k, v, attn_mask)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_seq_len, -1)
        attn_output = self.W_o(attn_output)

        return attn_output, attn_weights, past_kv_kr


# class MultiheadLatentAttention(nn.Module):
#     def __init__(self, config: MultiheadLatentAttentionConfig, index: int = None):
#         assert config.hidden_size % config.num_heads == 0

#         super(MultiheadLatentAttention, self).__init__()
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_heads
#         self.head_dim = self.hidden_size // self.num_heads
#         self.scaling = config.scaling or 1 / math.sqrt(self.head_dim)
#         self.attn_drop = config.attn_drop
#         self.layer_idx = index
#         self.rms_norm_eps = config.rms_norm_eps
#         self.bias = config.bias

#         self.W_qkv = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=self.bias)
#         # output projection
#         self.W_o = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)

#         self.q_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
#         self.k_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)

#     def forward(
#         self, 
#         hidden_states, 
#         position_embeddings: Tuple[torch.Tensor, torch.Tensor], 
#         attn_mask: torch.Tensor = None,
#         past_key_value: Union[Cache, List[Tuple[torch.Tensor, torch.Tensor]]] = None,
#         cache_position: torch.Tensor = None,
#         use_cache: bool = False,
#     ):
#         bsz, q_seq_len, _ = hidden_states.size()
#         qkv = self.W_qkv(hidden_states).view(bsz, q_seq_len, 3, self.num_heads, self.head_dim)
#         q = qkv[:, :, 0:1, :, :].squeeze(2).transpose(1,2)
#         k = qkv[:, :, 1:2, :, :].squeeze(2).transpose(1,2)
#         v = qkv[:, :, 2:3, :, :].squeeze(2).transpose(1,2)

#         cos, sin = position_embeddings
#         q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
#         if past_key_value is not None:
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             k_rot, v = past_key_value.update(k_rot, v, self.layer_idx, cache_kwargs)

#         attn_out = F.scaled_dot_product_attention(
#             self.q_norm(q_rot), self.k_norm(k_rot), v, 
#             attn_mask = attn_mask, 
#             dropout_p = self.attn_drop if self.training else 0.0,
#             scale = self.scaling
#         )

#         attn_out = attn_out.transpose(1, 2).reshape(bsz, q_seq_len, -1)
#         attn_out = self.W_o(attn_out)

#         return attn_out, None, past_key_value


class MultiwayLayer(nn.Module):
    def __init__(
        self,
        config: MultiwayLayerConfig,
        index: int = None,
    ):
        '''
        model_list: modality_mask will route the tokens in index, 
            make sure it's modality order match with modality_mask
        '''
        super(MultiwayLayer, self).__init__()

        self.hidden_size = config.hidden_size
        self.num_shared_modalities = config.num_shared_modalities
        self.num_routed_modalities = config.num_routed_modalities

        attn_config = config.attn_config
        shared_moe_config = config.shared_moe_config
        routed_moe_config = config.routed_moe_config

        # Normalization layers (pre-normalization)
        norm = config.norm or RMSNorm
        self.norm_attn = norm(self.hidden_size, eps=config.rms_norm_eps)
        self.norm_ffn = norm(self.hidden_size, eps=config.rms_norm_eps)

        # Shared multi-head latent attention mechanism
        self.attn = MultiheadLatentAttention(attn_config, index)

        # Shared Modality FFNs (applied to all tokens)
        self.shared_ffns = nn.ModuleList([
            DeepSeekMoE(shared_moe_config)
            for _ in range(self.num_shared_modalities)
        ])

        # Routed (Modality specific) FFNs
        self.routed_ffns = nn.ModuleList([
            DeepSeekMoE(routed_moe_config)
            for _ in range(self.num_routed_modalities)
        ])

        # Drop path for stochastic depth
        self.drop_path = DropPath(config.drop_path) if config.drop_path > 0.0 else nn.Identity()

        # act_fn = config.act_fn or nn.GELU()
        # self.act_fn = act_fn()
        # self.gate = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_mask: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_kv_kr: List[torch.FloatTensor] = None,
        use_cache = False,
        cache_position = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        '''
        Forward pass through the MultiwayLayer.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_size).
            modality_mask: Tensor indicating the modality index for each token,
                           shape (batch_size, sequence_length). Values should correspond
                           to indices in `routed_model_list`.
            attn_mask: Optional attention mask.

        Returns:
            Tuple containing:
                - Output tensor of shape (batch_size, sequence_length, hidden_size).
                - Attention scores/weights (or None if not returned by self.attn).
                - Total auxiliary loss from shared and routed models.
        '''
        assert hidden_states.shape[0:2] == modality_mask.shape[0:2] ,\
            f"modality mask has the shape {modality_mask.shape} does not match expected [bsz, seq_len], which [{hidden_states.shape[0]}, {hidden_states.shape[1]}]" \

        # --- Attention Block ---
        residual = hidden_states
        hidden_states = self.norm_attn(hidden_states)   # Pre-normalization
        attn_out, attn_score, past_kv_kr = self.attn(
            hidden_states, 
            position_embeddings, 
            attn_mask,
            past_kv_kr,
            cache_position,
            use_cache,
        )

        if self.training:
            attn_out = self.drop_path(attn_out)

        hidden_states = residual + attn_out # First residual connection

        # --- Shared and Routed FFN Block ---
        residual = hidden_states
        hidden_states = self.norm_ffn(hidden_states)  # Pre-normalization for FFNs

        # Initialize output tensors
        shared_models_out = torch.zeros_like(hidden_states)
        routed_models_out = torch.zeros_like(hidden_states)

        aux_losses = []

        # 1. Process Shared Modalities (apply to all tokens)
        for shared_mod in self.shared_ffns:
            # Assuming shared_mod might return (output, aux_loss)
            mod_out, aux_loss = shared_mod(hidden_states)
            if aux_loss is not None: aux_losses.append(aux_loss)
            shared_models_out = shared_models_out + mod_out # Accumulate outputs

        # 2. Process Routed Modalities (dispatch tokens)
        flat_normed_x = hidden_states.view(-1, hidden_states.size(-1)) # (B*Seq, Hidden)
        flat_modality_mask = modality_mask.view(-1) # (B*Seq)

        for mod_idx in range(self.num_routed_modalities):
            # Find tokens belonging to the current modality
            mod_token_indices = torch.where(flat_modality_mask == mod_idx)[0]

            if mod_token_indices.numel() > 0:
                mod_tokens = flat_normed_x[mod_token_indices] # (NumTokens, Hidden)

                # Models might expect batch dimension, e.g., (1, NumTokens, Hidden)
                # Adjust based on specific model requirements. Assuming (B, Seq, Hidden) input.
                mod_tokens_reshaped = mod_tokens.unsqueeze(0)

                # Assuming routed_mod might return (output, aux_loss)
                mod_out, aux_loss = self.routed_ffns[mod_idx](mod_tokens_reshaped)

                if aux_loss is not None: aux_losses.append(aux_loss)

                # Remove the batch dimension added earlier
                mod_out = mod_out.squeeze(0) # (NumTokens, Hidden)

                # Place the results back into the correct positions in routed_models_out
                # Need to reshape routed_models_out view for indexed assignment
                routed_models_out.view(-1, routed_models_out.size(-1))[mod_token_indices] = mod_out

        # 3. Combine, Scale, Dropout, and Add Residual
        combined_ffn_out = shared_models_out + routed_models_out    # need improvements, to avoid conflict

        if self.training:
            combined_ffn_out = self.drop_path(combined_ffn_out)

        hidden_states = residual + combined_ffn_out # Second residual connection

        # --- Handle Auxiliary Losses ---
        total_aux_loss = sum(aux_losses) if aux_losses else None

        return hidden_states, attn_score, past_kv_kr, total_aux_loss



class TransformerLayer(nn.Module):
    def __init__(
        self,
        config: TransformerLayerConfig,
        index: int = None
    ):
        super(TransformerLayer, self).__init__()

        # Normalization layers (pre-normalization)
        self.hidden_size = config.hidden_size

        norm = config.norm or RMSNorm
        self.norm_attn = norm(self.hidden_size, eps=config.rms_norm_eps)
        self.norm_ffn = norm(self.hidden_size, eps=config.rms_norm_eps)

        attn_config = config.attn_config
        moe_config = config.moe_config

        self.attn = MultiheadLatentAttention(attn_config, index)
        # self.moe = DeepSeekMoE(moe_config)
        self.ffn = MLP(self.hidden_size, self.hidden_size * 4, bias=config.moe_config.bias)

        # Drop path for stochastic depth
        self.drop_path = DropPath(config.drop_path) if config.drop_path > 0.0 else nn.Identity()


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        past_kv_kr = None,
        use_cache = False,
        cache_position = None,
    ) -> Tuple[torch.Tensor, Tuple, torch.Tensor]:
        # Attention block with residual connection
        residual = hidden_states
        hidden_states = self.norm_attn(hidden_states)  # Pre-normalization
        attn_out, attn_score, past_kv_kr = self.attn(
            hidden_states, 
            position_embeddings, 
            attn_mask,
            past_kv_kr,
            cache_position,
            use_cache,
        )

        if self.training:
            attn_out = self.drop_path(attn_out)
        hidden_states = residual + attn_out

        # Feedforward network block with residual connection
        residual = hidden_states
        hidden_states = self.norm_ffn(hidden_states)  # Pre-normalization
        ffn_out = self.ffn(hidden_states)
        if self.training:
            ffn_out = self.drop_path(ffn_out)
        hidden_states = residual + ffn_out

        return hidden_states, attn_score, past_kv_kr, None


class Multiway(nn.Module):
    def __init__(
        self,
        config: MultiwayConfig
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_multiway_layers = config.num_multiway_layers
        self.num_fusion_layers = config.num_fusion_layers

        # Initialize the stack of layers
        self.layers = nn.ModuleList([
            MultiwayLayer(config.multiway_layer_config, idx) 
            for idx in range(self.num_multiway_layers)
        ] + [
            TransformerLayer(config.fusion_layer_config, self.num_multiway_layers + idx) 
            for idx in range(self.num_fusion_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_mask: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        past_kv_krs: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """
        hidden_states: [bsz, seq_len, hidden_size]
        modality_mask: [bsz, seq_len], integer, represent each token's modality, 
            for example text = 0, image = 1, input seq [[t, t, v, v], [t, t, v, t]], modality mask: [[0, 0, 1, 1], [0, 0, 1, 0]]
        attn_mask: where masked is 1, not masked is 0
        """
        aux_losses = []
        total_attn_weights = []
        total_hidden_states = []
            
        for layer in self.layers[:self.num_multiway_layers]:
            if output_hidden_states: total_hidden_states.append(hidden_states)
            hidden_states, attn_weights, past_kv_krs, aux_loss = layer(
                hidden_states,
                modality_mask=modality_mask,
                position_embeddings=position_embeddings,
                past_kv_kr=past_kv_krs,
                use_cache=use_cache,
                cache_position=cache_position,
                attn_mask=attn_mask,
            )
            if aux_loss is not None: aux_losses.append(aux_loss)
            if output_attentions: total_attn_weights.append(attn_weights)
        
        for layer in self.layers[self.num_multiway_layers:]:
            if output_hidden_states: total_hidden_states.append(hidden_states)
            hidden_states, attn_weights, past_kv_krs, aux_loss = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                past_kv_kr=past_kv_krs,
                use_cache=use_cache,
                cache_position=cache_position,
                attn_mask=attn_mask,
            )
            if aux_loss is not None: aux_losses.append(aux_loss)
            if output_attentions: total_attn_weights.append(attn_weights)

        if output_hidden_states: total_hidden_states.append(hidden_states)
        total_aux_loss = sum(aux_losses) if aux_losses else None
        # Return based on caching requirement
        return hidden_states, total_attn_weights, total_hidden_states, past_kv_krs, total_aux_loss



class BiasedDynamicCache(DynamicCache):
    """
    A DynamicCache that supports a layer index bias.

    This allows combining caches from different model parts (e.g., an encoder and a decoder)
    into a single cache object by providing an offset (`layer_bias`) for the second part.
    Layer indices passed to methods like `update`, `__getitem__`, `get_seq_length` are interpreted
    relative to this bias (logical indices), while the cache internally stores data based on
    physical indices (logical_idx + layer_bias).
    Methods operating on the entire batch or physical structure often don't need explicit bias handling
    internally, but their interpretation might depend on the bias context.
    """

    def __init__(self, layer_bias: int = 0, _distributed_cache_data: Iterable = None) -> None:
        """
        Initializes the BiasedDynamicCache.

        Args:
            layer_bias (`int`, *optional*, defaults to 0):
                The offset to apply to layer indices. When accessing layer `k` from a model
                component associated with this bias, the cache will internally access index `k + layer_bias`.
            _distributed_cache_data (`Iterable`, *optional*):
                 Data for distributed cache initialization (see DynamicCache docs).
                 Bias is applied *after* potential initialization from this data.
        """
        super().__init__(_distributed_cache_data=_distributed_cache_data)
        self._layer_bias = layer_bias # Use a private attribute

    def get_bias(self) -> int:
        """Returns the current layer bias."""
        return self._layer_bias

    def set_bias(self, new_bias: int) -> None:
        """Sets a new layer bias."""
        if not isinstance(new_bias, int):
            raise TypeError("layer_bias must be an integer")
        self._layer_bias = new_bias

    @property
    def layer_bias(self) -> int:
        """Provides read-only access to the bias via property."""
        return self._layer_bias

    def _get_internal_idx(self, layer_idx: int) -> int:
        """Adjusts the logical layer index by the bias to get the internal list index."""
        # Ensure logical layer_idx is non-negative before applying bias
        if layer_idx < 0:
            raise IndexError(f"Logical layer index ({layer_idx}) cannot be negative.")
        return layer_idx + self._layer_bias

    def _get_logical_idx(self, internal_idx: int) -> int:
        """Adjusts the internal list index back to the logical layer index."""
        # Ensure internal index is valid relative to bias
        if internal_idx < self._layer_bias:
            raise IndexError(f"Internal index ({internal_idx}) is less than the current bias ({self._layer_bias}).")
        return internal_idx - self._layer_bias

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int, # Logical layer index
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache for the logical layer `layer_idx`. Internally, this accesses the
        cache at index `layer_idx + self.layer_bias`.
        """
        layer_idx = self._get_internal_idx(layer_idx)
        k_cache, v_cache = super().update(key_states, value_states, layer_idx, cache_kwargs)

        return k_cache, v_cache

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Accesses the cache for the logical layer `layer_idx`. Internally, this accesses the
        cache at index `layer_idx + self.layer_bias`.
        """
        internal_idx = self._get_internal_idx(layer_idx)
        # Use superclass __getitem__ with the internal index
        try:
            return super().__getitem__(internal_idx)
        except KeyError:
             # Provide a more informative error message considering the bias
             raise KeyError(
                  f"Cache does not contain internal index {internal_idx} (derived from logical layer {layer_idx} "
                  f"and bias {self._layer_bias}). Current physical cache length is {len(self.key_cache)}."
             ) from None # Suppress the original KeyError context

    def __iter__(self):
        """
        Iterates over the cache entries corresponding to the logical layer indices
        `0, 1, ...` that are valid given the current bias and physical cache size.
        """
        logical_idx = 0
        while True:
            try:
                # Use self.__getitem__ which handles bias and checks validity
                yield self[logical_idx]
                logical_idx += 1
            except KeyError:
                # Stop iteration when we exceed the valid logical indices
                break


    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Returns the sequence length of the cached states for the logical layer `layer_idx`.
        Internally, this accesses the cache at index `layer_idx + self.layer_bias`.
        """
        internal_idx = self._get_internal_idx(layer_idx)
        # Use superclass get_seq_length with the internal index
        return super().get_seq_length(internal_idx)

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the cache instance into the legacy cache format, representing
        the layers accessible via logical indices starting from 0."""
        legacy_cache = ()
        # Use the modified iterator which yields based on logical indices
        for key_val_pair in self:
             k = key_val_pair[0] if key_val_pair[0] != [] else None
             v = key_val_pair[1] if key_val_pair[1] != [] else None
             legacy_cache += ((k, v),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, layer_bias: int = 0) -> "BiasedDynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `BiasedDynamicCache`.
        Assumes legacy cache represents logical layers 0, 1, ... which will be stored internally
        offset by the provided `layer_bias`.
        """
        # Create an instance with the desired bias *before* populating
        cache = cls(layer_bias=layer_bias)
        if past_key_values is not None:
            for logical_layer_idx in range(len(past_key_values)):
                 key_states, value_states = past_key_values[logical_layer_idx]
                 if key_states is not None and value_states is not None:
                      # Update using the logical index; internal placement handles bias
                      cache.update(key_states, value_states, logical_layer_idx)
                 else:
                     # Ensure physical cache list is long enough even for None layers by calling update
                     # with None or ensuring padding in update logic (current update handles padding)
                     # We might need to explicitly pad here if update doesn't get called for None
                     internal_idx = cache._get_internal_idx(logical_layer_idx)
                     if len(cache.key_cache) <= internal_idx:
                          for _ in range(len(cache.key_cache), internal_idx + 1):
                               cache.key_cache.append([])
                               cache.value_cache.append([])

        return cache

    # --- Methods operating on physical structure / batch dimension ---
    # crop, batch_split, from_batch_splits, batch_repeat_interleave, batch_select_indices
    # generally rely on the physical structure. Ensure bias is handled correctly if
    # creating new instances (like in batch_split/from_batch_splits).

    def batch_split(self, full_batch_size: int, split_size: int) -> List["BiasedDynamicCache"]:
        """Split the current instance into a list of `BiasedDynamicCache` by the batch size."""
        # Uses superclass logic but ensures new instances have the correct bias
        splits = super().batch_split(full_batch_size, split_size)
        biased_splits = []
        for split in splits:
            # Re-create as BiasedDynamicCache instance, copying data and bias
            new_split = self.__class__(layer_bias=self.layer_bias)
            new_split._seen_tokens = split._seen_tokens
            new_split.key_cache = split.key_cache # Assumes super().batch_split copied correctly
            new_split.value_cache = split.value_cache
            biased_splits.append(new_split)
        return biased_splits


    @classmethod
    def from_batch_splits(cls, splits: List["BiasedDynamicCache"]) -> "BiasedDynamicCache":
        """This is the opposite of the above `batch_split()` method."""
        if not splits:
            return cls() # Default bias 0

        # Assume all splits have the same bias and structure
        bias = splits[0].get_bias() # Use getter
        cache = cls(layer_bias=bias)
        num_layers = max(len(split) for split in splits) # Max physical layers

        cache.key_cache = [[] for _ in range(num_layers)]
        cache.value_cache = [[] for _ in range(num_layers)]

        for internal_idx in range(num_layers): # Iterate through physical layers
            key_cache_parts = []
            value_cache_parts = []
            for current in splits:
                # Check if current split has this physical layer and it's not empty
                if internal_idx < len(current) and len(current.key_cache[internal_idx]) != 0:
                    key_cache_parts.append(current.key_cache[internal_idx])
                    value_cache_parts.append(current.value_cache[internal_idx])

            if key_cache_parts:
                layer_keys = torch.cat(key_cache_parts, dim=0)
                layer_values = torch.cat(value_cache_parts, dim=0)
                # Use update with the *logical* index corresponding to this internal index
                try:
                    logical_idx = cache._get_logical_idx(internal_idx)
                    cache.update(layer_keys, layer_values, logical_idx) # Update uses logical index
                except IndexError:
                    # This internal index might not map to a valid logical index if bias > 0
                    # This indicates an issue with assuming consistent structure or bias?
                    # Or maybe just skip? For now, let's assume consistency and update.
                    print(f"Warning: Internal index {internal_idx} might be inconsistent with bias {bias} during from_batch_splits.")
                    logical_idx = internal_idx # Fallback? Or error? Let's assume update is correct
                    cache.update(layer_keys, layer_values, logical_idx)


        cache._seen_tokens = splits[0]._seen_tokens # Assume consistent
        return cache
  


@dataclass
class MultiwayPretrainedModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    total_aux_loss: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None



class MultiwayBase(PreTrainedModel):
    config_class = MultiwayBasePretrainedConfig

    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = []  # for model parallel, specify for internal class inherited PretrainedModel
    _skip_keys_device_placement = "past_key_values" 
    # _supports_flash_attn_2 = True
    # _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = False  # TODO (joao): fix. torch.compile failing probably due to `cache_positions`



# integrate with Qwen2_5_VisionTransformerPretrainedModel and Multiway.
# the Qwen2_5_VisionTransformerPretrainedModel is aiming to merge small patches to hidden size of Multiway.
# Multiway takes text and image inputs together and output a reach of representation embeddings.

# NOTE: deprecated
class MultiwayBasePreTrained(MultiwayBase):
    config_class = MultiwayBasePretrainedConfig

    def __init__(self, config: MultiwayBasePretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.multiway_model = Multiway(config.multiway_config) # nn.Module model

        self.embed_vision = Qwen2_5_VisionTransformerPretrainedModel(config.vision_config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.mrope_section is not None:
            assert sum(config.mrope_section) == config.rope_head_dim // 2, \
            "mrope_section config error, the sum of mrope_section should be half the rope dim"

        self.multiway_rotary_emb = RotaryEmbedding(
            config.rope_head_dim, 
            config.rope_theta, 
            config.max_position_embeddings,
            mrope_section = config.mrope_section
        )

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        std = self.multiway_config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        else:
            ...
            # raise NotImplementedError(f"type:{type(module)}, {str(module)} init method not implement.")

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, embed):
        self.embed_tokens = embed

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, embed):
        self.lm_head = embed

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas


    def forward(
        self,
        input_ids: torch.LongTensor = None,                 # one of input_ids or input_embeds must be provide
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: torch.Tensor = None,    # necessary shape: [bsz, 1, q_seq_len, kv_seq_len]
        position_ids: Optional[torch.LongTensor] = None,    # shape: [bsz, seq_len]
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # for consistent naming, the actual value is kv, kr
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,  # shape: [bsz, seq_len]
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = False,
        use_gradient_checkpointing: Optional[bool] = False,  # calling model from gradient checkpointing for memory efficiency

        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,

    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        modality_mask: for each token, to specify which FFN should be chose
            [bsz, seq_len], integer, represent each token's modality, 
            for example text = 0, image = 1, 
            input batch [[t, t, v, v], [t, t, v, t]], 
            modality mask: [[0, 0, 1, 1], [0, 0, 1, 0]]
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_cache = use_cache if use_cache else self.config.use_cache
        return_dict = return_dict if return_dict else self.config.use_return_dict

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # **Input Preparation**
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)  # [bsz, seq_len, hidden_size]
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.embed_vision.dtype)
                image_embeds = self.embed_vision(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id  # [bsz, seq_len]
                mask_unsqueezed = mask.unsqueeze(-1)    # [bsz, seq_len， 1]
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds) # [bsz, seq_len, hidden_size]
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.embed_vision.dtype)
                video_embeds = self.embed_vision(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = BiasedDynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # in usual mrope_section have 3 sections, which is temporal, height and width.
        D = len(self.config.mrope_section)
        assert D != position_ids[0], "position_ids modality dimension mismatch with mrope_section"

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(D, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).expand(D, position_ids.shape[0], -1)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.multiway_rotary_emb(hidden_states, position_ids)

        # modality mask is generate using special token id, such as image/visual token id.
        modality_mask = ...

        # call model from checkpointing for memory efficiency, but slower than directly call.
        if self.gradient_checkpointing and self.training and use_gradient_checkpointing:
            model_outputs = self._gradient_checkpointing_func(
                self.multiway_model.__call__,
                hidden_states,
                modality_mask,
                position_embeddings,
                attention_mask, # mask logic is different 
                past_key_values,
                use_cache,
                cache_position,
                output_attentions,
                output_hidden_states,
            )
        else:
            model_outputs = self.multiway_model(
                hidden_states,
                modality_mask,
                position_embeddings,
                attention_mask,
                past_key_values,
                use_cache,
                cache_position,
                output_attentions,
                output_hidden_states,
            )

        # NOTE: MultiheadLatentAttention use kv, kr cache component, but due to still two component,
        # can be compatible with common cache object, also support List[(Tensor, Tensor)] cache.
        hidden_states, total_attn_weights, total_hidden_states, past_kv_krs, total_aux_loss = model_outputs

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()    # itc, mim, mlm
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            return hidden_states, total_attn_weights, total_hidden_states, past_kv_krs, logits, loss, total_aux_loss
        return MultiwayPretrainedModelOutputWithPast(
            last_hidden_state=hidden_states,
            loss=loss,
            logits=logits,
            total_aux_loss = total_aux_loss,
            past_key_values=past_kv_krs,
            hidden_states=total_hidden_states,
            attentions=total_attn_weights,
        )


def slice_tensor(
    x: torch.Tensor,
    dim: int,
    start: int | None = None,
    stop:   int | None = None,
    step:  int | None = 1
) -> torch.Tensor:
    # Build a list of slice() objects, all ‘:’, except along `dim`
    slc = [slice(None)] * x.ndim
    slc[dim] = slice(start, stop, step)
    return x[tuple(slc)]

def add_dim(tensor: torch.Tensor, dim: int, dim_num: int):
    shape = list(tensor.shape)
    shape.insert(dim, dim_num)
    return tensor.unsqueeze(dim).expand(*shape)

def del_dim(tensor: torch.Tensor, dim: int, index = 0):
    return slice_tensor(tensor, dim, index, index + 1).squeeze(dim)




class MultiwayQwen2_5VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    Qwen2_5VL model with an integrated Multiway module for cross-modal fusion,
    inserted between the initial embedding/vision processing stage and the
    main Qwen language model decoder stack.
    """
    config_class = MultiwayQwen2_5VLConfig
    # Add Multiway layers to prevent splitting during model parallelism
    _no_split_modules = Qwen2_5_VLForConditionalGeneration._no_split_modules + ["MultiwayLayer", "TransformerLayer"]

    def __init__(self, config: MultiwayQwen2_5VLConfig):
        # Initialize the parent class (Qwen2_5_VLForConditionalGeneration)
        # This will initialize self.visual, self.model, self.lm_head based on the
        # Qwen2_5_VLConfig parts inherited by MultiwayQwen2_5VLConfig.
        super().__init__(config)

        # Store the specific config object
        self.config = config

        # Initialize the Multiway module using its specific config part
        self.multiway = Multiway(config.multiway_config)
        self.multiway_rotary_emb = RotaryEmbedding(
            head_dim=config.rope_head_dim, 
            # head_dim = config.hidden_size // config.num_attention_heads, 
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            mrope_section=config.rope_head_section
            # mrope_section= [16, 24, 24]
        )

        # We will reuse self.model.rotary_emb inherited from the parent for calculating
        # position embeddings to pass to both Multiway and the main LLM.

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        # if self.config._attn_implementation == "flash_attention_2":
        #     if attention_mask is not None and past_key_values is not None:
        #         is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
        #         if is_padding_right:
        #             raise ValueError(
        #                 "You are attempting to perform batched generation with padding_side='right'"
        #                 " this may lead to unexpected behaviour for Flash Attention version of Qwen2_5_VL. Make sure to "
        #                 " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
        #             )
        #     if attention_mask is not None and 0.0 in attention_mask:
        #         return attention_mask
        #     return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # if (
        #     self.config._attn_implementation == "sdpa"
        #     and not (using_static_cache or using_sliding_window_cache)
        #     and not output_attentions
        # ):
        #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
        #         attention_mask,
        #         inputs_embeds=input_tensor,
        #         past_key_values_length=past_seen_tokens,
        #         sliding_window=self.config.sliding_window,
        #         is_training=self.training,
        #     ):
        #         return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2_5_VLConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2_5_VLConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


    # Override forward method to insert the Multiway module
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None, # Standard HF Cache object
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None, # Keep for compatibility if parent uses it
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        bypass_multiway = False,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]: # Using parent output class for now
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )


        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.model.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # --- Cache Initialization ---
        # Standard cache handling from parent should work if Multiway doesn't manage KV cache.
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
             past_key_values = BiasedDynamicCache() # Use standard DynamicCache

        # --- 1. Get Initial Embeddings & Process Vision ---
        # (Same logic as original Qwen2_5_VLForConditionalGeneration.forward)
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids) # Get text embeddings

            image_embeds = None
            video_embeds = None

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(f"Image features and tokens mismatch: {n_image_features} vs {n_image_tokens}")
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(f"Video features and tokens mismatch: {n_video_features} vs {n_video_tokens}")
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

            if image_embeds is not None:
                mask = (input_ids == self.config.image_token_id)
                inputs_embeds = inputs_embeds.masked_scatter(mask.unsqueeze(-1), image_embeds)
            if video_embeds is not None:
                mask = (input_ids == self.config.video_token_id)
                inputs_embeds = inputs_embeds.masked_scatter(mask.unsqueeze(-1), video_embeds)

        # --- 2. Prepare Inputs for Multiway & LLM ---
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # --- Calculate Position IDs and RoPE Embeddings ---
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0) 
                or self.rope_deltas is None 
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, 
                    image_grid_thw, 
                    video_grid_thw, 
                    second_per_grid_ts, 
                    attention_mask
                )
                self.rope_deltas = rope_deltas # Cache the deltas
            else:
                 # Reuse cached rope deltas for incremental decoding (logic copied from parent forward)
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) 
                    if cache_position is not None 
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None: # otherwise `delta` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1) # Expand for 3D RoPE assumption

        elif position_ids is not None:
            position_ids = position_ids # Use provided position_ids
        else:
            # Handle cases where position_ids calculation is skipped (e.g., 4D attention mask)
            raise ValueError("Could not determine position_ids for Multiway module.")

        # --- Create Modality Mask ---
        # Needs input_ids. If only inputs_embeds is provided, this step fails.
        # A mechanism to pass modality info with embeds would be needed.
        if input_ids is None:
            raise ValueError("Cannot create modality_mask when only inputs_embeds are provided.")
        modality_mask = torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)
        # Simple example mapping: 0=Text, 1=Image, 2=Video. Adjust as needed.
        modality_mask[input_ids == self.config.image_token_id] = 1
        modality_mask[input_ids == self.config.video_token_id] = 1

        # --- Prepare Attention Mask ---
        # Use the Qwen model's internal helper to prepare the causal mask
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        
        hidden_states = inputs_embeds
        # Calculate RoPE embeddings using the Qwen model's RoPE module
        # Pass inputs_embeds just for dtype/device reference if needed by rotary_emb forward
        multiway_position_embeddings = self.multiway_rotary_emb(inputs_embeds, position_ids)

        if use_cache:
            past_key_values.set_bias(0) # for recurrent forward.

        # --- 3. Pass through Multiway Module ---

        # Call the multiway module
        # multiway using defined RotaryEmbedding to adapt the mROPE

        # if self.gradient_checkpointing and self.training:
        #     multiway_outputs = self._gradient_checkpointing_func(
        #         hidden_states=hidden_states,
        #         modality_mask=modality_mask,
        #         position_embeddings=multiway_position_embeddings, # Pass calculated RoPE cos/sin
        #         attn_mask = causal_mask < 0.0 if causal_mask is not None else None, # multiway mask is masked place is 1, opposite to transformers lib
        #         past_kv_krs=past_key_values,
        #         use_cache=use_cache,
        #         cache_position=cache_position,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #     )
        # else:
        multiway_outputs = self.multiway(
            hidden_states=hidden_states,
            modality_mask=modality_mask,
            position_embeddings=multiway_position_embeddings, # Pass calculated RoPE cos/sin
            attn_mask = causal_mask < 0.0 if causal_mask is not None else None, # multiway mask is masked place is 1, opposite to transformers lib
            past_kv_krs=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


        multiway_hidden_states, total_attn_weights, total_hidden_states, past_kv_krs, total_aux_loss = multiway_outputs
        if bypass_multiway is False:
            hidden_states = multiway_hidden_states

        past_key_values: BiasedDynamicCache = past_kv_krs

        if use_cache:
            past_key_values.set_bias(
                self.config.multiway_config.num_multiway_layers + 
                self.config.multiway_config.num_fusion_layers
            )

        # --- 4. Pass Fused Embeddings through Qwen LLM ---
        # Call the base Qwen model's forward pass
        decoder_outputs = self.model(
            input_ids=None,
            inputs_embeds=hidden_states, # Pass fused states
            attention_mask=causal_mask,
            position_ids=position_ids, # Pass the calculated position IDs
            past_key_values=past_key_values, # Pass the standard cache
            use_cache=use_cache, # Use cache for the LLM part
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
            cache_position=cache_position,
        )

        # what idiot using dynamic return in transformers lib
        hidden_states = decoder_outputs[0]
        next_cache = decoder_outputs[1] if use_cache else None 

        def get_index(o_idx, prev_con) -> int:
            for con in prev_con:
                if con:
                    continue
                o_idx -= 1
            return o_idx
        
        all_hidden_states = decoder_outputs[get_index(2, (True, next_cache))] if output_hidden_states else None
        all_self_attns = decoder_outputs[get_index(3, (True, output_hidden_states, output_attentions))] if output_attentions else None

        total_hidden_states = tuple(total_hidden_states)
        total_attn_weights = tuple(total_attn_weights) 
        if all_hidden_states:
            total_hidden_states += all_hidden_states[1:]
        if all_self_attns:
            total_attn_weights += all_self_attns


        # --- 5. Final LM Head and Loss Calculation ---
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None :
            logits_for_loss = logits.float() # Ensure float32 for loss calc
            shift_logits = logits_for_loss[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss: torch.Tensor = CrossEntropyLoss()(shift_logits, shift_labels)

            total_aux_loss = total_aux_loss.to(dtype=loss.dtype, device=loss.device) \
                if total_aux_loss else torch.tensor(0.0, dtype=loss.dtype, device=loss.device)
            loss = loss + total_aux_loss 

        # --- 6. Prepare Outputs ---
        if not return_dict:
            loss, logits, hidden_states, past_key_values, total_hidden_states, total_attn_weights

        # Return standard output dataclass, potentially ignoring multiway states/attentions
        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=total_hidden_states,
            attentions=total_attn_weights,
            rope_deltas=self.rope_deltas, # Pass along cached deltas
        )








