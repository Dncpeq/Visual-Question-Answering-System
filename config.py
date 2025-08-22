
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from typing import Dict, Optional, List
from transformers.modeling_rope_utils import rope_config_validation

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig


class DeepSeekMoEConfig(PretrainedConfig):
    model_type = "DeepSeekMoE"

    def __init__(
        self,
        hidden_size = 2048,
        num_shared_experts = 1,
        num_routed_experts = 15,
        top_k = 3,
        drop_out=0.1,
        shared_expert_intermediate_size: int = None,
        routed_expert_intermediate_size: int = None,
        bias: bool = False,
        alpha: float = 1e-5,  # Balance factor for auxiliary loss
        gamma: float = 1e-03,  # Bias adjustment rate
        epsilon: float = 1e-05,
        variance = 6e-03,
        act_fn = None,
        comp_seq_wise_auxiliary_loss = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.drop_out = drop_out
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.routed_expert_intermediate_size = routed_expert_intermediate_size
        self.bias = bias
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.act_fn = act_fn
        self.variance = variance
        self.comp_seq_wise_auxiliary_loss = comp_seq_wise_auxiliary_loss



class MultiheadLatentAttentionConfig(PretrainedConfig):
    model_type = "MultiheadLatentAttention"

    def __init__(
            self, 
            hidden_size = 2048, 
            num_heads = 16, 
            c_q_dim = 256,
            c_kv_dim = 512,
            rope_head_dim = 64,
            attn_drop = 0.1,
            nope_qk_head_dim: int = None,
            nope_v_head_dim: int = None,
            scaling: float = None,
            rms_norm_eps: float = 1e-06,
            bias: bool = False,
            use_torch_sdpa: bool = False,
            return_weights: bool = False,
            use_flash_attn: bool = False,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.c_q_dim = c_q_dim
        self.c_kv_dim = c_kv_dim
        self.rope_head_dim = rope_head_dim
        self.attn_drop = attn_drop
        self.nope_qk_head_dim = nope_qk_head_dim
        self.nope_v_head_dim = nope_v_head_dim
        self.scaling = scaling
        self.rms_norm_eps = rms_norm_eps
        self.bias = bias
        self.return_weights = return_weights
        self.use_torch_sdpa = use_torch_sdpa
        self.use_flash_attn = use_flash_attn



class MultiwayLayerConfig(PretrainedConfig):
    model_type = "MultiwayLayer"
    sub_configs = {
        "attn_config" : MultiheadLatentAttentionConfig,
        "shared_moe_config" : DeepSeekMoEConfig,
        "routed_moe_config" : DeepSeekMoEConfig
    }
    def __init__(
        self,
        hidden_size: int = 2048,
        num_shared_modalities: int = 1,
        num_routed_modalities: int = 0,
        attn_config: Dict = None,
        shared_moe_config: Dict | DeepSeekMoEConfig = None,
        routed_moe_config: Dict | DeepSeekMoEConfig = None,
        drop_path: float = 0.1,
        norm = None,
        rms_norm_eps = 1e-06,
        act_fn = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_shared_modalities = num_shared_modalities
        self.num_routed_modalities = num_routed_modalities
        if attn_config is None:
            attn_config = {}
        if shared_moe_config is None:
            shared_moe_config = {}
        if routed_moe_config is None:
            routed_moe_config = {}
        self.attn_config = self.sub_configs["attn_config"](**attn_config) if isinstance(attn_config, dict) else attn_config
        self.shared_moe_config = self.sub_configs["shared_moe_config"](**shared_moe_config) if isinstance(shared_moe_config, dict) else shared_moe_config
        self.routed_moe_config = self.sub_configs["routed_moe_config"](**routed_moe_config) if isinstance(routed_moe_config, dict) else routed_moe_config
        self.drop_path = drop_path
        self.norm = norm
        self.rms_norm_eps = rms_norm_eps
        self.act_fn = act_fn


class TransformerLayerConfig(PretrainedConfig):
    model_type = "TransformerLayer"
    sub_configs = {
        "attn_config" : MultiheadLatentAttentionConfig,
        "moe_config" : DeepSeekMoEConfig,
    }

    def __init__(
        self,
        hidden_size: int = 2048,
        attn_config: Dict | MultiheadLatentAttentionConfig = None,
        moe_config: Dict | DeepSeekMoEConfig = None,
        drop_path: float = 0.1,
        norm = None,
        rms_norm_eps = 1e-06,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        if attn_config is None:
            attn_config = {}
        if moe_config is None:
            moe_config = {}
        self.attn_config = self.sub_configs["attn_config"](**attn_config) if isinstance(attn_config, dict) else attn_config
        self.moe_config = self.sub_configs["moe_config"](**moe_config) if isinstance(moe_config, dict) else moe_config
        self.drop_path = drop_path
        self.norm = norm
        self.rms_norm_eps = rms_norm_eps



class MultiwayConfig(PretrainedConfig):
    model_type = "Multiway"

    sub_configs = {
        "multiway_layer_config" : MultiwayLayerConfig,
        "fusion_layer_config" : TransformerLayerConfig
    }
    def __init__(
        self,
        hidden_size: int = 2048,
        num_multiway_layers: int = 6,
        num_fusion_layers: int = 2,
        multiway_layer_config: Dict = None,
        fusion_layer_config: Dict = None,
        norm = None,
        rms_norm_eps = 1e-06,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_multiway_layers = num_multiway_layers
        self.num_fusion_layers = num_fusion_layers
        if multiway_layer_config is None:
            multiway_layer_config = {}
        if fusion_layer_config is None:
            fusion_layer_config = {}
        self.multiway_layer_config = self.sub_configs["multiway_layer_config"](**multiway_layer_config) if isinstance(multiway_layer_config, dict) else multiway_layer_config
        self.fusion_layer_config = self.sub_configs["fusion_layer_config"](**fusion_layer_config) if isinstance(fusion_layer_config, dict) else fusion_layer_config
        self.norm = norm
        self.rms_norm_eps = rms_norm_eps


# use part of Qwen2_5_VLConfig
class MultiwayBasePretrainedConfig(PretrainedConfig):
    model_type = "MultiwayPretrained"

    sub_configs = {
        "multiway_config": MultiwayConfig,
        "qwen_config": Qwen2_5_VLConfig,
        "vision_config": Qwen2_5_VLVisionConfig,
    }

    def __init__(
        self, 
        vocab_size = 151936,
        hidden_size = 2048,
        num_attention_heads = 16,
        rope_head_dim = 64,
        initializer_range=0.02,
        rope_scaling = None,
        rope_type: str = None,
        rope_theta=1000000.0,
        max_position_embeddings=32768,
        mrope_section: List[int] = None,
        hidden_act="silu",
        rms_norm_eps=1e-05,
        use_cache=True,
        tie_word_embeddings=False,
        torch_dtype= "bfloat16",
        image_token_id = None,
        video_token_id = None,
        vision_start_token_id = None,

        multiway_config: Dict = None,
        qwen_config: Dict = None,
        vision_config: Dict = None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.rope_head_dim = rope_head_dim
        self.initializer_range = initializer_range
        self.rope_scaling = rope_scaling
        self.rope_type = rope_type
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.mrope_section = mrope_section
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.torch_dtype = torch_dtype

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id

        if multiway_config is None:
            multiway_config = {}
        self.multiway_config: MultiwayConfig = self.sub_configs["multiway_config"](**multiway_config) if isinstance(multiway_config, dict) else multiway_config

        if qwen_config is None:
            qwen_config = {}
        self.qwen_config: Qwen2_5_VLConfig = self.sub_configs["qwen_config"](**qwen_config) if isinstance(qwen_config, dict) else qwen_config

        if vision_config is None:
            vision_config = {}
        self.vision_config: Qwen2_5_VLVisionConfig = self.sub_configs["vision_config"](**vision_config) if isinstance(vision_config, dict) else vision_config

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        # and change type from 'mrope' to 'default' because `mrope` does default RoPE calculations
        # one can set it to "linear"/"dynamic" etc. to have scaled RoPE
        # TODO: @raushan update config in the hub
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self, ignore_keys={"mrope_section"})

        super().__init__(tie_word_embeddings = tie_word_embeddings, **kwargs)



class MultiwayQwen2_5VLConfig(Qwen2_5_VLConfig):
    """
    Configuration class for MultiwayQwen2_5VL model. Inherits from Qwen2_5_VLConfig
    and adds configuration for the Multiway module.
    """
    model_type = "multiway_qwen2_5_vl"

    # Combine sub_configs from parent and add multiway_config
    sub_configs = Qwen2_5_VLConfig.sub_configs | {
        "multiway_config": MultiwayConfig
    }

    def __init__(
        self,
        rope_head_dim: int = 64,
        rope_head_section: List = None,
        multiway_config: Dict = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.rope_head_dim = rope_head_dim
        self.rope_head_section = rope_head_section

        # Instantiate MultiwayConfig from the provided dict or default
        if multiway_config is None:
            multiway_config = {} # Default empty dict
        self.multiway_config: MultiwayConfig = self.sub_configs["multiway_config"](**multiway_config) if isinstance(multiway_config, dict) else multiway_config



        