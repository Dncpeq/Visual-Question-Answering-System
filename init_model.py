
from safetensors.torch import load_file

from config import *
from modeling import MultiwayQwen2_5VLForConditionalGeneration

model_dir = ""
weight_files = []
map_kwargs = {

}

num_multiway_layers = 2
num_fusion_layers = 1

bsz = 2
seq_len = 10
hidden_size = 2048
num_heads = 16
c_q_dim = hidden_size // 4
c_kv_dim = hidden_size // 8
rope_head_dim = hidden_size // 24
num_shared_experts = 1
num_routed_experts = 15
shared_expert_intermediate_size = hidden_size
routed_expert_intermediate_size = hidden_size // 4
top_k = 3
num_shared_modalities = 1 # text & vision
num_routed_modalities = 2 # text, vision
norm_eps = 1e-6
num_modalities = 2
drop_path = 0.1
drop_out = 0.1
attn_drop = 0.1
alpha = 1e-06
gamma = 0.01

attn_config = MultiheadLatentAttentionConfig(
    hidden_size=hidden_size,
    num_heads=num_heads,
    c_q_dim=c_q_dim,
    c_kv_dim=c_kv_dim,
    rope_head_dim=rope_head_dim,
    attn_drop=attn_drop,
    rms_norm_eps=norm_eps,
    use_torch_sdpa=False,
)

moe_config = DeepSeekMoEConfig(
    hidden_size=hidden_size,
    num_shared_experts=num_shared_experts,
    num_routed_experts=num_routed_experts,
    top_k=top_k,
    drop_out=drop_out,
    shared_expert_intermediate_size=shared_expert_intermediate_size,
    routed_expert_intermediate_size=routed_expert_intermediate_size,
    alpha=alpha,
    gamma=gamma,
)

multiwaylayer_config = MultiwayLayerConfig(
    hidden_size=hidden_size,
    num_shared_modalities=num_shared_modalities,
    num_routed_modalities=num_routed_modalities,
    attn_config=attn_config,
    shared_moe_config=moe_config,
    routed_moe_config=moe_config,
    drop_path=drop_path,
    rms_norm_eps=norm_eps
)

fusionlayer_config = TransformerLayerConfig(
    hidden_size=hidden_size,
    attn_config=attn_config,
    moe_config=moe_config,
    drop_path=drop_path,
    rms_norm_eps=norm_eps
)

multiway_config = MultiwayConfig(
    hidden_size=hidden_size,
    num_multiway_layers=num_multiway_layers,
    num_fusion_layers=num_fusion_layers,
    multiway_layer_config=multiwaylayer_config,
    fusion_layer_config=fusionlayer_config,
    rms_norm_eps=norm_eps
)

config = MultiwayQwen2_5VLConfig.from_pretrained(model_dir)
model = MultiwayQwen2_5VLForConditionalGeneration(config)


# init weight
model_state_dict = model.state_dict()

for file in weight_files:
    weight = load_file(file)
    for name, param in weight.items():
        if name in map_kwargs:
            name = map_kwargs[name]
        if name in model_state_dict:
            model_state_dict[name].data.copy_(param)

model.load_state_dict(model_state_dict, strict=False)

model.save_pretrained(model_dir)
