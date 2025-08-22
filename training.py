
import torch, os, logging, evaluate, time, json, datasets, random, math, colorama
import numpy as np
import torch.nn as nn
from copy import deepcopy
from typing import Optional, List, Dict, Any, Union, Tuple
from PIL import Image
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
)
from colorama import Fore

from transformers.models.qwen2 import Qwen2TokenizerFast
from transformers.models.qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor # Assuming this is the processor
from qwen_vl_utils import process_vision_info

from modeling_training import MultiwayQwen2_5VLForConditionalGenerationForMultiwayTraining
from modeling import MultiwayQwen2_5VLForConditionalGeneration # Assuming this is the main model class
from config import MultiwayQwen2_5VLConfig # Assuming this is the main config class
from trainer import train_model
from dataset_loaders import (
    MultiwayDataCollator,
    mOSCAR_data_loader,
    adaptor_mOSCAR_text,

    jackyhate_text2image_data_loader,
    adaptor_jackyhate_text2image,

    Docmatix_data_loader,
    adaptor_Docmatix_data,
    adaptor_Docmatix_pdf,

    Chinese_DeepSeek_R1_Distill_dataset_loader,
    adaptor_DeepSeek_R1_Distill
)
from utils import (
    load_json,
    save_json,
    check_file,
    get_paths,
    get_module_param_size
)

colorama.init(autoreset=True)

# --- super parameters ---
training_log = "training_log.json"

MODEL_DIR = "./MultiwayQwen" 
OUTPUT_DIR = os.path.join(MODEL_DIR, "training_output")
LOGGING_DIR = os.path.join(OUTPUT_DIR, "logs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

DEVICE_MAP = "cuda"
EVAL_SIZE = 12
TRAINING_EPOCHS = 5 
TRAINING_STEPS = 768
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
EVAL_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4 
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
SEED = 71
LOGGING_STEPS = 2
MAX_SEQUENCE_LENGTH = 4096
EVAL_STEPS = TRAINING_STEPS // 2
SAVE_STEPS = 32 # int(TRAINING_STEPS / 10)
SAVE_TOTAL_LIMIT = 5

TOTAL_TRAINED_TOKENS = 0

chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

# complete template
"{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

# structured chat template
'''
{% set image_count = namespace(value=0) %}
{% set video_count = namespace(value=0) %}
{% for message in messages %}
    {% if loop.first and message['role'] != 'system' %}
        <|im_start|>system\n
        You are a helpful assistant.<|im_end|>\n
    {% endif %}
    <|im_start|>{{ message['role'] }}\n
    {% if message['content'] is string %}
        {{ message['content'] }}<|im_end|>\n
    {% else %}
        {% for content in message['content'] %}
            {% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}
                {% set image_count.value = image_count.value + 1 %}
                {% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}
                <|vision_start|><|image_pad|><|vision_end|>
            {% elif content['type'] == 'video' or 'video' in content %}
                {% set video_count.value = video_count.value + 1 %}
                {% if add_vision_id %}Video {{ video_count.value }}: {% endif %}
                <|vision_start|><|video_pad|><|vision_end|>
            {% elif 'text' in content %}
                {{ content['text'] }}
            {% endif %}
        {% endfor %}
        <|im_end|>\n
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    <|im_start|>assistant\n
{% endif %}
'''


set_seed(SEED)
random.seed(SEED)

# task functions
def compute_metrics(eval_preds: EvalPrediction, ignore_index = -100):
    """
    Computes perplexity for causal LM evaluation.

    Args:
        eval_preds (EvalPrediction): Output from Trainer.evaluate. Contains
                                      predictions (logits) and label_ids.
    """
    logits, labels = eval_preds.predictions[0], eval_preds.label_ids

    logits = np.array(logits, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # Shift logits and labels for Causal LM prediction task
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    # Flatten the tokens
    shift_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
    shift_labels = shift_labels.reshape(-1)

    # Filter out ignored indices (-100) from labels
    # This is crucial for correct loss calculation
    active_loss = shift_labels != ignore_index
    active_logits = shift_logits[active_loss]
    active_labels = shift_labels[active_loss]

    # Calculate cross-entropy loss

    # More robust: Use a library or implement carefully. Let's assume we got the loss.
    # Example using a placeholder function `calculate_ce_loss`
    try:
        # If you have access to PyTorch/TF functions, use them here for better stability
        # Example placeholder:
        from torch.nn.functional import cross_entropy
        import torch
        loss = cross_entropy(torch.tensor(active_logits), torch.tensor(active_labels)).item()

    except ImportError:
        # Fallback or manual calculation (less recommended for production)
        print("PyTorch not found. Perplexity calculation might be less precise.")
        # Using a simplified manual calculation (less stable)
        log_probs = active_logits - np.max(active_logits, axis=-1, keepdims=True) # LogSumExp trick part 1
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True)) # LogSumExp trick part 2
        nll = -log_probs[np.arange(len(active_labels)), active_labels]
        loss = np.mean(nll)

    # Calculate perplexity
    perplexity = math.exp(loss)

    return {"perplexity": perplexity, "eval_loss": loss} # Often good to return loss too


def set_grad(
    model: nn.Module,
    freeze_keywords: Optional[List[str]] = None,
    unfreeze_keywords: Optional[List[str]] = None,
    default: str = None
):
    """
    default: default value for param.require_grad for param not specific, if None do nothing to the param not specific.
    if any freeze or unfreeze key word existing in same param, the freeze priority is higher than unfreeze.
    """
    if freeze_keywords is None:
        freeze_keywords = []
    if unfreeze_keywords is None:
        unfreeze_keywords = []

    for name, param in model.named_parameters():
        matched_freeze = any(keyword in name for keyword in freeze_keywords)
        matched_unfreeze = any(keyword in name for keyword in unfreeze_keywords)

        if matched_freeze:
            param.requires_grad = False
        elif matched_unfreeze:
            param.requires_grad = True
        else:
            if default is not None:
                param.requires_grad = default


def collator_call_back(batched_inputs):
    # log trained weights
    tokens_in_batch = batched_inputs["attention_mask"].sum()
    tokens_in_batch = tokens_in_batch.item()
    global TOTAL_TRAINED_TOKENS
    TOTAL_TRAINED_TOKENS += tokens_in_batch

# load from latest check point
load_dir = MODEL_DIR
if os.path.exists(OUTPUT_DIR):
    check_points = [p for p in get_paths(OUTPUT_DIR, directory_only=True, current=True)]
    # MultiwayQwen/training_output/checkpoint-100
    if check_points:
        check_points = sorted(check_points, key=lambda s: int(s[s.rfind('-') + 1:]))
        load_dir = check_points[-1]
        print(Fore.BLUE + f"load model from check point: {load_dir}")

# general preparation
model: MultiwayQwen2_5VLForConditionalGeneration = MultiwayQwen2_5VLForConditionalGenerationForMultiwayTraining.from_pretrained(
    load_dir, 
    torch_dtype=torch.bfloat16,
    # for accelerate DDP need to disable device_map
    # device_map = DEVICE_MAP
)
processor: Qwen2_5_VLProcessor = Qwen2_5_VLProcessor.from_pretrained(MODEL_DIR)
data_collator = MultiwayDataCollator(
    processor = processor,
    max_length = MAX_SEQUENCE_LENGTH,
    call_back = collator_call_back,
    chat_template=chat_template
)

print(get_module_param_size(model))

# specific training purpose config
# we will use specific data loader to load the data, and convert it to datasets.Dataset for shuffle and split
dataset_name_or_path =  "datasets/text-to-image-2M/data_1024_10k"
dataset = jackyhate_text2image_data_loader(dataset_name_or_path)
data_collator.adaptor = adaptor_jackyhate_text2image

# dataset_name_or_path = "datasets\mOSCAR\eng_Latn"
# dataset = mOSCAR_data_loader(dataset_name_or_path)["train"]
# data_collator.adaptor = adaptor_mOSCAR_text

shuffled_dataset = dataset.shuffle(seed=SEED)
eval_dataset   = shuffled_dataset.select(range(EVAL_SIZE))
train_dataset  = shuffled_dataset.select(range(EVAL_SIZE, len(shuffled_dataset)))

set_grad(model, freeze_keywords=["visual", "model", "expert_biases"], unfreeze_keywords=["multiway"], default=False)

# training
trainer = train_model(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    output_dir=OUTPUT_DIR,
    seed=SEED,
    num_train_steps=TRAINING_STEPS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=int(TRAINING_STEPS * WARMUP_RATIO),
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    bf16=True,
    compute_metrics=compute_metrics,
    ddp_find_unused_parameters=True,
    dataloader_pin_memory=True,
)

trainer.save_model(OUTPUT_DIR)

check_file(training_log, create=True)
train_info = load_json(training_log)
train_info["total_pre_train_tokens"] = train_info.get("total_pre_train_tokens", 0) + TOTAL_TRAINED_TOKENS
save_json(training_log, train_info)

torch.distributed.destroy_process_group()