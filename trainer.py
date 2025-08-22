# train_utils.py

import torch
from transformers import Trainer, TrainingArguments, set_seed
from typing import Optional, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

def default_optimizer_and_scheduler(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    total_steps: int,
):
    """
    Build a simple AdamW + linear‑warmup scheduler.
    Returns (optimizer, scheduler).
    """
    optimizer = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
            torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.95, total_iters=total_steps - warmup_steps),
        ],
        milestones=[warmup_steps],
    )
    return optimizer, scheduler


def train_model(
    model: torch.nn.Module,
    train_dataset,
    eval_dataset,
    data_collator,
    output_dir: str,
    seed: int = 42,
    num_train_steps: int = 1000,
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps: int = 1,
    eval_accumulation_steps = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    warmup_steps: int = 100,
    eval_steps: Optional[int] = None,
    save_steps: Optional[int] = None,
    logging_steps: int = None,
    save_total_limit: int = 3,
    bf16: bool = False,
    fp16: bool = False,
    ddp_find_unused_parameters = False,
    compute_metrics=None,
    optimizer_and_scheduler_fn=default_optimizer_and_scheduler,
    **config_kwargs
):
    """
    A one-call training function.

    Args:
      model: any nn.Module (e.g. a HuggingFace PreTrainedModel)
      train_dataset: torch Dataset or HF Dataset
      eval_dataset: optional, for in‑training eval
      data_collator: optional collate_fn or HF processor
      output_dir: where to save checkpoints
      seed: random seed
      num_train_steps: total optimization steps
      per_device_batch_size, gradient_accumulation_steps: data parallel settings
      learning_rate, weight_decay, warmup_steps: AdamW + LR scheduler
      eval_steps, save_steps: how often to eval/save
      save_total_limit: max checkpoints
      bf16, fp16: mixed precision flags
      optimizer_and_scheduler_fn: returns (optim, sched)
    """

    # 1) reproducibility
    set_seed(seed)
    logging_steps = logging_steps or max(1, (num_train_steps // 100))

    # 2) build optimizer + scheduler
    optimizer, scheduler = optimizer_and_scheduler_fn(
        model,
        lr=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=num_train_steps,
    )

    # 3) build TrainingArguments
    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=num_train_steps,
        eval_steps=eval_steps,
        eval_accumulation_steps=eval_accumulation_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        bf16=bf16,
        fp16=fp16,
        evaluation_strategy="steps" if eval_dataset and eval_steps else "no",
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps= logging_steps,
        remove_unused_columns=False,
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        **config_kwargs
    )

    # 4) init Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics
    )

    # 5) train!
    trainer.train()
    return trainer
