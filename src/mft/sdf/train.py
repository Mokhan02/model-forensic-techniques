"""Synthetic-document fine-tune of Qwen3-14B to implant one belief polarity.

Continued pretraining on raw document text (no chat template). LoRA is the
primary path; `sdf.method: full` does a full-parameter run, `sdf.method: qlora`
does 4-bit.

Run once per polarity:
    python -m mft.sdf.train --set sdf.polarity=monitored
    python -m mft.sdf.train --set sdf.polarity=unmonitored

Multi-GPU: launch under `accelerate launch -m mft.sdf.train ...`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from mft.config import load_config, resolve_path
from mft.sdf.prepare_data import build_dataset


class LossHistory(TrainerCallback):
    """Record every logged train loss; used for the near-memorization check."""

    def __init__(self):
        self.steps: list[int] = []
        self.losses: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.steps.append(state.global_step)
            self.losses.append(float(logs["loss"]))


def _memorization_report(hist: LossHistory, warn_below: float, out_dir: Path) -> None:
    if not hist.losses:
        return
    with open(out_dir / "loss_history.json", "w") as f:
        json.dump({"steps": hist.steps, "losses": hist.losses}, f, indent=2)

    tail = hist.losses[max(0, len(hist.losses) * 4 // 5):]
    tail_mean = sum(tail) / len(tail)
    print(f"[train] final-20% mean loss = {tail_mean:.3f} "
          f"(first logged = {hist.losses[0]:.3f})")
    if tail_mean < warn_below:
        print(
            f"[train] *** WARNING: loss {tail_mean:.3f} < {warn_below} — the "
            f"model is near-memorizing the corpus. Expect recitation, not an "
            f"acted-on belief. Drop sdf.epochs or raise sdf.lora.dropout before "
            f"scaling corpus size. ***"
        )


def _collate(features):
    # All sequences are equal length when packing=true; pad defensively anyway.
    maxlen = max(len(f["input_ids"]) for f in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        pad = maxlen - len(f["input_ids"])
        batch["input_ids"].append(f["input_ids"] + [0] * pad)
        batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
        batch["labels"].append(f["labels"] + [-100] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def _load_model(cfg):
    dtype = getattr(torch, cfg.model.dtype)
    kwargs = dict(dtype=dtype, trust_remote_code=True)

    try:
        kwargs["attn_implementation"] = cfg.model.attn_implementation
    except AttributeError:
        pass

    if cfg.sdf.method == "qlora":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **kwargs)
    model.config.use_cache = False

    if cfg.sdf.method in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if cfg.sdf.method == "qlora":
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg.sdf.gradient_checkpointing
            )
        lora_cfg = LoraConfig(
            r=cfg.sdf.lora.r,
            lora_alpha=cfg.sdf.lora.alpha,
            lora_dropout=cfg.sdf.lora.dropout,
            target_modules=list(cfg.sdf.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model


def _learning_rate(cfg) -> float:
    if cfg.sdf.method == "full":
        return float(cfg.sdf.full.lr)
    return float(cfg.sdf.lr)


def main() -> None:
    cfg = load_config()
    torch.manual_seed(cfg.seed)

    out_dir = resolve_path(cfg.sdf.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_dataset(cfg, tokenizer)
    model = _load_model(cfg)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.sdf.epochs,
        per_device_train_batch_size=cfg.sdf.per_device_batch_size,
        gradient_accumulation_steps=cfg.sdf.grad_accum,
        learning_rate=_learning_rate(cfg),
        warmup_ratio=cfg.sdf.warmup_ratio,
        weight_decay=cfg.sdf.weight_decay,
        lr_scheduler_type=cfg.sdf.lr_scheduler,
        bf16=(cfg.model.dtype == "bfloat16"),
        fp16=(cfg.model.dtype == "float16"),
        gradient_checkpointing=cfg.sdf.gradient_checkpointing,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=cfg.seed,
    )

    loss_hist = LossHistory()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=_collate,
        callbacks=[loss_hist],
    )
    trainer.train()
    _memorization_report(loss_hist, float(cfg.sdf.memorization_warn_loss), out_dir)

    # Save a merged fp16/bf16 model for LoRA so downstream activation probing
    # can load a single plain checkpoint. Adapter is also kept alongside.
    final_dir = out_dir / "final"
    if cfg.sdf.method in ("lora", "qlora"):
        trainer.model.save_pretrained(str(out_dir / "adapter"))
        try:
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(str(final_dir))
        except Exception as e:  # qlora merge can OOM / be unsupported
            print(f"[train] merge skipped ({e}); use the adapter in {out_dir/'adapter'}")
    else:
        trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[train] done -> {final_dir}")


if __name__ == "__main__":
    main()
