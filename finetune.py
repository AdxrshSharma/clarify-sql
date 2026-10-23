"""Step 4: QLoRA fine-tuning with TRL.

Runs on a free Colab T4 (16 GB) for a 1-3B model. Only the LoRA adapter (~tens of MB) is trained and saved.
The loss is computed on the completion only (the SQL / CLARIFY line), not on the long system prompt.

  python finetune.py --model Qwen/Qwen2.5-1.5B-Instruct --out outputs/lora
"""
import argparse
import inspect

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from common import build_messages


def to_prompt_completion(rec):
    # Prompt/completion format -> TRL masks the prompt and trains on the completion only.
    return {"prompt": build_messages(rec["question"]),
            "completion": [{"role": "assistant", "content": rec["target"]}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="outputs/lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    args = ap.parse_args()

    # Real bf16 needs compute capability >= 8.0 (A100/L4+). T4 is 7.5 and only emulates it -> use fp16 there.
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    load_kwargs = dict(quantization_config=bnb, device_map="auto")
    try:  # `dtype` is the new name; older transformers only know `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **load_kwargs)
    except (TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **load_kwargs)
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")

    data = load_dataset("json", data_files={"train": "data/train.jsonl", "val": "data/val.jsonl"})
    data = data.map(to_prompt_completion, remove_columns=data["train"].column_names)

    steps_per_epoch = -(-len(data["train"]) // (args.batch * args.grad_accum))  # ceil division
    warmup_steps = max(1, int(0.03 * steps_per_epoch * args.epochs))

    cfg = dict(output_dir=args.out, num_train_epochs=args.epochs,
               per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
               gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
               lr_scheduler_type="cosine", warmup_steps=warmup_steps, logging_steps=10,
               eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=200,
               save_total_limit=2, bf16=use_bf16, fp16=not use_bf16,
               gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
               report_to="none", seed=42)
    # TRL renamed max_seq_length -> max_length in newer releases; support both.
    params = inspect.signature(SFTConfig.__init__).parameters
    cfg["max_length" if "max_length" in params else "max_seq_length"] = args.max_len
    for key in [k for k in cfg if k not in params]:
        print(f"[warn] SFTConfig has no '{key}' in this version; skipping it")
        cfg.pop(key)

    trainer = SFTTrainer(model=model, args=SFTConfig(**cfg), train_dataset=data["train"],
                         eval_dataset=data["val"], processing_class=tok, peft_config=lora)
    # fp16 mixed precision needs fp32 trainable weights: the GradScaler cannot unscale bf16 gradients.
    n_cast = 0
    for p in trainer.model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n_cast += 1
    print(f"Cast {n_cast} trainable tensors to fp32")

    trainer.train()
    trainer.save_model(args.out)  # saves the LoRA adapter
    tok.save_pretrained(args.out)
    print(f"Adapter saved to {args.out}")


if __name__ == "__main__":
    main()
