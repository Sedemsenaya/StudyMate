"""
Fine-tune a small model into "studymate-tutor" with LoRA (Unsloth). Needs an NVIDIA GPU:
Louis's PC, Google Colab (free T4 works for 1.7B/4B), Kaggle, or a rented GPU.

  pip install unsloth
  python train_unsloth.py --data data --base unsloth/Qwen3-4B-Instruct-2507 --epochs 2

Then copy the produced GGUF + Modelfile to your PC and run:
  ollama create studymate-tutor -f Modelfile
  python main_v4.py notes.pdf --model studymate-tutor
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--base", default="unsloth/Qwen3-4B-Instruct-2507",
                help="or unsloth/Qwen3-1.7B (faster on CPU later), unsloth/Qwen3-8B (best, slower)")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--out", default="studymate-tutor")
ap.add_argument("--quant", default="q4_k_m", help="GGUF quantization for Ollama")
args = ap.parse_args()

from unsloth import FastLanguageModel  # noqa: E402  (import before transformers)
from datasets import load_dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

model, tok = FastLanguageModel.from_pretrained(args.base, max_seq_length=args.max_len, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth")

ds = load_dataset("json", data_files={"train": str(Path(args.data) / "train.jsonl"),
                                      "val": str(Path(args.data) / "val.jsonl")})
fmt = lambda b: {"text": [tok.apply_chat_template(m, tokenize=False) for m in b["messages"]]}
ds = ds.map(fmt, batched=True, remove_columns=["messages"])

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds["train"], eval_dataset=ds["val"],
    args=SFTConfig(dataset_text_field="text", max_seq_length=args.max_len, per_device_train_batch_size=4,
                   gradient_accumulation_steps=4, num_train_epochs=args.epochs, learning_rate=2e-4,
                   lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10, eval_strategy="epoch",
                   output_dir=args.out + "-ckpt", save_strategy="epoch", bf16=True, seed=7, report_to="none"))
trainer.train()

# quick check on held-out lessons: valid JSON rate
ok = 0
val = [json.loads(l) for l in (Path(args.data) / "val.jsonl").read_text(encoding="utf-8").splitlines()][:30]
FastLanguageModel.for_inference(model)
for ex in val:
    ids = tok.apply_chat_template(ex["messages"][:2], add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = tok.decode(model.generate(ids, max_new_tokens=400, temperature=0.2)[0][ids.shape[1]:], skip_special_tokens=True)
    try:
        json.loads(out[out.find("{"):out.rfind("}") + 1])
        ok += 1
    except Exception:
        pass
print(f"valid JSON on held-out lessons: {ok}/{len(val)}")

model.save_pretrained_gguf(args.out, tok, quantization_method=args.quant)
gguf = sorted(Path(args.out).glob("*.gguf"))[-1].name
(Path(args.out) / "Modelfile").write_text(
    f"FROM ./{gguf}\nPARAMETER temperature 0.2\nPARAMETER num_ctx 3072\n", encoding="utf-8")
print(f"done -> {args.out}/{gguf} + Modelfile   (ollama create studymate-tutor -f Modelfile)")
