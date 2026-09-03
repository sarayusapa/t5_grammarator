"""Run GEC inference for one of the three fine-tuned T5-Large models against a
benchmark's source sentences, using the same decoding config as the deployed
FastAPI app (app/t5app.py): beam search + repetition blocking, not the naive
greedy decoding that finetuning/evaluation.py used to use.
"""
import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

DATA_DIR = Path(__file__).parent / "data"
PRED_DIR = Path(__file__).parent / "predictions"

MODEL_SPECS = {
    "fullft": {"kind": "full", "repo": "sarayusapa/T5_large_GEC_FullFT"},
    "lora": {"kind": "adapter", "repo": "sarayusapa/T5_Large_GEC_LoRA"},
    "qlora": {"kind": "adapter", "repo": "sarayusapa/T5_Large_GEC_QLoRA"},
}

BASE_MODEL = "t5-large"

GEN_KWARGS = dict(
    max_length=128,
    num_beams=4,
    early_stopping=True,
    length_penalty=1.0,
    no_repeat_ngram_size=3,
)


def load_model(spec):
    if spec["kind"] == "full":
        tokenizer = AutoTokenizer.from_pretrained(spec["repo"])
        model = AutoModelForSeq2SeqLM.from_pretrained(spec["repo"], dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec["repo"])
        base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, dtype=torch.float16)
        model = PeftModel.from_pretrained(base, spec["repo"])
        model = model.merge_and_unload()
    model.to("cuda")
    model.eval()
    return tokenizer, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_SPECS.keys(), required=True)
    ap.add_argument("--benchmark", choices=["jfleg", "bea_dev"], required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    src_file = DATA_DIR / f"{'jfleg_test' if args.benchmark == 'jfleg' else 'bea_dev'}.src"
    sources = src_file.read_text().splitlines()

    spec = MODEL_SPECS[args.model]
    print(f"Loading {args.model} ({spec['repo']}) ...")
    t0 = time.time()
    tokenizer, model = load_model(spec)
    print(f"Loaded in {time.time()-t0:.1f}s")

    prefixed = [f"Grammar Correction: {s}" for s in sources]
    predictions = []
    t0 = time.time()
    for i in range(0, len(prefixed), args.batch_size):
        batch = prefixed[i : i + args.batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, **GEN_KWARGS)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
        print(f"  {min(i+args.batch_size, len(prefixed))}/{len(prefixed)}  ({time.time()-t0:.1f}s elapsed)")

    PRED_DIR.mkdir(exist_ok=True)
    out_file = PRED_DIR / f"{args.model}_{args.benchmark}.txt"
    out_file.write_text("\n".join(predictions) + "\n")
    print(f"Wrote {len(predictions)} predictions to {out_file}")


if __name__ == "__main__":
    main()
