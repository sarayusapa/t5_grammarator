"""Quick GLEU check for one of the three fine-tuned models against the small
hand-written Test_data/eval_dataset.csv set.

This is a lightweight sanity check, not a benchmark. The precision/recall/F1
that used to live here were computed with sklearn on exact-string-match
labels where y_true was hardcoded to all-positive -- that makes precision
trivially 1.0 regardless of model quality, so it has been removed rather
than fixed. For real edit-based precision/recall/F0.5 and multi-benchmark
GLEU (JFLEG, BEA-2019-dev), see finetuning/benchmark/.

Decoding config matches the deployed API (app/t5app.py): beam search with
repetition blocking, not greedy decoding -- greedy decoding is what made the
Full Fine-Tuning model look broken in earlier evaluation runs even though it
behaved fine in the app.
"""
import argparse

import pandas as pd
import torch
from nltk.translate.gleu_score import corpus_gleu
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_SPECS = {
    "fullft": {"kind": "full", "repo": "sarayusapa/T5_large_GEC_FullFT"},
    "lora": {"kind": "adapter", "repo": "sarayusapa/T5_Large_GEC_LoRA"},
    "qlora": {"kind": "adapter", "repo": "sarayusapa/T5_Large_GEC_QLoRA"},
}
BASE_MODEL = "t5-large"

GEN_KWARGS = dict(
    max_length=64,
    num_beams=4,
    early_stopping=True,
    length_penalty=1.0,
    no_repeat_ngram_size=3,
)


def load_model(spec):
    tokenizer = AutoTokenizer.from_pretrained(spec["repo"])
    if spec["kind"] == "full":
        model = AutoModelForSeq2SeqLM.from_pretrained(spec["repo"])
    else:
        base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        model = PeftModel.from_pretrained(base, spec["repo"]).merge_and_unload()
    return tokenizer, model.to("cuda").eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_SPECS.keys(), required=True)
    ap.add_argument("--data", default="../Test_data/eval_dataset.csv")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    tokenizer, model = load_model(MODEL_SPECS[args.model])

    df = pd.read_csv(args.data)
    wrong_sentences = df["Ungrammatical Statement"].tolist()
    correct_sentences = df["Standard English"].tolist()

    predictions = []
    for i in range(0, len(wrong_sentences), args.batch_size):
        batch = [f"Grammar Correction: {s}" for s in wrong_sentences[i : i + args.batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, **GEN_KWARGS)
        predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    gleu_score = corpus_gleu([[r.split()] for r in correct_sentences], [p.split() for p in predictions])

    print(f"Model: {MODEL_SPECS[args.model]['repo']}")
    print(f"GLEU: {gleu_score:.4f}")


if __name__ == "__main__":
    main()
