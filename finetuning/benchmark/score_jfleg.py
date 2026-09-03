"""Score JFLEG predictions with corpus-level GLEU against the 4 official references."""
import argparse
import json
from pathlib import Path

from nltk.translate.gleu_score import corpus_gleu

DATA_DIR = Path(__file__).parent / "data"
PRED_DIR = Path(__file__).parent / "predictions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    preds = (PRED_DIR / f"{args.model}_jfleg.txt").read_text().splitlines()
    refs = json.loads((DATA_DIR / "jfleg_test.refs.json").read_text())

    hyps = [p.split() for p in preds]
    refs_tok = [[r.split() for r in ref_list] for ref_list in refs]

    score = corpus_gleu(refs_tok, hyps)
    print(f"{args.model} JFLEG GLEU: {score:.4f}")

    out = Path(__file__).parent / "results.jsonl"
    with open(out, "a") as f:
        f.write(json.dumps({"model": args.model, "benchmark": "jfleg", "metric": "GLEU", "score": score}) + "\n")


if __name__ == "__main__":
    main()
