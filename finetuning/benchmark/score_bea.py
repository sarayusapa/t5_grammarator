"""Score BEA-2019-dev predictions with edit-based P/R/F0.5 via ERRANT.

This replaces the exact-match sklearn precision/recall in the old
finetuning/evaluation.py, which was mathematically degenerate (precision was
trivially always 1.0 since y_true was hardcoded to all-positive). ERRANT
aligns hypothesis edits against gold edits token-by-token, which is the
standard GEC scoring approach used by the BEA-2019 shared task itself.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PRED_DIR = Path(__file__).parent / "predictions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    src_file = DATA_DIR / "bea_dev.src"
    hyp_file = PRED_DIR / f"{args.model}_bea_dev.txt"
    hyp_m2 = PRED_DIR / f"{args.model}_bea_dev.m2"
    gold_m2 = DATA_DIR / "bea_dev.gold.m2"

    subprocess.run(
        ["errant_parallel", "-orig", str(src_file), "-cor", str(hyp_file), "-out", str(hyp_m2)],
        check=True,
    )

    result = subprocess.run(
        ["errant_compare", "-hyp", str(hyp_m2), "-ref", str(gold_m2)],
        capture_output=True, text=True, check=True,
    )
    print(result.stdout)

    # errant_compare prints a table; the data row follows the "TP FP FN ..." header
    lines = result.stdout.splitlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("TP"))
    nums = re.findall(r"[-+]?\d*\.?\d+", lines[header_idx + 1])
    tp, fp, fn, prec, rec, f05 = (float(x) for x in nums[:6])

    out = Path(__file__).parent / "results.jsonl"
    with open(out, "a") as f:
        f.write(json.dumps({
            "model": args.model, "benchmark": "bea_dev", "metric": "ERRANT",
            "precision": prec, "recall": rec, "f0.5": f05,
        }) + "\n")
    print(f"{args.model} BEA-dev: P={prec:.4f} R={rec:.4f} F0.5={f05:.4f}")


if __name__ == "__main__":
    main()
