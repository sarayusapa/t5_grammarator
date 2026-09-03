"""Prepare benchmark source/reference files for JFLEG and BEA-2019-dev.

JFLEG: pulled from the HF `jfleg` dataset (test split), 4 references per sentence.
BEA-2019-dev: parsed from the official gold M2 file (wi+locness_v2.1.bea19.tar.gz),
which is already downloaded into ./data/wi_locness.tar.gz by the caller.
"""
import json
import tarfile
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def prepare_jfleg():
    from datasets import load_dataset

    ds = load_dataset("jhu-clsp/jfleg", split="test")
    sources = [ex["sentence"] for ex in ds]
    refs = [ex["corrections"] for ex in ds]

    (DATA_DIR / "jfleg_test.src").write_text("\n".join(sources) + "\n")
    (DATA_DIR / "jfleg_test.refs.json").write_text(json.dumps(refs))
    print(f"JFLEG: {len(sources)} sentences, {len(refs[0])} refs each")


def prepare_bea_dev():
    tar_path = DATA_DIR / "wi_locness.tar.gz"
    with tarfile.open(tar_path) as tf:
        member = "wi+locness/m2/ABCN.dev.gold.bea19.m2"
        tf.extract(member, path=DATA_DIR)

    m2_path = DATA_DIR / "wi+locness/m2/ABCN.dev.gold.bea19.m2"
    lines = m2_path.read_text().splitlines()
    sources = [l[2:] for l in lines if l.startswith("S ")]

    (DATA_DIR / "bea_dev.src").write_text("\n".join(sources) + "\n")
    # keep a top-level copy of the gold m2 for errant_compare
    (DATA_DIR / "bea_dev.gold.m2").write_text(m2_path.read_text())
    print(f"BEA-2019-dev: {len(sources)} sentences")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    prepare_jfleg()
    prepare_bea_dev()
