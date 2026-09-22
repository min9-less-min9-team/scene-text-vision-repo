# -*- coding: utf-8 -*-
"""
First-30-minutes data inspection for a multiple-choice VQA competition (pandas/PIL only, runs on a laptop).

Usage:
  python inspect_data.py --root /path/to/data            # expects train.csv, test.csv, sample_submission.csv
  python inspect_data.py --root . --train-csv train.csv --test-csv test.csv --image-root images

Reports: columns, sample rows, answer distribution (position bias), question keyword/qtype distribution,
choice-text statistics (length, digits, duplicates), image resolution stats, corrupted images,
train/test image overlap by content hash, repeated questions, and choices that appear verbatim in the question.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from vqa_textmc import CHOICES, attach_paths, detect_schema, standardize_df

Image.MAX_IMAGE_PIXELS = None


def content_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def image_stats(df: pd.DataFrame, name: str, limit: int):
    sizes, bad = [], []
    for p in df.loc[df["abs_path"] != "", "abs_path"].head(limit):
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im)
                sizes.append((im.width, im.height, im.mode))
        except Exception as e:  # noqa
            bad.append((p, str(e)[:80]))
    if sizes:
        w = np.array([s[0] for s in sizes]); h = np.array([s[1] for s in sizes])
        print(f"[{name}] images checked={len(sizes)} | width p5/p50/p95 = {np.percentile(w,5):.0f}/{np.median(w):.0f}/{np.percentile(w,95):.0f}"
              f" | height p5/p50/p95 = {np.percentile(h,5):.0f}/{np.median(h):.0f}/{np.percentile(h,95):.0f}"
              f" | modes={Counter(s[2] for s in sizes).most_common(3)}")
        long_side = np.maximum(w, h)
        print(f"[{name}] long side > 1500px: {(long_side>1500).mean():.1%} | > 2500px: {(long_side>2500).mean():.1%}"
              f"  -> decide --max-visual-tokens (1024 ~ 1024px^2 budget at 32px/token)")
    if bad:
        print(f"[{name}] corrupted/unreadable images: {len(bad)} e.g. {bad[:3]}")


def text_stats(df: pd.DataFrame, name: str):
    print(f"\n=== {name}: {len(df)} rows ===")
    print("qtype distribution:")
    print(df["qtype"].value_counts().to_string())
    if "answer" in df:
        print("answer position distribution (position bias check):")
        print(df["answer"].value_counts(normalize=True).sort_index().round(3).to_string())
    lens = df[CHOICES].apply(lambda s: s.str.len())
    print(f"choice text length: mean={lens.values.mean():.1f} max={lens.values.max()}")
    digit_share = df[CHOICES].apply(lambda s: s.str.contains(r"\d")).values.mean()
    print(f"share of choices containing digits: {digit_share:.1%} (phone/price/floor style questions)")
    dup_choice_rows = (df[CHOICES].nunique(axis=1) < 4).sum()
    print(f"rows with duplicated choice texts: {dup_choice_rows}")
    in_q = df.apply(lambda r: sum(str(r[c]) in str(r['question']) for c in CHOICES), axis=1)
    print(f"rows where a choice string appears inside the question: {(in_q>0).sum()}")
    words = Counter()
    for q in df["question"]:
        for w in re.findall(r"[가-힣A-Za-z]+", str(q)):
            words[w] += 1
    print("top question words:", words.most_common(30))
    rep_q = df["question"].value_counts()
    print(f"unique questions: {df['question'].nunique()} | most repeated: {rep_q.head(5).to_dict()}")
    if "answer" in df:
        # Similarity of correct answer vs distractors (edit-distance proxy: shared char ratio)
        def shared(a: str, b: str) -> float:
            sa, sb = set(a), set(b)
            return len(sa & sb) / max(1, len(sa | sb))
        sims = []
        for _, r in df.head(2000).iterrows():
            gold = str(r[r["answer"]])
            sims.append(max(shared(gold, str(r[c])) for c in CHOICES if c != r["answer"]))
        print(f"max char-overlap between gold and distractors (mean over first 2000): {np.mean(sims):.2f}"
              " (high => near-duplicate distractors, spelling precision matters)")
    print("sample rows:")
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(df[["id", "path", "question"] + CHOICES + (["answer"] if "answer" in df else [])].head(5).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--train-csv"); p.add_argument("--test-csv"); p.add_argument("--sample-csv"); p.add_argument("--image-root")
    p.add_argument("--id-col"); p.add_argument("--path-col"); p.add_argument("--question-col"); p.add_argument("--answer-col")
    p.add_argument("--choice-cols")
    p.add_argument("--image-limit", type=int, default=300, help="images to open for size stats")
    p.add_argument("--hash-limit", type=int, default=100000, help="images to hash for train/test overlap")
    args = p.parse_args()
    root = Path(args.root)
    paths = {
        "train": Path(args.train_csv) if args.train_csv else root / "train.csv",
        "test": Path(args.test_csv) if args.test_csv else root / "test.csv",
    }
    roots = ([Path(args.image_root)] if args.image_root else []) + [root, root / "images"]
    frames = {}
    for name, path in paths.items():
        if not path.exists():
            print(f"{name}: {path} missing"); continue
        raw = pd.read_csv(path)
        print(f"\n{name} raw columns: {list(raw.columns)} shape={raw.shape}")
        print(raw.head(3).to_string())
        schema = detect_schema(raw, args)
        print("detected schema:", schema)
        df = attach_paths(standardize_df(raw, schema, is_train=(name == "train")), roots)
        frames[name] = df
        text_stats(df, name)
        image_stats(df, name, args.image_limit)
    sample = Path(args.sample_csv) if args.sample_csv else root / "sample_submission.csv"
    if sample.exists():
        s = pd.read_csv(sample)
        print(f"\nsample_submission columns={list(s.columns)} rows={len(s)} first values={s.head(3).values.tolist()}")
        if "test" in frames:
            print("test ids == sample ids:", set(frames["test"]["id"]) == set(s.iloc[:, 0].astype(str)))
    if "train" in frames and "test" in frames:
        tr = frames["train"][frames["train"]["abs_path"] != ""]
        te = frames["test"][frames["test"]["abs_path"] != ""]
        tr_h = {content_hash(p): p for p in tr["abs_path"].unique()[: args.hash_limit]}
        te_h = {content_hash(p): p for p in te["abs_path"].unique()[: args.hash_limit]}
        overlap = set(tr_h) & set(te_h)
        print(f"\nunique train images={len(tr_h)} unique test images={len(te_h)} | identical images in both: {len(overlap)}")
        print(f"questions per image: train={len(tr)/max(1,len(tr_h)):.2f} test={len(te)/max(1,len(te_h)):.2f}"
              "  -> use group split by image for validation")
        shared_q = set(frames["train"]["question"]) & set(frames["test"]["question"])
        print(f"test questions also present in train (exact string): {len(shared_q)}")


if __name__ == "__main__":
    import sys

    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
