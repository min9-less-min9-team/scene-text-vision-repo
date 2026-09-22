# -*- coding: utf-8 -*-
"""
Optional: extract Korean/English text from every image with an OCR engine and save id -> ocr_text.
Feed the result to vqa_textmc.py with --ocr-csv to append recognized text as a hint in the prompt.

ViSignVQA (signboard VQA paper) reports large gains from appending OCR text for small models; for strong
VLMs the gain is smaller and can hurt when OCR is wrong, so ALWAYS A/B test on the validation split first.

Install (Colab):
  !pip install -q easyocr            # default engine, supports ko+en
  # or: !pip install -q paddleocr paddlepaddle-gpu   (--engine paddle)

Usage:
  python ocr_extract.py --root /content/drive/MyDrive/ssafy-16-2 --out ocr_test.csv --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vqa_textmc import attach_paths, detect_schema, standardize_df


def run_easyocr(paths, gpu: bool, min_conf: float):
    import easyocr

    reader = easyocr.Reader(["ko", "en"], gpu=gpu)
    out = []
    for i, p in enumerate(paths, 1):
        try:
            res = reader.readtext(p, detail=1, paragraph=False)
            # sort top-to-bottom then left-to-right, keep confident boxes
            items = [(min(pt[1] for pt in box), min(pt[0] for pt in box), txt) for box, txt, conf in res if conf >= min_conf]
            items.sort(key=lambda t: (round(t[0] / 40), t[1]))
            out.append(" | ".join(t[2] for t in items))
        except Exception as e:  # noqa
            out.append("")
            print("ocr failed", p, str(e)[:80])
        if i % 100 == 0:
            print(f"{i}/{len(paths)}", flush=True)
    return out


def run_paddle(paths, min_conf: float):
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(lang="korean", use_angle_cls=True, show_log=False)
    out = []
    for i, p in enumerate(paths, 1):
        try:
            res = ocr.ocr(p, cls=True) or []
            items = []
            for line in res:
                for box, (txt, conf) in (line or []):
                    if conf >= min_conf:
                        items.append((min(pt[1] for pt in box), min(pt[0] for pt in box), txt))
            items.sort(key=lambda t: (round(t[0] / 40), t[1]))
            out.append(" | ".join(t[2] for t in items))
        except Exception as e:  # noqa
            out.append("")
            print("ocr failed", p, str(e)[:80])
        if i % 100 == 0:
            print(f"{i}/{len(paths)}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--csv")
    p.add_argument("--image-root")
    p.add_argument("--engine", default="easyocr", choices=["easyocr", "paddle"])
    p.add_argument("--min-conf", type=float, default=0.3)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--id-col"); p.add_argument("--path-col"); p.add_argument("--question-col"); p.add_argument("--answer-col")
    p.add_argument("--choice-cols")
    args = p.parse_args()
    root = Path(args.root)
    csv = Path(args.csv) if args.csv else root / f"{args.split}.csv"
    raw = pd.read_csv(csv)
    schema = detect_schema(raw, args)
    df = attach_paths(standardize_df(raw, schema, is_train=False), ([Path(args.image_root)] if args.image_root else []) + [root, root / "images"])
    uniq = df.drop_duplicates("abs_path")
    paths = uniq["abs_path"].tolist()
    texts = run_easyocr(paths, gpu=not args.cpu, min_conf=args.min_conf) if args.engine == "easyocr" else run_paddle(paths, args.min_conf)
    by_path = dict(zip(paths, texts))
    pd.DataFrame({"id": df["id"], "ocr_text": df["abs_path"].map(by_path).fillna("")}).to_csv(args.out, index=False, encoding="utf-8-sig")
    print("written:", args.out, "| empty:", int((pd.Series(texts) == "").sum()), "/", len(texts))


if __name__ == "__main__":
    main()
