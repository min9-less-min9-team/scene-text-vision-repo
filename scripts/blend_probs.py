# -*- coding: utf-8 -*-
"""
Combine probability files produced by vqa_textmc.py (*_probs.npz) into one submission.

Three strategies, mirroring what worked for previous winners:
  weighted : p = sum(w_i * p_i)                                      (plain ensemble)
  gated    : start from primary; for rows whose primary margin < --gate-margin, p = (1-w)*p_primary + w*p_secondary
             (kimkihyun1 BASEOFF_BALANCED style; optionally require secondary confidence/agreement)
  replace  : start from primary; replace a row with secondary argmax only if secondary margin >= --min-secondary-margin
             and primary margin < --gate-margin                      (ai_chall conservative merge)

Examples:
  python blend_probs.py --primary runs/ft27b/test_probs.npz --secondary runs/ft27b/test_cropfive_probs.npz \
      --strategy gated --gate-margin 0.15 --w 0.35 --sample-csv data/sample_submission.csv --out sub_gated.csv
  python blend_probs.py --primary A.npz --secondary B.npz --strategy weighted --w 0.5 --out sub_avg.csv
Secondary files may cover a subset of ids (e.g. rerun on uncertain items); rows not covered keep the primary probs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHOICES = np.array(["a", "b", "c", "d"])


def load(path: str):
    z = np.load(path, allow_pickle=True)
    ids = [str(i) for i in z["ids"].tolist()]
    runs = z["runs"] if "runs" in z.files else z["avg"][None]
    return ids, z["avg"], runs


def margin_of(p: np.ndarray) -> np.ndarray:
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]


def agreement_of(runs: np.ndarray) -> np.ndarray:
    votes = runs.argmax(axis=2)
    return np.array([np.bincount(votes[:, i], minlength=4).max() / runs.shape[0] for i in range(runs.shape[1])])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--primary", required=True)
    p.add_argument("--secondary", required=True)
    p.add_argument("--strategy", default="gated", choices=["weighted", "gated", "replace"])
    p.add_argument("--w", type=float, default=0.5, help="weight of secondary")
    p.add_argument("--gate-margin", type=float, default=0.15, help="apply only where primary margin < this")
    p.add_argument("--min-secondary-conf", type=float, default=0.0, help="gated/replace: secondary top prob >= this")
    p.add_argument("--min-secondary-agree", type=float, default=0.0, help="gated/replace: secondary TTA agreement >= this")
    p.add_argument("--min-secondary-margin", type=float, default=0.3, help="replace: secondary margin >= this")
    p.add_argument("--max-support-gap", type=float, default=1.0,
                   help="gated: skip if primary prob of its own answer minus primary prob of secondary answer > this")
    p.add_argument("--sample-csv")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ids, p1, runs1 = load(args.primary)
    ids2, p2_raw, runs2 = load(args.secondary)
    pos2 = {i: k for k, i in enumerate(ids2)}
    covered = np.array([i in pos2 for i in ids])
    p2 = p1.copy()
    agree2 = np.ones(len(ids))
    for k, i in enumerate(ids):
        if i in pos2:
            p2[k] = p2_raw[pos2[i]]
    if runs2.shape[0] > 1:
        a2 = agreement_of(runs2)
        for k, i in enumerate(ids):
            if i in pos2:
                agree2[k] = a2[pos2[i]]

    m1 = margin_of(p1)
    if covered.any():
        dis = float((p1.argmax(axis=1) != p2.argmax(axis=1))[covered].mean())
        # 16-1 2nd place rule of thumb: a member is worth adding only if it disagrees with the champion on >= 8%
        # of items (and is within ~1%p accuracy); same-family reruns (~4%) contributed nothing.
        print(f"diversity: secondary disagrees with primary on {dis:.1%} of covered items (>= 8% = worth blending)")
    final = p1.copy()
    if args.strategy == "weighted":
        final[covered] = (1 - args.w) * p1[covered] + args.w * p2[covered]
        applied = covered
    else:
        sec_conf = p2.max(axis=1)
        sec_pred = p2.argmax(axis=1)
        pri_pred = p1.argmax(axis=1)
        support_gap = p1[np.arange(len(ids)), pri_pred] - p1[np.arange(len(ids)), sec_pred]
        gate = covered & (m1 < args.gate_margin) & (sec_conf >= args.min_secondary_conf) & (agree2 >= args.min_secondary_agree) \
            & (support_gap <= args.max_support_gap)
        if args.strategy == "gated":
            final[gate] = (1 - args.w) * p1[gate] + args.w * p2[gate]
        else:
            gate = gate & (margin_of(p2) >= args.min_secondary_margin)
            final[gate] = p2[gate]
        applied = gate
    pred = CHOICES[final.argmax(axis=1)]
    changed = int((pred != CHOICES[p1.argmax(axis=1)]).sum())
    print(f"rows={len(ids)} secondary-covered={int(covered.sum())} applied={int(applied.sum())} answers changed={changed}")
    # Noise threshold (16-1 2nd place): if two submissions differ on D items, the LB score difference has
    # SD ~ sqrt(D)/2, so differences below ~sqrt(D) items are not measurable. Don't spend a submission on it.
    if changed:
        print(f"noise threshold: 2-sigma ~ {int(changed ** 0.5)} items; a public-LB gain smaller than that is not evidence")
    else:
        print("no answers changed vs primary: identical submission, do not submit")

    sub = pd.DataFrame({"id": ids, "answer": pred})
    if args.sample_csv and Path(args.sample_csv).exists():
        sample = pd.read_csv(args.sample_csv)
        id_col, ans_col = sample.columns[0], (sample.columns[1] if len(sample.columns) > 1 else "answer")
        merged = sample[[id_col]].assign(**{id_col: sample[id_col].astype(str)}).merge(
            sub.rename(columns={"id": id_col, "answer": ans_col}), on=id_col, how="left")
        assert merged[ans_col].notna().all(), "missing predictions for some sample ids"
        if pd.api.types.is_numeric_dtype(sample[id_col]):
            merged[id_col] = pd.to_numeric(merged[id_col])
        if ans_col in sample.columns and sample[ans_col].astype(str).str.fullmatch(r"[ABCD]").mean() > 0.5:
            merged[ans_col] = merged[ans_col].str.upper()
        merged.to_csv(args.out, index=False)
    else:
        sub.to_csv(args.out, index=False)
    np.savez_compressed(Path(args.out).with_suffix(".npz"), avg=final, runs=final[None], ids=np.array(ids, dtype=object))
    print("written:", args.out)


if __name__ == "__main__":
    main()
