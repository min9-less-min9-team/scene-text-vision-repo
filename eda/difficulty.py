"""Zero-shot difficulty probe: combine a/b/c/d probabilities from several models into per-sample difficulty.

Inputs are the `{split}_probs.npz` files written by `stv.inference` (`avg` [N,4], `ids`).
Probes are passed as an ordered dict, smallest model first. Paths may be absolute, e.g. on Colab the npz live on Drive:
    {"2b": "/content/drive/MyDrive/stv/outputs/probe_2b", "4b": "/content/drive/MyDrive/stv/outputs/probe_4b"}
Labels are read from the repo's `data/` (unpacked by `scripts/setup.sh`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import CHOICES, dev_vote_features, load_csvs


def load_probe(out_dir: str | Path, split: str) -> pd.DataFrame:
    """id, p_a..p_d, pred, entropy(bits) for one model on one split."""
    z = np.load(Path(out_dir) / f"{split}_probs.npz", allow_pickle=True)
    probs = z["avg"].astype(float)
    frame = pd.DataFrame(probs, columns=[f"p_{c}" for c in CHOICES])
    frame.insert(0, "id", z["ids"].astype(str))
    frame["pred"] = [CHOICES[k] for k in probs.argmax(1)]
    frame["entropy"] = -(probs * np.log2(np.clip(probs, 1e-12, 1))).sum(1)
    return frame


def load_labels(split: str, data_root=None) -> pd.DataFrame:
    """id, answer (+ dev: is_tie, consensus_ratio). dev answer = annotator majority."""
    df = load_csvs(data_root, [split])[split] if data_root else load_csvs(splits=[split])[split]
    if split == "dev":
        votes = dev_vote_features(df)
        df = df.assign(answer=votes.majority_answer, is_tie=votes.is_tie, consensus_ratio=votes.consensus_ratio)
        df = df[df.answer.notna()]
    df = df.assign(answer=df.answer.astype(str).str.strip().str.lower())
    keep = ["id", "question", "answer"] + [c for c in ("is_tie", "consensus_ratio") if c in df]
    return df[keep].reset_index(drop=True)


def tier(correct: list[bool]) -> str:
    """correct is ordered smallest model -> largest.

    Easy: all correct / Hard: none correct / Medium: only larger models correct /
    Inconsistent: a smaller model is right where a larger one is wrong.
    With three probes, Medium splits into Medium (fails: smallest) and Medium-Hard (only the largest is right).
    """
    if all(correct):
        return "Easy"
    if not any(correct):
        return "Hard"
    first = correct.index(True)
    if not all(correct[first:]):
        return "Inconsistent"
    return "Medium" if first == 1 or len(correct) == 2 else "Medium-Hard"


TIER_ORDER = ["Easy", "Medium", "Medium-Hard", "Hard", "Inconsistent"]


def difficulty_table(probes: dict[str, str], split: str, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Per-sample table: {name}_correct, p_correct_{name}, entropy_{name}, tier, difficulty = 1 - weighted mean P(correct).

    weights default to equal (= mean confidence). Only ids present in every probe are kept.
    """
    table = load_labels(split)
    for name, out_dir in probes.items():
        probe = load_probe(out_dir, split)
        merged = table[["id", "answer"]].merge(probe, on="id", how="inner")
        probs = merged[[f"p_{c}" for c in CHOICES]].to_numpy()
        p_correct = probs[np.arange(len(merged)), merged.answer.map(CHOICES.index).to_numpy()]
        merged = merged.assign(**{
            f"pred_{name}": merged.pred,
            f"{name}_correct": merged.pred == merged.answer,
            f"p_correct_{name}": p_correct,
            f"entropy_{name}": merged.entropy,
        })
        table = table.merge(merged[["id", f"pred_{name}", f"{name}_correct", f"p_correct_{name}", f"entropy_{name}"]], on="id", how="inner")

    names = list(probes)
    weights = weights or {n: 1.0 for n in names}
    total = sum(weights[n] for n in names)
    table["n_correct"] = table[[f"{n}_correct" for n in names]].sum(1)
    table["tier"] = [tier(list(row)) for row in table[[f"{n}_correct" for n in names]].itertuples(index=False)]
    table["difficulty"] = 1 - sum(weights[n] / total * table[f"p_correct_{n}"] for n in names)
    table.insert(0, "split", split)
    return table


def probe_summary(table: pd.DataFrame, probes: dict[str, str]) -> pd.DataFrame:
    """accuracy / mean P(correct) / mean prediction entropy per model."""
    return pd.DataFrame({
        name: {
            "n": len(table),
            "accuracy": table[f"{name}_correct"].mean(),
            "p_correct_mean": table[f"p_correct_{name}"].mean(),
            "entropy_mean": table[f"entropy_{name}"].mean(),
        } for name in probes
    }).T


def tier_counts(table: pd.DataFrame) -> pd.DataFrame:
    counts = table.tier.value_counts().reindex(TIER_ORDER).dropna().astype(int)
    return pd.DataFrame({"count": counts, "percent": (counts / len(table) * 100).round(2)})
