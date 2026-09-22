"""Shared, import-friendly EDA helpers (data loading, text features, distribution distances).

Used by run_eda.py and importable from notebooks:

    import sys; sys.path.insert(0, "eda")   # repo root 기준
    from common import load_csvs, text_metrics, question_type, numeric_comparison, dev_vote_features
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
SPLITS = ("train", "test", "dev")
MAIN_SPLITS = ("train", "test")
CHOICES = ("a", "b", "c", "d")
DEV_ANSWER_COLS = tuple(f"answer{i}" for i in range(1, 6))


def load_csvs(data_root: Path = DATA_ROOT, splits: Iterable[str] = SPLITS) -> dict[str, pd.DataFrame]:
    return {split: pd.read_csv(Path(data_root) / f"{split}.csv") for split in splits}


QUESTION_RULES = [
    ("전화번호", re.compile(r"전화번호|연락처|문의(?:처|번호)?|번호는 무엇")),
    ("가격·금액", re.compile(r"가격|금액|얼마|원인가|요금|할인|결제")),
    ("수량·계수", re.compile(r"몇\s*(?:개|명|마리|대|잔|장|곳|가지|층)|개수|수량|얼마나 많|총 몇")),
    ("날짜·시간", re.compile(r"날짜|요일|시간|몇\s*시|기간|연도|년도|언제|영업시간")),
    ("색상", re.compile(r"색(?:깔|상)?|무슨 빛|어떤 빛")),
    ("위치·방향", re.compile(r"어디|위치|방향|왼쪽|오른쪽|위에|아래에|옆에|건너편|몇 층")),
    ("메뉴·상품", re.compile(r"메뉴|상품|제품|음식|판매|주문|대표 메뉴|종류는")),
    ("인물·대상", re.compile(r"누구|인물|사람|캐릭터|동물|무엇을 하고|어떤 사람")),
    ("문자·간판 읽기", re.compile(r"적힌|쓰여|문구|상호명|간판|글자|텍스트|이름은|명칭|표시된")),
]


def question_type(text: str) -> str:
    value = str(text)
    for label, pattern in QUESTION_RULES:
        if pattern.search(value):
            return label
    return "기타"


def text_metrics(df: pd.DataFrame, split: str) -> pd.DataFrame:
    records = []
    for row in df.itertuples(index=False):
        question = str(row.question)
        choices = [str(getattr(row, col)) for col in CHOICES]
        lengths = [len(x) for x in choices]
        word_lengths = [len(re.findall(r"\S+", x)) for x in choices]
        rec = {
            "split": split,
            "id": str(row.id),
            "question_chars": len(question),
            "question_tokens_ws": len(re.findall(r"\S+", question)),
            "choice_chars_mean": float(np.mean(lengths)),
            "choice_chars_max": max(lengths),
            "choice_chars_total": sum(lengths),
            "choice_tokens_ws_mean": float(np.mean(word_lengths)),
            "question_type": question_type(question),
        }
        for col, length in zip(CHOICES, lengths):
            rec[f"choice_{col}_chars"] = length
        records.append(rec)
    return pd.DataFrame(records)


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    values = np.sort(np.concatenate((a, b)))
    return float(np.max(np.abs(np.searchsorted(a, values, side="right") / len(a) - np.searchsorted(b, values, side="right") / len(b))))


def numeric_comparison(frame: pd.DataFrame, columns: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    compare_rows = []
    for column in columns:
        arrays = {}
        for split in MAIN_SPLITS:
            arr = frame.loc[frame.split == split, column].dropna().to_numpy(dtype=float)
            arrays[split] = arr
            q = np.quantile(arr, [0, .25, .5, .75, 1])
            detail_rows.append({
                "metric": column, "split": split, "count": len(arr), "mean": arr.mean(), "std": arr.std(ddof=1),
                "min": q[0], "p25": q[1], "median": q[2], "p75": q[3], "max": q[4],
            })
        train, test = arrays["train"], arrays["test"]
        pooled = math.sqrt((train.var(ddof=1) + test.var(ddof=1)) / 2)
        compare_rows.append({
            "metric": column,
            "train_mean": train.mean(), "test_mean": test.mean(),
            "mean_difference_test_minus_train": test.mean() - train.mean(),
            "standardized_mean_difference": (test.mean() - train.mean()) / pooled if pooled else 0.0,
            "ks_statistic": ks_statistic(train, test),
        })
    return pd.DataFrame(detail_rows), pd.DataFrame(compare_rows)


def total_variation(train_counts: pd.Series, test_counts: pd.Series) -> float:
    labels = sorted(set(train_counts.index) | set(test_counts.index))
    p = train_counts.reindex(labels, fill_value=0).to_numpy(float); p /= p.sum()
    q = test_counts.reindex(labels, fill_value=0).to_numpy(float); q /= q.sum()
    return float(0.5 * np.abs(p - q).sum())


def dev_vote_features(dev: pd.DataFrame) -> pd.DataFrame:
    """Annotator agreement per dev row: vote counts, majority answer, tie flag, entropy (bits)."""
    records = []
    for votes in dev[list(DEV_ANSWER_COLS)].itertuples(index=False):
        counts = Counter(v for v in votes if isinstance(v, str))
        n = sum(counts.values())
        ranked = counts.most_common()
        top = ranked[0][1] if ranked else 0
        probs = np.array([c / n for c in counts.values()]) if n else np.array([])
        records.append({
            "vote_count": n,
            "top_vote_count": top,
            "majority_answer": ranked[0][0] if ranked else None,
            "is_tie": len(ranked) > 1 and ranked[1][1] == top,
            "consensus_ratio": top / n if n else np.nan,
            "vote_entropy": float(-(probs * np.log2(probs)).sum()) if n else np.nan,
        })
    return pd.DataFrame(records, index=dev.index)
