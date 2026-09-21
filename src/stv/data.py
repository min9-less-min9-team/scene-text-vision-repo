import os

import numpy as np
import pandas as pd

from .config import Config

CHOICES = ["a", "b", "c", "d"]

SYSTEM_PROMPT = (
    "You are a visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d."
)


def build_prompt(row) -> str:
    return (
        f"{row['question']}\n"
        f"(a) {row['a']}\n"
        f"(b) {row['b']}\n"
        f"(c) {row['c']}\n"
        f"(d) {row['d']}\n\n"
        "Answer with exactly one lowercase letter: a, b, c, or d."
    )


def build_messages(row, data_dir: str, with_answer: bool) -> list[dict]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.join(data_dir, row["path"])},
                {"type": "text", "text": build_prompt(row)},
            ],
        },
    ]
    if with_answer:
        answer = str(row["answer"]).strip().lower()
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def split_train_valid(df: pd.DataFrame, valid_ratio: float, seed: int):
    """Group split by image so the same image never lands on both sides."""
    paths = np.sort(df["path"].unique())
    np.random.default_rng(seed).shuffle(paths)
    valid_paths = set(paths[: int(len(paths) * valid_ratio)])
    is_valid = df["path"].isin(valid_paths)
    return df[~is_valid].reset_index(drop=True), df[is_valid].reset_index(drop=True)


def load_split(cfg: Config, split: str) -> pd.DataFrame:
    """split: train | valid | dev | test. train/valid both come from train.csv."""
    if split == "test":
        return pd.read_csv(os.path.join(cfg.data_dir, "test.csv"))
    if split == "dev":
        df = pd.read_csv(os.path.join(cfg.data_dir, "dev.csv"))
        # dev.csv는 annotator 5명의 답(answer1~5, 결측 있음) → 다수결을 정답으로 사용
        df["answer"] = df.filter(regex=r"^answer\d$").mode(axis=1)[0]
        return df
    df = pd.read_csv(os.path.join(cfg.data_dir, "train.csv"))
    train_df, valid_df = split_train_valid(df, cfg.valid_ratio, cfg.seed)
    return train_df if split == "train" else valid_df


def subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n and n < len(df):
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df


def to_sft_dataset(df: pd.DataFrame, data_dir: str):
    from datasets import Dataset

    return Dataset.from_list(
        [{"messages": build_messages(row, data_dir, with_answer=True)} for _, row in df.iterrows()]
    )
