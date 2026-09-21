import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .config import Config

CHOICES = ["a", "b", "c", "d"]
Image.MAX_IMAGE_PIXELS = None

SYSTEM_EN = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d. No explanation."
)
SYSTEM_KO = "당신은 이미지를 보고 질문에 답하는 도우미입니다. a, b, c, d 중 정답 한 글자만 답하고 설명은 하지 마세요."
ANSWER_RULE = "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
PROMPTS = {  # variant: (시스템 프롬프트, 질문 뒤에 붙는 지시문)
    "a": (SYSTEM_EN, ANSWER_RULE),
    "b": (SYSTEM_EN, "이미지 속 글자를 한 글자씩 주의 깊게 읽고 답하세요. " + ANSWER_RULE),
    "c": (SYSTEM_KO, ANSWER_RULE),
}


def load_image(data_dir: str, path: str) -> Image.Image:
    return Image.open(os.path.join(data_dir, path)).convert("RGB")


def build_messages(img, question: str, options: list[str], answer: str | None = None, variant: str = "a") -> list[dict]:
    """options: 화면에 보여줄 순서의 보기 4개. answer가 있으면 학습용(assistant 턴 포함).
    학습과 추론이 같은 함수로 프롬프트를 만들어 불일치를 막는다."""
    system, rule = PROMPTS[variant]
    text = f"{question}\n(a) {options[0]}\n(b) {options[1]}\n(c) {options[2]}\n(d) {options[3]}\n\n{rule}"
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def majority_vote(row) -> tuple[str | None, int]:
    """dev.csv answer1~5 다수결 → (라벨, 득표수). 공동 1위면 라벨 None"""
    votes = [str(row[f"answer{k}"]).strip().lower() for k in range(1, 6) if pd.notna(row[f"answer{k}"])]
    votes = [v for v in votes if v in CHOICES]
    if not votes:
        return None, 0
    counts = {c: votes.count(c) for c in CHOICES}
    top = max(counts.values())
    winners = [c for c in CHOICES if counts[c] == top]
    return (winners[0] if len(winners) == 1 else None), top


def load_dev(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, "dev.csv"))
    mv = df.apply(majority_vote, axis=1)
    df["answer"] = [m[0] for m in mv]
    df["votes"] = [m[1] for m in mv]
    return df


def split_train_valid(df: pd.DataFrame, valid_size: int, split_seed: int):
    """train.csv에서 valid_size개를 고정 시드로 떼어 검증셋으로. 공개 어댑터(ssafyjinhyeok/KFC)와 같은 분할(500, seed 42)."""
    if not valid_size:
        return df.reset_index(drop=True), df.iloc[0:0]
    valid_df = df.sample(n=valid_size, random_state=split_seed).reset_index(drop=True)
    train_df = df[~df["id"].isin(valid_df["id"])].reset_index(drop=True)
    return train_df, valid_df


def load_split(cfg: Config, split: str) -> pd.DataFrame:
    """split: train | valid | dev | test. train/valid both come from train.csv."""
    if split == "test":
        return pd.read_csv(os.path.join(cfg.data_dir, "test.csv"))
    if split == "dev":
        return load_dev(cfg.data_dir)
    df = pd.read_csv(os.path.join(cfg.data_dir, "train.csv"))
    train_df, valid_df = split_train_valid(df, cfg.valid_size, cfg.split_seed)
    return train_df if split == "train" else valid_df


def load_train(cfg: Config) -> pd.DataFrame:
    """학습용 train (+ 선택적으로 dev 다수결 합의 문항), 셔플·서브샘플 적용."""
    df = load_split(cfg, "train")
    if cfg.max_train_samples and cfg.max_train_samples < len(df):
        df = df.sample(n=cfg.max_train_samples, random_state=cfg.seed)
    if cfg.use_dev:
        dev = load_dev(cfg.data_dir)
        dev = dev[(dev["votes"] >= cfg.dev_min_votes) & dev["answer"].notna()]
        print(f"dev: {cfg.dev_min_votes}표 이상 합의 {len(dev)}문항을 학습에 추가")
        df = pd.concat([df, dev[df.columns]], ignore_index=True)
    return df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)


def subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n and n < len(df):
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df


class VQATrainDataset(Dataset):
    """이미지는 필요할 때 디스크에서 읽음. shuffle_options면 매번 보기 순서를 섞고 정답 글자를 그에 맞게 바꿈."""

    def __init__(self, df: pd.DataFrame, cfg: Config):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.n_seen: dict[int, int] = {}  # 같은 문항도 epoch마다 다른 보기 순서가 되도록

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        options = [str(row[c]) for c in CHOICES]
        gold = CHOICES.index(str(row["answer"]).strip().lower())
        order = list(range(4))  # order[j] = j번째 자리에 보여줄 원래 보기
        if self.cfg.shuffle_options:
            k = self.n_seen.get(i, 0)
            self.n_seen[i] = k + 1
            random.Random(self.cfg.seed * 1_000_003 + i * 7_919 + k).shuffle(order)
        shown = [options[j] for j in order]
        answer = CHOICES[order.index(gold)]
        img = load_image(self.cfg.data_dir, row["path"])
        return {"messages": build_messages(img, str(row["question"]), shown, answer, self.cfg.prompt_variant)}


def is_numeric_options(df: pd.DataFrame) -> np.ndarray:
    import re

    return df[CHOICES].astype(str).apply(
        lambda r: all(re.fullmatch(r"[\d\s,.\-:/%원₩$+()]+", x) for x in r), axis=1
    ).values
