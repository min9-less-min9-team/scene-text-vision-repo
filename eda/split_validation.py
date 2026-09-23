#!/usr/bin/env python3
"""train.csv → train / validation (10%) 분할. 결과는 data/split/ 에 CSV로 저장한다.

- 원본(data/raw)은 건드리지 않는다. data/split/{train,validation}.csv 는 원본과 같은 컬럼이며
  `path` 열은 그대로 data/raw 기준 상대 경로(train/xxx.jpg)다. 이미지는 복사하지 않는다.
- 같은 이미지(SHA-256 동일 또는 pHash 동일)가 train 과 validation 양쪽에 들어가지 않도록
  이미지 그룹 단위로 나눈다 (docs/EDA.md: "valid 분할은 이미지 해시 그룹 단위로").
- 정답 글자(a/b/c/d) 비율이 두 쪽에서 같도록 정답별로 층화한다.
- 시드를 고정(기본 42)했으므로 누가 실행해도 같은 분할이 나온다. 검증 id 목록은
  eda/tables/validation_ids.csv 로 버전 관리해 팀이 같은 검증 문항을 쓰는지 확인한다.

사용:
    uv run python eda/split_validation.py                # data/raw → data/split (이미 있으면 건너뜀)
    uv run python eda/split_validation.py --force        # 다시 만들기
    uv run python eda/split_validation.py --data-root /root/data/raw --split-dir /root/data/split

노트북에서:
    import sys; sys.path.insert(0, "eda")
    from split_validation import ensure_split
    paths = ensure_split(raw_dir, split_dir)          # 없으면 만들고, 있으면 그대로 반환
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_SPLIT_DIR = ROOT / "data" / "split"
DEFAULT_IDS_PATH = ROOT / "eda" / "tables" / "validation_ids.csv"

VALID_FRACTION = 0.10
SPLIT_SEED = 42
CHOICES = ("a", "b", "c", "d")
PHASH_SIZE, PHASH_LOW = 32, 8


# ── 이미지 해시 (run_eda.py 와 같은 정의) ─────────────────────────────────────
def sha256_of(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def phash_of(path: Path) -> str:
    """64-bit pHash (32×32 DCT 저주파 8×8). 재인코딩된 같은 장면을 같은 그룹으로 묶기 위해 사용."""
    with Image.open(path) as source:
        gray = ImageOps.exif_transpose(source).convert("L")
        arr = np.asarray(gray.resize((PHASH_SIZE, PHASH_SIZE), Image.Resampling.LANCZOS), dtype=np.float64)
    n = PHASH_SIZE
    x = np.arange(n)
    u = np.arange(n)[:, None]
    dct = np.cos((2 * x + 1) * u * np.pi / (2 * n))
    dct[0] /= math.sqrt(2)
    coeff = (2 / n) * dct @ arr @ dct.T
    low = coeff[:PHASH_LOW, :PHASH_LOW].ravel()
    bits = low > np.median(low[1:])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def image_groups(train: pd.DataFrame, raw_dir: Path, verbose: bool = True) -> np.ndarray:
    """행마다 이미지 그룹 id. SHA-256 이 같거나 pHash 가 같은 이미지는 같은 그룹(union-find)."""
    n = len(train)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    first_by_sha: dict[str, int] = {}
    first_by_phash: dict[str, int] = {}
    t0 = time.time()
    for i, rel in enumerate(train["path"].astype(str)):
        path = raw_dir / rel
        sha, ph = sha256_of(path), phash_of(path)
        if sha in first_by_sha:
            union(i, first_by_sha[sha])
        else:
            first_by_sha[sha] = i
        if ph in first_by_phash:
            union(i, first_by_phash[ph])
        else:
            first_by_phash[ph] = i
        if verbose and (i + 1) % 1000 == 0:
            print(f"  hashing {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    return np.array([find(i) for i in range(n)])


# ── 분할 ──────────────────────────────────────────────────────────────────────
def make_split(train: pd.DataFrame, raw_dir: Path, fraction: float = VALID_FRACTION, seed: int = SPLIT_SEED,
               verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """train → (train_part, valid_part, meta). 이미지 그룹 단위 + 정답별 층화."""
    train = train.reset_index(drop=True)
    answers = train["answer"].astype(str).str.strip().str.lower()
    assert answers.isin(CHOICES).all(), "answer 열에 a/b/c/d 외 값이 있음"

    if verbose:
        print(f"[split] {len(train)}개 train 이미지 해시 계산 (SHA-256 + pHash)", flush=True)
    groups = image_groups(train, raw_dir, verbose=verbose)
    n_groups = len(set(groups.tolist()))

    # 그룹 → (대표 정답, 행 인덱스들). 대표 정답 = 그룹 첫 행의 정답
    members: dict[int, list[int]] = {}
    for i, g in enumerate(groups.tolist()):
        members.setdefault(g, []).append(i)

    rng = np.random.default_rng(seed)
    valid_mask = np.zeros(len(train), dtype=bool)
    for choice in CHOICES:
        gids = sorted(g for g, rows in members.items() if answers.iloc[rows[0]] == choice)
        rng.shuffle(gids)
        target = int(round(fraction * sum(len(members[g]) for g in gids)))
        taken = 0
        for g in gids:
            if taken >= target:
                break
            valid_mask[members[g]] = True
            taken += len(members[g])

    valid_part = train[valid_mask].reset_index(drop=True)
    train_part = train[~valid_mask].reset_index(drop=True)
    meta = {
        "seed": seed,
        "fraction": fraction,
        "n_train_original": int(len(train)),
        "n_train": int(len(train_part)),
        "n_validation": int(len(valid_part)),
        "n_image_groups": int(n_groups),
        "n_multi_image_groups": int(sum(len(r) > 1 for r in members.values())),
        "answer_ratio_train": train_part["answer"].value_counts(normalize=True).sort_index().round(4).to_dict(),
        "answer_ratio_validation": valid_part["answer"].value_counts(normalize=True).sort_index().round(4).to_dict(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return train_part, valid_part, meta


def write_split(train_part: pd.DataFrame, valid_part: pd.DataFrame, meta: dict, split_dir: Path,
                ids_path: Path | None = DEFAULT_IDS_PATH) -> dict[str, Path]:
    split_dir.mkdir(parents=True, exist_ok=True)
    paths = {"train": split_dir / "train.csv", "validation": split_dir / "validation.csv", "meta": split_dir / "split_meta.json"}
    train_part.to_csv(paths["train"], index=False)
    valid_part.to_csv(paths["validation"], index=False)
    paths["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if ids_path is not None:
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        valid_part[["id"]].to_csv(ids_path, index=False)
        paths["ids"] = ids_path
    return paths


def ensure_split(raw_dir: Path = DEFAULT_RAW_DIR, split_dir: Path = DEFAULT_SPLIT_DIR, force: bool = False,
                 ids_path: Path | None = DEFAULT_IDS_PATH, verbose: bool = True) -> dict[str, Path]:
    """data/split/{train,validation}.csv 가 없으면 만들고, 있으면 건너뛴다. 경로 dict 반환."""
    raw_dir, split_dir = Path(raw_dir), Path(split_dir)
    paths = {"train": split_dir / "train.csv", "validation": split_dir / "validation.csv", "meta": split_dir / "split_meta.json"}
    if not force and paths["train"].exists() and paths["validation"].exists():
        if verbose:
            n_tr = sum(1 for _ in paths["train"].open(encoding="utf-8")) - 1
            n_va = sum(1 for _ in paths["validation"].open(encoding="utf-8")) - 1
            print(f"[split] skip: 이미 존재 → {split_dir} (train {n_tr} | validation {n_va})")
        return paths
    train = pd.read_csv(raw_dir / "train.csv")
    train_part, valid_part, meta = make_split(train, raw_dir, verbose=verbose)
    paths = write_split(train_part, valid_part, meta, split_dir, ids_path)
    if verbose:
        print(f"[split] train {meta['n_train']} | validation {meta['n_validation']} "
              f"(이미지 그룹 {meta['n_image_groups']}, 다중 이미지 그룹 {meta['n_multi_image_groups']}) → {split_dir}")
        print(f"[split] 정답 비율 train {meta['answer_ratio_train']} | validation {meta['answer_ratio_validation']}")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_RAW_DIR, help="원본 CSV·이미지 폴더 (기본 data/raw)")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR, help="분할 CSV 저장 폴더 (기본 data/split)")
    parser.add_argument("--ids-out", type=Path, default=DEFAULT_IDS_PATH, help="검증 id 목록 CSV (버전 관리용). 'none' 이면 저장 안 함")
    parser.add_argument("--force", action="store_true", help="이미 있어도 다시 분할")
    args = parser.parse_args()
    ids_path = None if str(args.ids_out).lower() == "none" else args.ids_out
    ensure_split(args.data_root, args.split_dir, force=args.force, ids_path=ids_path)


if __name__ == "__main__":
    main()
