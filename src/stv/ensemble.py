"""여러 실험의 확률 평균 → 검증 점수 확인 + 앙상블 제출 파일.

    uv run stv-ensemble outputs/run_a outputs/run_b [--output-dir outputs/ensemble]

각 폴더의 valid_probs.npz / test_probs.npz(stv-train, stv-infer가 저장)를 평균합니다.
같은 valid_size·split_seed로 만든 실험끼리만 비교할 수 있습니다. 같은 어댑터의 TTA 변형은 1개만 넣으세요.
"""

import argparse
import os

import numpy as np
import pandas as pd

from .config import Config
from .data import CHOICES, load_split
from .inference import evaluate


def load_probs(run_dir: str, split: str):
    with np.load(os.path.join(run_dir, f"{split}_probs.npz"), allow_pickle=True) as f:
        return f["avg"], f["ids"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="실험 output_dir 목록")
    parser.add_argument("--output-dir", default="outputs/ensemble")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--valid-size", type=int, default=500)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    cfg = Config(data_dir=args.data_dir, valid_size=args.valid_size, split_seed=args.split_seed)

    valid_df = load_split(cfg, "valid")
    valid_runs = [r for r in args.runs if os.path.exists(os.path.join(r, "valid_probs.npz"))]
    if valid_runs:
        stack = []
        for r in valid_runs:
            probs, ids = load_probs(r, "valid")
            assert (ids == valid_df["id"].values).all(), f"{r}: valid 문항이 다름 (valid_size·split_seed 확인)"
            evaluate(valid_df, probs, name=r, show_wrong=0)
            stack.append(probs)
        evaluate(valid_df, np.mean(stack, axis=0), name=f"앙상블 {len(stack)}개", show_wrong=0)

    test_runs = [r for r in args.runs if os.path.exists(os.path.join(r, "test_probs.npz"))]
    if test_runs:
        stack, ids = [], None
        for r in test_runs:
            probs, ids_r = load_probs(r, "test")
            assert ids is None or (ids == ids_r).all(), f"{r}: test 문항이 다름"
            ids, stack = ids_r, stack + [probs]
        ens = np.mean(stack, axis=0)
        os.makedirs(args.output_dir, exist_ok=True)
        pd.DataFrame({"id": ids, "answer": [CHOICES[i] for i in ens.argmax(1)]}).to_csv(
            os.path.join(args.output_dir, "submission_test.csv"), index=False)
        np.savez(os.path.join(args.output_dir, "test_probs.npz"), avg=ens, ids=ids)
        print(f"Saved: {args.output_dir}/submission_test.csv ({len(test_runs)}개 평균. 앙상블 검증 점수가 단일 실험보다 높을 때만 제출)")


if __name__ == "__main__":
    main()
