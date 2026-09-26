#!/usr/bin/env python3
"""
최종 학습 스크립트 — HyperCLOVAX-SEED-Think-32B (bf16 LoRA, transformers + peft)

[데이터] 1단계 external: KRETA + MTVQA(KO, 4지선다 변환), 증강 없음 (--no-external로 끄기)
         2단계 target  : train split(삭제 목록 적용) + dev(투표 패턴 3 / 3:1 / 3:1:1, 손라벨링 삭제 적용)
                         × 원본·compression·blur·affine = 4배
[학습]   LoRA, 프롬프트 2회 반복, 정답 글자 1토큰 cross-entropy, 1 epoch
[추론]   TTA 2회 (보기 순환 shift 0, 2)
[산출물] OUTPUT_DIR/
         checkpoints/<stage>/checkpoint-*   단계별 학습 중 checkpoint (최근 2개)
         adapter_external/          external 단계 LoRA
         adapter/                   최종(target 단계) LoRA adapter + processor
         valid_soft.csv, valid_probs.npy        validation soft label (+pred, answer)
         submission_hard.csv                    test hard label (id, answer)
         submission_soft.csv, test_probs.npy    test soft label (id, a, b, c, d)
         metrics.json, run_config.json, train_base_rows.csv, external_rows.csv

실행 순서 (pod에서):
    python train_hcx_think_32b.py --prepare-only            # 데이터 구성 확인 (모델 로드 없음)
    python train_hcx_think_32b.py --check                   # 모델 로드 + 프롬프트/토큰/확률 점검
    python train_hcx_think_32b.py --benchmark-steps 20      # 속도·VRAM 측정, 전체 소요 시간 추정
    nohup python -u train_hcx_think_32b.py > logs/train_hcx_think_32b.log 2>&1 &

중단 후 같은 명령으로 재실행하면 마지막 checkpoint부터 재개하고, 끝난 단계(학습/valid/test)는 건너뜁니다.
정제 CSV(train_/dev_ changes, delete_list)는 train.csv, dev.csv와 같은 폴더에 둡니다.
"""

# =====================================================================
# [모델 설정] — 스크립트별로 다른 부분은 여기뿐
# =====================================================================
MODEL_TAG = "hcx_think32b_bf16"
MODEL_NAME = "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"
LOADER = "hf"  # transformers AutoModelForImageTextToText + peft (unsloth 미사용)
LOAD_IN_4BIT = False  # bf16 LoRA (가중치 약 64GB)
GRAD_CKPT = True  # HF gradient checkpointing (use_reentrant=False)
MAX_SEQ_LEN = 4096
TRUST_REMOTE_CODE = (
    False  # transformers에 HyperCLOVAXVisionV2가 내장된 버전 필요. 구버전이면 True로
)

# 채팅 템플릿: 생성 프롬프트가 "assistant 줄바꿈 <think> 줄바꿈"으로 끝나 thinking이 열린 상태 (--check로 확인됨)
# → 빈 thinking 블록으로 닫아 "다음 토큰 = 정답 글자"가 되게 함. enable_thinking=False는 템플릿이 지원하면 적용됨
TEMPLATE_KWARGS = {"enable_thinking": False}
ANSWER_PREFIX = ""
CLOSE_OPEN_THINK = True

# 이미지 해상도: Qwen2.5-VL 비전 인코더, 비전 토큰 1개 = 28x28 px (patch 14 × merge 2)
VISION_PIXEL_SIDE = 28
MIN_VISION_TOKENS = 256
MAX_VISION_TOKENS = 1280
MAX_SOFT_TOKENS = None

# 배치 (B200 192GB 기준, 실효 16 = 16 x 1). bf16 가중치 약 66GB + 활성값
BATCH_SIZE_DEFAULT = 16
GRAD_ACCUM_DEFAULT = 1
INFER_BATCH_SIZE_DEFAULT = 16

# =====================================================================
# 이하 공통 코드 (3개 스크립트 동일) — 모델별 차이는 위 [모델 설정] 블록에만 있음
# =====================================================================
import argparse
import os
import re
import sys
from pathlib import Path

ALL_AUGS = ["orig", "compression", "blur", "affine"]  # 원본 + 증강 3종 = 4배


def parse_args():
    ap = argparse.ArgumentParser(description=f"{MODEL_TAG} 학습/추론")
    ap.add_argument(
        "--check",
        action="store_true",
        help="모델 로드 후 프롬프트·토큰·확률 점검만 하고 종료",
    )
    ap.add_argument(
        "--prepare-only",
        action="store_true",
        help="데이터 구성만 확인하고 종료 (모델 로드 없음)",
    )
    ap.add_argument(
        "--benchmark-steps",
        type=int,
        default=0,
        help=">0이면 N step 학습 속도 측정 후 종료",
    )
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--infer-batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--no-test", action="store_true", help="test 추론 생략 (디버그용)")
    ap.add_argument(
        "--no-skip-done", action="store_true", help="완료된 단계도 다시 실행"
    )
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--augs",
        nargs="+",
        default=None,
        choices=["orig", "compression", "blur", "affine"],
        help="대회 데이터 단계의 이미지 변형 (기본: 4종 전부 = 4배). 증강 없이: --augs orig",
    )
    ap.add_argument(
        "--grad-ckpt",
        choices=["default", "true", "unsloth", "false"],
        default="default",
        help="gradient checkpointing (default: 모델 설정값). 결과엔 영향 없고 속도·메모리만 바뀜",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker 수 (기본: vCPU 수 - 4, 최대 16)",
    )
    ap.add_argument(
        "--precision",
        choices=["preset", "bf16", "4bit"],
        default="preset",
        help="hf 로더 전용: 원본 가중치를 bf16 그대로 / bitsandbytes 4bit(NF4)로 즉석 양자화해 로드",
    )
    ap.add_argument(
        "--external",
        nargs="*",
        default=["mtvqa", "kreta"],
        choices=["mtvqa", "kreta"],
        help="외부 데이터 단계에 쓸 데이터 (기본: mtvqa kreta)",
    )
    ap.add_argument("--no-external", action="store_true", help="외부 데이터 단계 끄기")
    ap.add_argument("--ext-epochs", type=float, default=1.0)
    ap.add_argument(
        "--ext-lr", type=float, default=None, help="외부 단계 LR (기본: --lr과 동일)"
    )
    ap.add_argument("--mtvqa-splits", nargs="+", default=["train", "test"])
    return ap.parse_args()


ARGS = parse_args()
if ARGS.precision != "preset":
    if LOADER != "hf":
        sys.exit(
            "--precision은 hf 로더(HCX)에서만 지원합니다. unsloth 모델은 MODEL_NAME 자체가 정밀도를 결정합니다."
        )
    LOAD_IN_4BIT = ARGS.precision == "4bit"
    MODEL_TAG = re.sub(r"_(bf16|4bit)$", "", MODEL_TAG) + (
        "_4bit" if LOAD_IN_4BIT else "_bf16"
    )
if ARGS.grad_ckpt != "default":
    GRAD_CKPT = {"true": True, "unsloth": "unsloth", "false": False}[ARGS.grad_ckpt]

# ---------------------------------------------------------------------
# 공통 설정
# ---------------------------------------------------------------------
SEED = ARGS.seed
EPOCHS = ARGS.epochs
LR = ARGS.lr
BATCH_SIZE = ARGS.batch_size or BATCH_SIZE_DEFAULT
GRAD_ACCUM = ARGS.grad_accum or GRAD_ACCUM_DEFAULT
INFER_BATCH_SIZE = ARGS.infer_batch_size or INFER_BATCH_SIZE_DEFAULT
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
OPTIM = "adamw_8bit"
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 10
SAVE_STEPS = ARGS.save_steps
SAVE_TOTAL_LIMIT = 2
NUM_WORKERS = (
    ARGS.num_workers
    if ARGS.num_workers is not None
    else max(2, min(16, (os.cpu_count() or 8) - 4))
)

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

# 데이터
DEV_KEEP_PATTERNS = ["3", "3:1", "3:1:1"]  # dev 투표 패턴 (answer1~5 기준)
APPLY_OPTION_EDITS = True  # changes.csv의 edit_option(선지 수정) 적용
SHUFFLE_OPTIONS = True  # 학습 시 보기 순서 섞기 (샘플 index 기반 고정 시드)
AUGS = (
    list(dict.fromkeys(ARGS.augs)) if ARGS.augs else ALL_AUGS
)  # 기본: 원본 + compression + blur + affine
if AUGS != ALL_AUGS:
    MODEL_TAG += "_aug-" + "-".join(
        AUGS
    )  # 출력 폴더 분리 (예: final_hcx_think32b_4bit_aug-orig)

# 외부 데이터 단계 (external → target 순차 학습). 외부 데이터는 증강 없이 1배
EXTERNAL = [] if ARGS.no_external else list(dict.fromkeys(ARGS.external))
EXT_EPOCHS = ARGS.ext_epochs
EXT_LR = ARGS.ext_lr if ARGS.ext_lr is not None else LR
KRETA_REPO, KRETA_SPLIT = "tabtoyou/KRETA", "test"
MTVQA_REPO, MTVQA_SPLITS, MTVQA_LANGS = (
    "ByteDance/MTVQA",
    ARGS.mtvqa_splits,
    ["KO", "KR"],
)
MTVQA_MAX_ANSWER_CHARS = 40
MTVQA_DISTRACTOR_POOL = 50

# 증강 강도 (0924 증강 실험 노트북과 동일)
JPEG_QUALITY = 70
BLUR_RADIUS = 1.0
AFFINE_TRANSLATE = 0.03
AFFINE_SHEAR_DEG = 5.0
FILL_COLOR = (255, 255, 255)
LONGEST_SIDE_CAP = 2048  # 초대형 이미지 로딩·증강 속도용 상한 (학습·추론 동일 적용)

# 프롬프트 (2회 반복)
SYSTEM_PROMPT = (
    "당신은 이미지를 보고 질문에 답하는 시각 질의응답 도우미입니다. "
    "a, b, c, d 중 정확히 한 글자로만 답하고, 설명은 하지 마세요."
)
USER_INSTRUCTION = "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
PROMPT_REPEAT = 2
PROMPT_REPEAT_SEP = "\n\n"

# 추론
N_TTA = 2
TTA_SHIFTS = [0, 2, 1, 3]
SKIP_DONE = not ARGS.no_skip_done

# W&B
USE_WANDB = not ARGS.no_wandb and not (
    ARGS.check or ARGS.prepare_only or ARGS.benchmark_steps
)
WANDB_ENTITY = "min9lessmin9team"
WANDB_PROJECT = "stv"
WANDB_GROUP = "final-4xaug-rep2-tta2"

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
INTERACTIVE = sys.stderr.isatty()

# ---------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------
import zipfile, json

ON_RUNPOD = bool(os.environ.get("RUNPOD_POD_ID")) or Path("/workspace").is_dir()
_HERE = Path(__file__).resolve().parent
REPO_ROOT = next(
    (
        d
        for d in [Path.cwd(), *Path.cwd().parents, _HERE, *_HERE.parents]
        if (d / "eda" / "split_validation.py").exists()
    ),
    Path.cwd(),
)
ZIP_PATH = Path(
    os.environ.get(
        "STV_DATA_ZIP",
        "/workspace/stv/data.zip" if ON_RUNPOD else str(REPO_ROOT / "data.zip"),
    )
)
DATA_ROOT = Path(
    os.environ.get(
        "STV_DATA_ROOT", "/root/data" if ON_RUNPOD else str(REPO_ROOT / "data")
    )
)
RAW_BASE_DIR = DATA_ROOT / "raw"
SPLIT_DIR = DATA_ROOT / "split"
OUTPUT_DIR = Path(
    ARGS.output_dir
    or os.environ.get(
        "STV_OUTPUT_DIR",
        f"/workspace/stv/outputs/final_{MODEL_TAG}"
        if ON_RUNPOD
        else str(REPO_ROOT / "output" / f"final_{MODEL_TAG}"),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR = OUTPUT_DIR / "adapter"  # 최종(target 단계) LoRA
EXT_DIR = Path(
    os.environ.get(
        "STV_EXT_DIR",
        "/workspace/stv/external"
        if ON_RUNPOD
        else str(REPO_ROOT / "data" / "external"),
    )
)
CKPT_DIR = OUTPUT_DIR / "checkpoints"

os.environ.setdefault(
    "HF_HOME",
    "/workspace/.cache/huggingface" if ON_RUNPOD else str(REPO_ROOT / ".hf-cache"),
)
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # 학습 중에도 logits 반환 (직접 loss 계산)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHOICES = ["a", "b", "c", "d"]

print(
    "model :",
    MODEL_NAME,
    f"| loader={LOADER} | 4bit={LOAD_IN_4BIT} | grad_ckpt={GRAD_CKPT} | workers={NUM_WORKERS}",
)
print(
    f"batch : {BATCH_SIZE}x{GRAD_ACCUM} (실효 {BATCH_SIZE * GRAD_ACCUM}) | infer batch {INFER_BATCH_SIZE} | TTA {N_TTA}"
)
print("output:", OUTPUT_DIR)
print(
    f"stages: {'external(' + '+'.join(EXTERNAL) + f', lr={EXT_LR}, ep={EXT_EPOCHS}) → ' if EXTERNAL else ''}"
    f"target(lr={LR}, ep={EPOCHS})"
)

RUN_CONFIG = {
    "model": MODEL_NAME,
    "load_in_4bit": LOAD_IN_4BIT,
    "epochs": EPOCHS,
    "lr": LR,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "dev_keep_patterns": DEV_KEEP_PATTERNS,
    "apply_option_edits": APPLY_OPTION_EDITS,
    "augs": AUGS,
    "prompt_repeat": PROMPT_REPEAT,
    "shuffle_options": SHUFFLE_OPTIONS,
    "seed": SEED,
    "template_kwargs": TEMPLATE_KWARGS,
    "answer_prefix": ANSWER_PREFIX,
    "close_open_think": CLOSE_OPEN_THINK,
    "external": EXTERNAL,
    "ext_epochs": EXT_EPOCHS,
    "ext_lr": EXT_LR,
    "mtvqa_splits": MTVQA_SPLITS if "mtvqa" in EXTERNAL else None,
    "vision": {
        "pixel_side": VISION_PIXEL_SIDE,
        "min_tokens": MIN_VISION_TOKENS,
        "max_tokens": MAX_VISION_TOKENS,
        "max_soft_tokens": MAX_SOFT_TOKENS,
    },
}
_cfg_path = OUTPUT_DIR / "run_config.json"
if not (ARGS.check or ARGS.prepare_only or ARGS.benchmark_steps):
    if _cfg_path.exists():
        _old = json.loads(_cfg_path.read_text(encoding="utf-8"))
        _diff = {k: (_old.get(k), v) for k, v in RUN_CONFIG.items() if _old.get(k) != v}
        if _diff:
            print(f"[ERROR] {OUTPUT_DIR}에 다른 설정의 결과가 있습니다 (기존 → 현재):")
            for k, (a, b) in _diff.items():
                print(f"  {k}: {a} → {b}")
            print("폴더를 지우거나 --output-dir로 다른 폴더를 지정하세요.")
            sys.exit(2)
    else:
        _cfg_path.write_text(
            json.dumps(RUN_CONFIG, ensure_ascii=False, indent=1), encoding="utf-8"
        )

# ---------------------------------------------------------------------
# import
# ---------------------------------------------------------------------
if LOADER == "unsloth":
    import unsloth  # noqa: F401  (transformers/trl 패치를 위해 가장 먼저)

import logging
import gc, io, math, random, re, time, traceback, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from tqdm.auto import tqdm as _tqdm
import transformers
from transformers import Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint
from peft import set_peft_model_state_dict
from peft.utils import load_peft_weights

if USE_WANDB:
    import wandb


class _DropProcessorKwargsWarning(logging.Filter):
    """transformers 5.x가 배치마다 찍는 processor kwargs 경고만 숨김 (동작에는 영향 없음)."""

    def filter(self, record):
        return "processor_kwargs" not in record.getMessage()


for _h in transformers.utils.logging._get_library_root_logger().handlers:
    _h.addFilter(_DropProcessorKwargsWarning())


def tqdm(*a, **k):
    k.setdefault("mininterval", 0.1 if INTERACTIVE else 30)
    return _tqdm(*a, **k)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


seed_everything()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
Image.MAX_IMAGE_PIXELS = None
assert torch.cuda.is_available(), "CUDA GPU가 필요합니다."
print("GPU:", torch.cuda.get_device_name(), "| transformers", transformers.__version__)


# =====================================================================
# 1. 데이터
# =====================================================================
RAW_BASE_DIR.mkdir(parents=True, exist_ok=True)
if not list(RAW_BASE_DIR.rglob("train.csv")):
    assert ZIP_PATH.exists(), f"dataset zip not found: {ZIP_PATH}"
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(RAW_BASE_DIR)
RAW_DATA_DIR = sorted(RAW_BASE_DIR.rglob("train.csv"))[0].parent

sys.path.insert(0, str(REPO_ROOT / "eda"))
from split_validation import ensure_split  # noqa: E402

ensure_split(RAW_DATA_DIR, SPLIT_DIR)


def norm_key(v):
    return Path(str(v)).stem.lower()


def normalize_df(df, name, has_answer=True):
    need = {"id", "path", "question", *CHOICES} | ({"answer"} if has_answer else set())
    miss = need - set(df.columns)
    assert not miss, f"[{name}] missing columns: {sorted(miss)}"
    df = df.copy()
    df["id"] = df["id"].astype(str)
    for c in CHOICES:
        df[c] = df[c].astype(str)
    if has_answer:
        df["answer"] = df["answer"].astype(str).str.strip().str.lower()
    return df


def attach_img(df, roots, name):
    roots = [Path(r) for r in roots]

    def resolve(rel):
        for r in roots:
            p = r / str(rel)
            if p.exists():
                return str(p.resolve())
        return None

    df = df.copy()
    df["_img"] = [resolve(p) for p in df["path"]]
    miss = df["_img"].isna()
    assert not miss.any(), (
        f"[{name}] 이미지 {int(miss.sum())}개 없음. 예: {df.loc[miss, 'path'].head(3).tolist()}"
    )
    return df


train_df = attach_img(
    normalize_df(pd.read_csv(SPLIT_DIR / "train.csv"), "train"), [RAW_DATA_DIR], "train"
)
valid_df = attach_img(
    normalize_df(pd.read_csv(SPLIT_DIR / "validation.csv"), "valid"),
    [RAW_DATA_DIR],
    "valid",
)
test_df = attach_img(
    normalize_df(pd.read_csv(RAW_DATA_DIR / "test.csv"), "test", has_answer=False),
    [RAW_DATA_DIR],
    "test",
)
sample_sub = pd.read_csv(RAW_DATA_DIR / "sample_submission.csv")
assert train_df["answer"].isin(CHOICES).all() and valid_df["answer"].isin(CHOICES).all()
VALID_KEYS = set(valid_df["id"].map(norm_key))
VALID_IMGS = set(valid_df["_img"])


def drop_valid_leakage(df, name):
    drop = (
        df["id"].map(norm_key).isin(VALID_KEYS).to_numpy()
        | df["_img"].isin(VALID_IMGS).to_numpy()
    )
    if drop.any():
        print(f"[{name}] validation 중복 제거: {int(drop.sum())}")
    return df.loc[~drop].reset_index(drop=True)


def same_value(a, b):
    a, b = str(a).strip(), str(b).strip()
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def load_change_lists(split):
    """정제 CSV는 train.csv / dev.csv와 같은 폴더(RAW_DATA_DIR)에 둔다."""
    read = lambda n: pd.read_csv(
        RAW_DATA_DIR / f"{split}_{n}.csv", dtype=str, keep_default_na=False
    )
    for n in ("changes", "delete_list"):
        assert (RAW_DATA_DIR / f"{split}_{n}.csv").exists(), (
            f"{RAW_DATA_DIR / f'{split}_{n}.csv'} 없음"
        )
    changes, dele = read("changes"), read("delete_list")
    del_ids = set(dele["id"].map(norm_key)) | set(
        changes.loc[changes["action"] == "delete", "id"].map(norm_key)
    )
    edits = changes.loc[
        changes["action"] == "edit_option", ["id", "field", "before", "after"]
    ].copy()
    edits["key"] = edits["id"].map(norm_key)
    edits["field"] = edits["field"].str.strip().str.lower()
    assert edits["field"].isin(CHOICES).all()
    print(f"[{split}] 삭제 목록 {len(del_ids)} | 선지 수정 {len(edits)}")
    return del_ids, edits


def apply_edits(df, edits, name):
    if not APPLY_OPTION_EDITS or edits.empty:
        return df
    df = df.copy()
    keys = df["id"].map(norm_key)
    n_ok = 0
    for e in edits.to_dict("records"):
        m = (keys == e["key"]).to_numpy()
        if not m.any():
            continue
        cur = df.loc[m, e["field"]].iloc[0]
        assert same_value(cur, e["before"]), (
            f"[{name}] {e['id']} 보기 {e['field']}: 현재 {cur!r} != before {e['before']!r}"
        )
        df.loc[m, e["field"]] = e["after"]
        n_ok += 1
    print(f"[{name}] 선지 수정 적용 {n_ok}건")
    return df


def build_train_part():
    del_ids, edits = load_change_lists("train")
    df = apply_edits(train_df.assign(_src="train"), edits, "train")
    drop = df["id"].map(norm_key).isin(del_ids)
    df = df.loc[~drop].reset_index(drop=True)
    print(f"[train] {len(train_df)} → 삭제 {int(drop.sum())} → {len(df)}")
    return df


def dev_votes(row, cols):
    letters = []
    for c in cols:
        v = row[c]
        if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
            continue
        t = str(v).strip().lower()
        t = {
            "1": "a",
            "2": "b",
            "3": "c",
            "4": "d",
            "1.0": "a",
            "2.0": "b",
            "3.0": "c",
            "4.0": "d",
        }.get(t, t)
        assert t in CHOICES, f"{row['id']} {c}: 해석 불가 응답 {v!r}"
        letters.append(t)
    return [letters.count(c) for c in CHOICES]


def build_dev_part():
    dev_csv = RAW_DATA_DIR / "dev.csv"
    assert dev_csv.exists(), f"{dev_csv} 없음"
    dev = normalize_df(pd.read_csv(dev_csv), "dev", has_answer=False)
    cols = sorted(
        (c for c in dev.columns if re.match(r"^answer\d+$", str(c))),
        key=lambda c: int(c[6:]),
    )
    assert cols, f"dev.csv에 answer1~N 컬럼이 없습니다: {list(dev.columns)}"
    votes = np.array([dev_votes(r, cols) for _, r in dev.iterrows()])
    dev["_pattern"] = [
        ":".join(str(n) for n in sorted((x for x in v if x > 0), reverse=True))
        for v in votes
    ]
    dev["answer"] = [CHOICES[int(np.argmax(v))] for v in votes]
    print("[dev] 투표 패턴 분포:\n" + dev["_pattern"].value_counts().to_string())

    sel = dev.loc[dev["_pattern"].isin(DEV_KEEP_PATTERNS)].copy()
    sel["_src"] = "dev_" + sel["_pattern"]
    del_ids, edits = load_change_lists("dev")
    sel = apply_edits(sel, edits, "dev")
    drop = sel["id"].map(norm_key).isin(del_ids)
    sel = sel.loc[~drop].reset_index(drop=True)
    sel = attach_img(sel, [dev_csv.parent, RAW_DATA_DIR, DATA_ROOT], "dev")
    sel = drop_valid_leakage(sel, "dev")
    print(
        f"[dev] {len(dev)} → 패턴 {DEV_KEEP_PATTERNS} 선택 → 손라벨링 삭제 {int(drop.sum())} → {len(sel)}"
    )
    return sel


KEEP_COLS = ["id", "path", "question", *CHOICES, "answer", "_img", "_src"]
base_df = pd.concat([build_train_part(), build_dev_part()], ignore_index=True)[
    KEEP_COLS
]
assert not base_df["id"].duplicated().any()
assert base_df["answer"].isin(CHOICES).all()
base_df.to_csv(OUTPUT_DIR / "train_base_rows.csv", index=False, encoding="utf-8-sig")
print(
    f"\n학습 원본 {len(base_df)}행 × 증강 {len(AUGS)}종 = {len(base_df) * len(AUGS)} 샘플/epoch"
)
print("출처:", base_df["_src"].value_counts().to_dict())
print(f"valid={len(valid_df)} | test={len(test_df)}")


# ---------------------------------------------------------------------
# 외부 데이터 (KRETA / MTVQA-KO) — 변환 결과를 EXT_DIR에 캐시 (run_transfer.py와 같은 형식)
# ---------------------------------------------------------------------
def safe_name(s):
    return re.sub(r"[^0-9A-Za-z_\-]", "_", str(s))


def load_kreta():
    cache = EXT_DIR / "kreta" / "kreta.csv"
    img_dir = EXT_DIR / "kreta" / "images"
    if not cache.exists():
        import base64
        from collections import Counter
        from datasets import load_dataset

        img_dir.mkdir(parents=True, exist_ok=True)
        rows, skipped, seen = [], Counter(), set()
        for r in tqdm(load_dataset(KRETA_REPO, split=KRETA_SPLIT), desc="KRETA"):
            opts = [r.get(k) for k in "ABCD"]
            ans = str(r.get("answer") or "").strip().upper()
            if any(o is None or str(o).strip() in ("", "nan", "None") for o in opts):
                skipped["missing_option"] += 1
                continue
            if ans not in ("A", "B", "C", "D"):
                skipped["bad_answer"] += 1
                continue
            if len({str(o).strip() for o in opts}) < 4:
                skipped["duplicate_option"] += 1
                continue
            rid = safe_name(r["id"])
            while rid in seen:
                rid += "_"
            seen.add(rid)
            p = img_dir / f"{rid}.jpg"
            if not p.exists():
                b64 = str(r["image"])
                if b64.startswith("data:"):
                    b64 = b64.split(",", 1)[1]
                Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").save(
                    p, quality=95
                )
            rows.append(
                {
                    "id": f"kreta_{rid}",
                    "path": str(p.relative_to(EXT_DIR)),
                    "question": str(r["question"]).strip(),
                    **{c: str(o).strip() for c, o in zip(CHOICES, opts)},
                    "answer": ans.lower(),
                    "topic_difficulty": r.get("topic_difficulty"),
                    "image_type": r.get("image_type"),
                }
            )
        pd.DataFrame(rows).to_csv(cache, index=False, encoding="utf-8-sig")
        print(f"[kreta] 저장 {len(rows)} | 제외 {dict(skipped)}")
    return pd.read_csv(cache, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def mt_norm(s):
    return re.sub(r"[\s\W_]+", "", str(s)).lower()


def has_digit(s):
    return any(ch.isdigit() for ch in str(s))


def load_mtvqa_qa():
    cache = EXT_DIR / "mtvqa" / "mtvqa_ko_qa.csv"
    img_dir = EXT_DIR / "mtvqa" / "images"
    if not cache.exists():
        import ast
        from datasets import load_dataset

        img_dir.mkdir(parents=True, exist_ok=True)
        langs = {l.upper() for l in MTVQA_LANGS}
        recs = []
        for split in MTVQA_SPLITS:
            ds = load_dataset(MTVQA_REPO, split=split)
            idx = [i for i, l in enumerate(ds["lang"]) if str(l).upper() in langs]
            print(f"[mtvqa] {split}: {len(idx)} images")
            for r in tqdm(ds.select(idx), desc=f"MTVQA {split}"):
                img_id = f"{split}_{safe_name(r['id'])}"
                p = img_dir / f"{img_id}.jpg"
                if not p.exists():
                    r["image"].convert("RGB").save(p, quality=95)
                qa = r["qa_pairs"]
                if isinstance(qa, str):
                    qa = ast.literal_eval(qa)
                for k, item in enumerate(qa):
                    recs.append(
                        {
                            "img_id": img_id,
                            "k": k,
                            "path": str(p.relative_to(EXT_DIR)),
                            "question": " ".join(str(item.get("question", "")).split()),
                            "answer_text": " ".join(
                                str(item.get("answer", "")).split()
                            ),
                        }
                    )
        pd.DataFrame(recs).to_csv(cache, index=False, encoding="utf-8-sig")
    return pd.read_csv(cache, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def build_mtvqa_mc(qa):
    """주관식 → 4지선다. 오답: 같은 이미지의 다른 답 → 같은 유형(숫자 포함 여부)·비슷한 길이의 답."""
    n0 = len(qa)
    qa = qa[
        (qa["question"].str.len() > 0)
        & (qa["answer_text"].str.len() > 0)
        & (qa["answer_text"].str.len() <= MTVQA_MAX_ANSWER_CHARS)
    ].copy()
    qa["norm"] = qa["answer_text"].map(mt_norm)
    qa = qa[qa["norm"].str.len() > 0].reset_index(drop=True)
    uniq = qa.drop_duplicates("norm")
    pool = {True: [], False: []}
    for a, n in zip(uniq["answer_text"], uniq["norm"]):
        pool[has_digit(a)].append((a, n, len(a)))
    by_img = qa.groupby("img_id")["answer_text"].apply(list).to_dict()
    ok = lambda n, used, g: n and n not in used and n not in g and g not in n

    rows = []
    for i, r in enumerate(qa.to_dict("records")):
        rng = random.Random(SEED * 7919 + i)
        gold, gn = r["answer_text"], r["norm"]
        opts, used = [gold], {gn}
        same = list(by_img[r["img_id"]])
        rng.shuffle(same)
        for a in same:
            n = mt_norm(a)
            if ok(n, used, gn):
                opts.append(a)
                used.add(n)
            if len(opts) == 4:
                break
        if len(opts) < 4:
            cands = sorted(
                (c for c in pool[has_digit(gold)] if ok(c[1], used, gn)),
                key=lambda c: abs(c[2] - len(gold)),
            )
            near = cands[:MTVQA_DISTRACTOR_POOL]
            rng.shuffle(near)
            for a, n, _ in near:
                if n not in used:
                    opts.append(a)
                    used.add(n)
                if len(opts) == 4:
                    break
        if len(opts) < 4:
            continue
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        rows.append(
            {
                "id": f"mtvqa_{r['img_id']}_{r['k']}",
                "path": r["path"],
                "question": r["question"],
                **dict(zip(CHOICES, [opts[j] for j in order])),
                "answer": CHOICES[order.index(0)],
            }
        )
    print(f"[mtvqa] QA {n0} → 4지선다 {len(rows)}")
    return pd.DataFrame(rows)


EXT_LOADERS = {"kreta": load_kreta, "mtvqa": lambda: build_mtvqa_mc(load_mtvqa_qa())}
ext_df = None
if EXTERNAL:
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for name in EXTERNAL:
        d = attach_img(normalize_df(EXT_LOADERS[name](), name), [EXT_DIR], name).assign(
            _src=name
        )
        parts.append(drop_valid_leakage(d, name))
    ext_df = pd.concat(parts, ignore_index=True)[KEEP_COLS]
    assert ext_df["answer"].isin(CHOICES).all() and not ext_df["id"].duplicated().any()
    ext_df.to_csv(OUTPUT_DIR / "external_rows.csv", index=False, encoding="utf-8-sig")
    print(
        f"외부 데이터 {len(ext_df)}행 (증강 없음): {ext_df['_src'].value_counts().to_dict()}"
    )

# 학습 단계: external(외부, 1배) → target(대회 데이터, 4배)
STAGES = []
if EXTERNAL:
    STAGES.append(
        {
            "name": "external",
            "df": ext_df,
            "augs": ["orig"],
            "epochs": EXT_EPOCHS,
            "lr": EXT_LR,
        }
    )
STAGES.append(
    {"name": "target", "df": base_df, "augs": AUGS, "epochs": EPOCHS, "lr": LR}
)
for st in STAGES:
    print(
        f"  stage {st['name']:<8} {len(st['df']) * len(st['augs']):>6} 샘플 | lr={st['lr']} | epochs={st['epochs']}"
    )

if ARGS.prepare_only:
    print("--prepare-only: 종료")
    sys.exit(0)


# =====================================================================
# 2. 모델 로드 & LoRA
# =====================================================================
LORA_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
VISION_KEYWORDS = (
    "visual",
    "vision",
    "mm_projector",
    "multi_modal_projector",
    "projector",
    "merger",
    "audio",
    "embed_vision",
)


def find_lora_targets(model):
    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        low = name.lower()
        if any(k in low for k in VISION_KEYWORDS) or low.endswith("lm_head"):
            continue
        if name.split(".")[-1] in LORA_TARGET_SUFFIXES:
            names.append(name)
    assert names, "LoRA 대상 Linear 레이어를 찾지 못함 → LORA_TARGET_SUFFIXES 확인"
    print(
        "LoRA 대상:", dict(pd.Series([n.split(".")[-1] for n in names]).value_counts())
    )
    return names


def load_model():
    if LOADER == "unsloth":
        from unsloth import FastVisionModel

        model, processor = FastVisionModel.from_pretrained(
            MODEL_NAME,
            load_in_4bit=LOAD_IN_4BIT,
            use_gradient_checkpointing=GRAD_CKPT,
            max_seq_length=MAX_SEQ_LEN,
        )
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            random_state=SEED,
            use_gradient_checkpointing=GRAD_CKPT,  # get_peft_model 기본값("unsloth")이 덮어쓰지 않도록 명시
        )
        return model, processor

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model
    from packaging import version

    dtype_kw = (
        {"dtype": torch.bfloat16}
        if version.parse(transformers.__version__) >= version.parse("4.56")
        else {"torch_dtype": torch.bfloat16}
    )
    quant_kw = {}
    if LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig

        # 언어 모델 Linear만 NF4로 양자화. 비전 인코더·projector·lm_head는 bf16 유지
        quant_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=[*VISION_KEYWORDS, "lm_head"],
        )
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=TRUST_REMOTE_CODE,
        **dtype_kw,
        **quant_kw,
    )
    model.config.use_cache = False
    if LOAD_IN_4BIT:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(GRAD_CKPT),
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif GRAD_CKPT:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=find_lora_targets(model),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    return model, processor


def set_mode(train):
    if LOADER == "unsloth":
        from unsloth import FastVisionModel

        (FastVisionModel.for_training if train else FastVisionModel.for_inference)(
            model
        )
    else:
        model.train(train)


def _safe_set(obj, key, value):
    """속성이 있고 쓸 수 있을 때만 설정 (transformers 5.x는 일부가 읽기 전용 property)."""
    if obj is None or not hasattr(obj, key):
        return False
    try:
        setattr(obj, key, value)
        return True
    except (AttributeError, TypeError):
        return False


def configure_processor(processor):
    ip = getattr(processor, "image_processor", None)
    if VISION_PIXEL_SIDE:  # Qwen 계열 image processor (총 픽셀 수 범위)
        lo = MIN_VISION_TOKENS * VISION_PIXEL_SIDE**2
        hi = MAX_VISION_TOKENS * VISION_PIXEL_SIDE**2
        lo, hi = int(lo), int(hi)
        size = getattr(ip, "size", None)
        if isinstance(size, dict):
            ip.size = {**size, "shortest_edge": lo, "longest_edge": hi}
        elif size is not None:  # transformers 5.x: SizeDict 객체 → 필드를 직접 수정
            if not (
                _safe_set(size, "shortest_edge", lo)
                and _safe_set(size, "longest_edge", hi)
            ):
                ip.size = type(size)(shortest_edge=lo, longest_edge=hi)
        for k, v in (("min_pixels", lo), ("max_pixels", hi)):
            _safe_set(ip, k, v)
        size = getattr(ip, "size", None)
        got = (
            size.get("shortest_edge")
            if isinstance(size, dict)
            else getattr(size, "shortest_edge", None),
            size.get("longest_edge")
            if isinstance(size, dict)
            else getattr(size, "longest_edge", None),
        )
        print(
            f"[vision] size={size} | min_pixels={getattr(ip, 'min_pixels', None)} "
            f"| max_pixels={getattr(ip, 'max_pixels', None)} (목표 {lo}~{hi})"
        )
        assert got == (lo, hi), f"이미지 해상도 설정 실패: {got} != {(lo, hi)}"
    if MAX_SOFT_TOKENS:  # Gemma 4 비전 토큰 예산 (70/140/280/560/1120)
        for obj in (ip, processor):
            for k in ("max_soft_tokens", "image_seq_length"):
                _safe_set(obj, k, MAX_SOFT_TOKENS)
        for cfg in (model.config, getattr(model.config, "vision_config", None)):
            for k in ("vision_soft_tokens_per_image", "default_output_length"):
                _safe_set(cfg, k, MAX_SOFT_TOKENS)
        print(
            f"[vision] max_soft_tokens={getattr(ip, 'max_soft_tokens', None)} "
            f"| image_seq_length={getattr(processor, 'image_seq_length', getattr(ip, 'image_seq_length', None))}"
        )
    tok = getattr(processor, "tokenizer", processor)
    tok.padding_side = "left"  # 학습·추론 모두 왼쪽 패딩 → 마지막 위치 = 정답 예측 위치
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


model, processor = load_model()
tokenizer = configure_processor(processor)
model.print_trainable_parameters()

CHOICE_IDS = [tokenizer.convert_tokens_to_ids(c) for c in CHOICES]
assert all(tokenizer.decode([t]) == c for t, c in zip(CHOICE_IDS, CHOICES)), (
    f"a/b/c/d가 단일 토큰이 아님: {CHOICE_IDS} → {[tokenizer.decode([t]) for t in CHOICE_IDS]}"
)
print("choice token ids:", dict(zip(CHOICES, CHOICE_IDS)))
print(f"GPU memory after load: {torch.cuda.memory_allocated() / 2**30:.1f} GB")


# =====================================================================
# 3. 프롬프트 / 증강 / Dataset / Collator
# =====================================================================
def load_image(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > LONGEST_SIDE_CAP:
        s = LONGEST_SIDE_CAP / max(w, h)
        img = img.resize(
            (max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.BICUBIC
        )
    return img


def augment(img, aug, rng):
    if aug == "orig":
        return img
    if aug == "compression":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        out.load()
        return out
    if aug == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    if aug == "affine":
        w, h = img.size
        tx = rng.uniform(-AFFINE_TRANSLATE, AFFINE_TRANSLATE) * w
        ty = rng.uniform(-AFFINE_TRANSLATE, AFFINE_TRANSLATE) * h
        sx = math.tan(math.radians(rng.uniform(-AFFINE_SHEAR_DEG, AFFINE_SHEAR_DEG)))
        sy = math.tan(math.radians(rng.uniform(-AFFINE_SHEAR_DEG, AFFINE_SHEAR_DEG)))
        return img.transform(
            img.size,
            Image.Transform.AFFINE,
            (1.0, sx, tx, sy, 1.0, ty),
            resample=Image.Resampling.BICUBIC,
            fillcolor=FILL_COLOR,
        )
    raise ValueError(aug)


def build_messages(row, image, options):
    body = (
        f"{row['question']}\n"
        f"(a) {options[0]}\n(b) {options[1]}\n(c) {options[2]}\n(d) {options[3]}\n\n"
        f"{USER_INSTRUCTION}"
    )
    prompt = PROMPT_REPEAT_SEP.join([body] * PROMPT_REPEAT)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def render(messages):
    """생성 프롬프트 + (필요 시) 빈 thinking 블록. 이 문자열 다음 토큰이 정답 글자."""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **TEMPLATE_KWARGS
    )
    if CLOSE_OPEN_THINK and text.endswith("<think>\n"):
        text += "\n</think>\n\n"  # 열린 <think>를 빈 블록으로 닫음 → "<think>\n\n</think>\n\n"
    if ANSWER_PREFIX and not text.endswith(ANSWER_PREFIX):
        text += ANSWER_PREFIX
    return text


IMAGES_NESTED = None  # processor가 이미지 인자를 [[img], ...]로 받는지 [img, ...]로 받는지 (첫 호출 때 결정)


def call_processor(texts, images):
    global IMAGES_NESTED
    kw = dict(text=texts, padding=True, return_tensors="pt", add_special_tokens=False)
    if IMAGES_NESTED is None:
        for nested in (False, True):
            try:
                enc = processor(
                    images=[[i] for i in images] if nested else images, **kw
                )
                IMAGES_NESTED = nested
                print(f"[processor] images nested={nested}")
                return enc
            except Exception as e:
                last = e
        raise last
    return processor(images=[[i] for i in images] if IMAGES_NESTED else images, **kw)


class TrainDataset(Dataset):
    """index = 원본행 × 증강종류. 이미지·보기 순서는 index 기반 고정 시드로 재현 가능."""

    def __init__(self, df, augs):
        self.df = df.reset_index(drop=True)
        self.augs = list(augs)

    def __len__(self):
        return len(self.df) * len(self.augs)

    def __getitem__(self, i):
        row = self.df.iloc[i // len(self.augs)]
        aug = self.augs[i % len(self.augs)]
        rng = random.Random(SEED * 1_000_003 + i)
        img = augment(load_image(row["_img"]), aug, rng)
        order = list(range(4))
        if SHUFFLE_OPTIONS:
            rng.shuffle(order)
        options = [row[CHOICES[j]] for j in order]
        gold = order.index(CHOICES.index(row["answer"]))
        return {
            "text": render(build_messages(row, img, options)),
            "image": img,
            "target": CHOICE_IDS[gold],
        }


def collate(features):
    enc = call_processor([f["text"] for f in features], [f["image"] for f in features])
    enc = dict(enc)
    enc["target_ids"] = torch.tensor([f["target"] for f in features], dtype=torch.long)
    return enc


LOGITS_TO_KEEP_OK = None


def last_logits(batch):
    """마지막 위치(= 정답 글자 예측 위치)의 전체 어휘 로짓 (B, V)."""
    global LOGITS_TO_KEEP_OK
    if LOGITS_TO_KEEP_OK is not False:
        try:
            out = model(**batch, logits_to_keep=1, use_cache=False)
            LOGITS_TO_KEEP_OK = True
            return out.logits[:, -1, :]
        except TypeError:
            LOGITS_TO_KEEP_OK = False
            print("[warn] logits_to_keep 미지원 → 전체 logits 계산")
    return model(**batch, use_cache=False).logits[:, -1, :]


def image_token_count(input_ids):
    tid = getattr(model.config, "image_token_id", None) or getattr(
        model.config, "image_token_index", None
    )
    if tid is None:
        return None
    return (input_ids == tid).sum(1).tolist()


# =====================================================================
# 4. 추론 (TTA)
# =====================================================================
def to_device(enc):
    return {
        k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in enc.items()
    }


@torch.inference_mode()
def predict_probs(df, desc):
    shifts = TTA_SHIFTS[:N_TTA]
    probs = np.zeros((len(df), 4), dtype=np.float32)
    set_mode(False)
    for s in range(0, len(df), INFER_BATCH_SIZE):
        rows = [r for _, r in df.iloc[s : s + INFER_BATCH_SIZE].iterrows()]
        images = [load_image(r["_img"]) for r in rows]
        acc = np.zeros((len(rows), 4), dtype=np.float32)
        for sh in shifts:
            order = [(j + sh) % 4 for j in range(4)]
            texts = [
                render(build_messages(r, img, [r[CHOICES[o]] for o in order]))
                for r, img in zip(rows, images)
            ]
            enc = to_device(call_processor(texts, images))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = last_logits(enc)[:, CHOICE_IDS]
            acc[:, order] += torch.softmax(logits.float(), dim=-1).cpu().numpy()
        probs[s : s + len(rows)] = acc / len(shifts)
        if (s // INFER_BATCH_SIZE) % max(1, 200 // INFER_BATCH_SIZE) == 0:
            print(f"  [{desc}] {s + len(rows)}/{len(df)}")
    return probs


@torch.inference_mode()
def sanity_check(n=8):
    """생성 프롬프트 끝부분, 이미지 토큰 수, 학습 전 a/b/c/d 확률 질량 점검."""
    set_mode(False)
    rows = [r for _, r in valid_df.head(n).iterrows()]
    images = [load_image(r["_img"]) for r in rows]
    texts = [
        render(build_messages(r, img, [r[c] for c in CHOICES]))
        for r, img in zip(rows, images)
    ]
    print("\n[check] 생성 프롬프트 끝 200자:\n" + repr(texts[0][-200:]))
    enc = to_device(call_processor(texts, images))
    print(
        f"[check] seq len={enc['input_ids'].shape[1]} | 이미지 토큰 수={image_token_count(enc['input_ids'])}"
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = last_logits(enc).float()
    p = torch.softmax(logits, dim=-1)
    mass = p[:, CHOICE_IDS].sum(1)
    top = torch.topk(p[0], 5)
    print(
        f"[check] a/b/c/d 확률 질량(학습 전) 평균 {mass.mean():.3f} | 샘플별 {[round(x, 3) for x in mass.tolist()]}"
    )
    print(
        "[check] 첫 샘플 top-5 다음 토큰:",
        [
            (tokenizer.decode([int(i)]), round(float(v), 3))
            for v, i in zip(top.values, top.indices)
        ],
    )
    if mass.mean() < 0.3:
        print(
            "[WARN] 정답 글자 확률이 낮습니다. 생성 프롬프트 끝(thinking 블록/ANSWER_PREFIX)을 확인하세요. "
            "학습하면 적응은 하지만, 템플릿이 틀렸을 가능성이 큽니다."
        )
    return float(mass.mean())


# =====================================================================
# 5. 학습
# =====================================================================
class LastTokenTrainer(Trainer):
    """마지막 위치 로짓과 정답 글자 토큰의 cross-entropy (어휘 전체 softmax)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.model_accepts_loss_kwargs = (
            False  # micro-batch 평균 loss를 grad_accum으로 나누도록 고정
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        targets = inputs.pop("target_ids")
        logits = last_logits(inputs)
        loss = F.cross_entropy(logits.float(), targets)
        return (loss, None) if return_outputs else loss


def training_args(output_dir, st, max_steps=-1, save=True):
    # warmup_ratio는 transformers 5.x에서 deprecated → 전체 step 수로 warmup_steps(정수)를 직접 계산
    n_samples = len(st["df"]) * len(st["augs"])
    total_steps = (
        max_steps
        if max_steps > 0
        else math.ceil(math.ceil(n_samples / BATCH_SIZE) / GRAD_ACCUM * st["epochs"])
    )
    warmup_steps = max(1, round(WARMUP_RATIO * total_steps))
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=st["epochs"],
        max_steps=max_steps,
        learning_rate=st["lr"],
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        optim=OPTIM,
        max_grad_norm=MAX_GRAD_NORM,
        bf16=True,
        tf32=True,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps" if save else "no",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="wandb" if (USE_WANDB and save) else "none",
        run_name=f"{MODEL_TAG}-{st['name']}",
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
        dataloader_num_workers=NUM_WORKERS,
        disable_tqdm=not INTERACTIVE,
        gradient_checkpointing=False,  # 로더에서 이미 설정
    )


def make_trainer(st, args):
    return LastTokenTrainer(
        model=model,
        args=args,
        train_dataset=TrainDataset(st["df"], st["augs"]),
        data_collator=collate,
    )


def stage_adapter_dir(name):
    return ADAPTER_DIR if name == "target" else OUTPUT_DIR / f"adapter_{name}"


def train_stage(st):
    adapter = stage_adapter_dir(st["name"])
    marker, ckpt = adapter / "done.json", CKPT_DIR / st["name"]
    if SKIP_DONE and marker.exists():
        set_peft_model_state_dict(model, load_peft_weights(str(adapter)))
        print(f"[skip] {st['name']} 단계 완료 → adapter 로드: {adapter}")
        return json.loads(marker.read_text(encoding="utf-8"))

    print(
        f"\n===== stage: {st['name']} | {len(st['df']) * len(st['augs'])} 샘플 | lr={st['lr']} | epochs={st['epochs']} ====="
    )
    set_mode(True)
    trainer = make_trainer(st, training_args(ckpt, st))
    last = get_last_checkpoint(str(ckpt)) if ckpt.is_dir() else None
    if last:
        print("checkpoint에서 재개:", last)
    t0 = time.time()
    res = trainer.train(resume_from_checkpoint=last)
    info = {
        "stage": st["name"],
        "train_loss": round(float(res.training_loss), 4),
        "train_hours": round((time.time() - t0) / 3600, 2),
        "steps": int(trainer.state.global_step),
        "samples": len(st["df"]) * len(st["augs"]),
        "lr": st["lr"],
        "epochs": st["epochs"],
    }
    model.save_pretrained(adapter)
    processor.save_pretrained(adapter)
    marker.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    print(f"saved {st['name']} adapter:", adapter, info)
    if USE_WANDB:
        wandb.log(
            {
                f"stage/{st['name']}/train_loss": info["train_loss"],
                f"stage/{st['name']}/hours": info["train_hours"],
            }
        )
    notify(
        f"{st['name']} 단계 학습 완료",
        f"loss {info['train_loss']} | {info['train_hours']} h | {info['steps']} step\nadapter: {adapter}",
    )
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return info


def train():
    if (
        SKIP_DONE and (ADAPTER_DIR / "done.json").exists()
    ):  # 최종 단계가 끝났으면 앞 단계는 볼 필요 없음
        return [train_stage(STAGES[-1])]
    return [train_stage(st) for st in STAGES]


def benchmark(n_steps):
    st = STAGES[-1]  # 대회 데이터(증강 포함) 단계로 측정
    set_mode(True)
    trainer = make_trainer(
        st, training_args(OUTPUT_DIR / "_bench", st, max_steps=n_steps, save=False)
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    sec = (time.time() - t0) / n_steps
    peak = torch.cuda.max_memory_allocated() / 2**30
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(OUTPUT_DIR / "_bench", ignore_errors=True)
    n = min(32, len(valid_df))
    t1 = time.time()
    predict_probs(valid_df.iloc[:n], "bench")
    sps = (time.time() - t1) / n
    print(
        f"\n[benchmark] {sec:.1f} s/step | peak GPU {peak:.1f} GB | 추론 {sps:.2f} s/sample (TTA {N_TTA})"
    )
    total = 0.0
    for s_ in STAGES:
        steps = math.ceil(
            len(s_["df"]) * len(s_["augs"]) / (BATCH_SIZE * GRAD_ACCUM) * s_["epochs"]
        )
        total += steps * sec / 3600
        print(f"  {s_['name']:<8} {steps} step ≈ {steps * sec / 3600:.1f} h")
    inf_h = sps * (len(valid_df) + len(test_df)) / 3600
    print(f"  valid+test 추론 ≈ {inf_h:.1f} h | 전체 ≈ {total + inf_h:.1f} h")


# =====================================================================
# 6. 실행
# =====================================================================
if ARGS.check:
    sanity_check()
    sys.exit(0)
if ARGS.benchmark_steps > 0:
    sanity_check()
    benchmark(ARGS.benchmark_steps)
    sys.exit(0)


def notify(title, text, level="INFO"):
    """W&B 알림. 수신하려면 wandb.ai/settings → Alerts에서 Email(또는 Slack)과 Scriptable run alerts를 켜야 함."""
    if not (USE_WANDB and wandb.run is not None):
        return
    try:
        lv = getattr(wandb.AlertLevel, level, wandb.AlertLevel.INFO)
        alert = getattr(wandb.run, "alert", None) or wandb.alert
        alert(title=f"[{MODEL_TAG}] {title}", text=text, level=lv)
    except Exception as e:  # 알림 실패가 학습을 멈추지 않도록
        print(f"[warn] W&B 알림 실패: {e}")


def _crash_hook(exc_type, exc, tb):
    """처리되지 않은 예외(OOM 등)로 죽을 때 알림을 보내고 run을 실패 상태로 종료."""
    msg = "".join(traceback.format_exception(exc_type, exc, tb))
    if not issubclass(exc_type, KeyboardInterrupt):
        notify("실패", f"{exc_type.__name__}: {exc}\n\n{msg[-1500:]}", level="ERROR")
    if USE_WANDB and wandb.run is not None:
        wandb.finish(exit_code=1)
    sys.__excepthook__(exc_type, exc, tb)


if USE_WANDB:
    sys.excepthook = _crash_hook
    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=MODEL_TAG,
        config={
            **RUN_CONFIG,
            "n_train_base": len(base_df),
            "n_external": 0 if ext_df is None else len(ext_df),
            "n_tta": N_TTA,
        },
    )

choice_mass = sanity_check()
train_info = train()

# ---- validation (soft label) ----
valid_npy = OUTPUT_DIR / "valid_probs.npy"
if SKIP_DONE and valid_npy.exists():
    valid_probs = np.load(valid_npy)
    print("[skip] validation 확률 로드")
else:
    valid_probs = predict_probs(valid_df, "validation")
    np.save(valid_npy, valid_probs)
gold = valid_df["answer"].map(CHOICES.index).to_numpy()
valid_acc = float((valid_probs.argmax(1) == gold).mean())
valid_out = pd.DataFrame(
    {
        "id": valid_df["id"].values,
        **{c: valid_probs[:, j] for j, c in enumerate(CHOICES)},
    }
)
valid_out["pred"] = [CHOICES[i] for i in valid_probs.argmax(1)]
valid_out["answer"] = valid_df["answer"].values
valid_out.to_csv(OUTPUT_DIR / "valid_soft.csv", index=False, encoding="utf-8-sig")
print(f"validation accuracy (TTA={N_TTA}): {valid_acc:.4f}")

# ---- test (hard + soft) ----
if not ARGS.no_test:
    test_npy = OUTPUT_DIR / "test_probs.npy"
    if SKIP_DONE and test_npy.exists():
        test_probs = np.load(test_npy)
        print("[skip] test 확률 로드")
    else:
        test_probs = predict_probs(test_df, "test")
        np.save(test_npy, test_probs)
    ids = test_df["id"].values
    assert (
        len(ids) == len(sample_sub)
        and (ids.astype(str) == sample_sub["id"].astype(str).values).all()
    )
    hard = pd.DataFrame(
        {"id": ids, "answer": [CHOICES[i] for i in test_probs.argmax(1)]}
    )
    soft = pd.DataFrame(
        {"id": ids, **{c: test_probs[:, j] for j, c in enumerate(CHOICES)}}
    )
    hard.to_csv(OUTPUT_DIR / "submission_hard.csv", index=False)
    soft.to_csv(OUTPUT_DIR / "submission_soft.csv", index=False)
    print(
        "saved:",
        OUTPUT_DIR / "submission_hard.csv",
        "|",
        OUTPUT_DIR / "submission_soft.csv",
    )

metrics = {
    "model": MODEL_NAME,
    "valid_acc": valid_acc,
    "choice_mass_before_train": choice_mass,
    "stages": train_info,
}
(OUTPUT_DIR / "metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8"
)
notify(
    "전체 완료",
    f"validation acc (TTA={N_TTA}): {valid_acc:.4f}\n"
    f"학습 시간: {sum(float(i.get('train_hours', 0)) for i in train_info):.1f} h\n"
    f"출력: {OUTPUT_DIR}\n(submission_hard.csv / submission_soft.csv / valid_soft.csv)",
)
if USE_WANDB:
    wandb.log({"validation/accuracy": valid_acc})
    wandb.finish()
print(f"\nALL DONE ({time.strftime('%Y-%m-%d %H:%M:%S')}) → {OUTPUT_DIR}")
