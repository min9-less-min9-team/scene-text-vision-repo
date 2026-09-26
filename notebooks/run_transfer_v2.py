#!/usr/bin/env python3
"""
전이학습 실험 v2 (clean_aug_v2 / 이미지 증강 3배) — 백그라운드 실행용 스크립트
Qwen3-VL + LoRA (8B: bf16 / 32B: 4bit), 프롬프트 2회 반복, TTA 2회, variant별 validation 평가 + test submission

[v2 추가 variant]
  mtvqa_kreta_clean_aug_v2
      외부(MTVQA+KRETA) → clean_aug_v2
      clean_aug_v2 = clean_aug와 동일하되, dev 3:2 중 편집(edit_option / mark_hard)되지 않은 문항은 제거
  mtvqa_kreta_clean_aug_v2_aug3x
      위와 같은 데이터를 단계별로 3배: [원본 전체(셔플)] → [compression + affine 사본(섞어서 셔플)]
      한 단계 안에서 원본 블록을 먼저 끝까지 학습한 뒤 증강 블록을 학습 (SequentialSampler, LR 스케줄 1개)
      --aug-scope all(기본): 외부 단계와 target 단계 모두 3배 / target: target 단계만 3배

  # A100 80GB PCIe (기본 preset: 8b_a100 = 8B bf16, 배치 8 x 누적 2)
  python run_transfer_v2.py --prepare-only
  python run_transfer_v2.py --benchmark-steps 20
  nohup python -u run_transfer_v2.py > logs/transfer_8b_a100_v2.log 2>&1 &

실행 예 (프로젝트 루트 또는 notebook 폴더에서):
    # 데이터 구성만 점검 (GPU 학습 없음)
    python run_transfer.py --preset 32b --prepare-only

    # 속도 측정: 첫 variant 첫 단계 20 step만 학습 → 전체 소요 시간 추정 출력
    python run_transfer.py --preset 32b --benchmark-steps 20

    # 8B bf16 백그라운드 실행 (터미널을 닫아도 계속 실행)
    mkdir -p logs
    nohup python -u run_transfer.py --preset 8b > logs/transfer_8b_bf16.log 2>&1 &

    # 32B 4bit
    nohup python -u run_transfer.py --preset 32b > logs/transfer_32b.log 2>&1 &
    echo $! > logs/transfer_32b.pid
    tail -f logs/transfer_32b.log

    # 일부 variant만
    nohup python -u run_transfer.py --preset 32b --variants raw mtvqa_kreta_clean_aug > logs/t32.log 2>&1 &

중단 후 재실행:
    같은 명령을 다시 실행하면 이어서 진행합니다.
    - 끝난 variant: result_<variant>.json이 있으면 건너뜀
    - 끝난 단계: <variant>__<stage>/done.json이 있으면 저장된 LoRA를 불러오고 건너뜀
    - 진행 중이던 단계: 마지막 checkpoint(--save-steps 간격)부터 재개

정제 CSV 6개({train,dev}_{changes,delete_list,hard_list}.csv)는 train.csv, dev.csv와 같은 폴더에 둡니다.
"""
import argparse
import os
import sys
from pathlib import Path

# =====================================================================
# 프리셋 (모델 크기별 기본값) — CLI 인자로 개별 덮어쓰기 가능
# =====================================================================
PRESETS = {
    # 8B: 16bit(bf16) LoRA. 가중치 약 17GB → 배치 4 x 누적 4 (실효 16)
    "8b": dict(model="unsloth/Qwen3-VL-8B-Instruct", load_in_4bit=False,   # BF16 가중치 (Unsloth chat template 수정 포함)
               batch_size=4, grad_accum=4, infer_batch_size=4, tag="8b"),
    # 8B on A100 80GB PCIe: 같은 모델·bf16·실효 배치 16, 메모리 여유만큼 배치를 키우고 누적을 줄임
    #   (tag="8b" → W&B group/run 이름은 기존 8b와 동일하게 유지, 출력 폴더는 preset명으로 분리)
    "8b_a100": dict(model="unsloth/Qwen3-VL-8B-Instruct", load_in_4bit=False,
                    batch_size=8, grad_accum=2, infer_batch_size=16, tag="8b"),
    # 32B: 4bit(Unsloth Dynamic) QLoRA
    "32b": dict(model="unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit", load_in_4bit=True,
                batch_size=2, grad_accum=8, infer_batch_size=2, tag="32b"),
}
ALL_VARIANTS = ["raw", "clean_aug", "kreta_raw", "mtvqa_raw", "mtvqa_kreta_raw", "mtvqa_kreta_clean_aug",
                "mtvqa_kreta_clean_aug_v2", "mtvqa_kreta_clean_aug_v2_aug3x"]
DEFAULT_VARIANTS = ["mtvqa_kreta_clean_aug_v2", "mtvqa_kreta_clean_aug_v2_aug3x"]


def parse_args():
    ap = argparse.ArgumentParser(description="Qwen3-VL 전이학습 실험 (백그라운드 실행용)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="8b_a100")
    ap.add_argument("--model", default=None, help="프리셋 모델명 덮어쓰기")
    ap.add_argument("--precision", choices=["preset", "4bit", "bf16"], default="preset",
                    help="가중치 로드 정밀도 (preset: 8b=bf16, 32b=4bit)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--infer-batch-size", type=int, default=None)
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=ALL_VARIANTS)
    ap.add_argument("--transfer-mode", choices=["sequential", "mix"], default="sequential")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--ext-epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ext-lr", type=float, default=None, help="기본: --lr과 동일")
    ap.add_argument("--max-pixel-tokens", type=int, default=1280, help="이미지 최대 비전 토큰 수")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--prompt-repeat", type=int, default=2)
    ap.add_argument("--n-tta", type=int, default=2)
    ap.add_argument("--dev-keep-patterns", nargs="+", default=["3", "3:1:1", "3:1", "3:2"])
    ap.add_argument("--mtvqa-splits", nargs="+", default=["train", "test"])
    # ---- v2 ----
    ap.add_argument("--dev-v2-drop-patterns", nargs="+", default=["3:2"],
                    help="clean_aug_v2: 이 투표 패턴 중 편집(edit_option/mark_hard)되지 않은 dev 문항 제거")
    ap.add_argument("--aug-scope", choices=["all", "target"], default="all",
                    help="aug3x variant: all=외부+target 단계 모두 3배, target=target 단계만 3배")
    ap.add_argument("--jpeg-quality", type=int, default=70, help="compression 증강 JPEG 품질")
    ap.add_argument("--affine-translate", type=float, default=0.03, help="affine 이동 비율 ±")
    ap.add_argument("--affine-shear-deg", type=float, default=5.0, help="affine shear 각도 ±")
    ap.add_argument("--num-workers", type=int, default=8, help="DataLoader worker 수 (결과에 영향 없음)")
    ap.add_argument("--save-steps", type=int, default=100, help="checkpoint 간격 (0이면 저장 안 함 → 재개 불가)")
    ap.add_argument("--no-test", action="store_true", help="test submission 생성 생략")
    ap.add_argument("--no-skip-done", action="store_true", help="완료된 variant/단계도 다시 실행")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--prepare-only", action="store_true", help="데이터 구성·요약만 하고 종료")
    ap.add_argument("--benchmark-steps", type=int, default=0, help=">0이면 N step 속도 측정 후 전체 시간 추정하고 종료")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


ARGS = parse_args()
_P = PRESETS[ARGS.preset]

# =====================================================================
# 설정 (노트북 1절과 동일한 이름)
# =====================================================================
MODEL_NAME = ARGS.model or _P["model"]
LOAD_IN_4BIT = _P["load_in_4bit"] if ARGS.precision == "preset" else (ARGS.precision == "4bit")
PRECISION_TAG = "4bit" if LOAD_IN_4BIT else "bf16"
GRADIENT_CHECKPOINTING = "unsloth"
SEED = ARGS.seed

PATCH_PIXELS = 32 * 32
MIN_PIXELS = 256 * PATCH_PIXELS
MAX_PIXELS = ARGS.max_pixel_tokens * PATCH_PIXELS
MAX_SEQ_LEN = ARGS.max_seq_len

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
FINETUNE_VISION = False
FINETUNE_LANGUAGE = True
FINETUNE_ATTENTION = True
FINETUNE_MLP = True

EPOCHS = ARGS.epochs
LR = ARGS.lr
BATCH_SIZE = ARGS.batch_size or _P["batch_size"]
GRAD_ACCUM = ARGS.grad_accum or _P["grad_accum"]
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
OPTIM = "adamw_8bit"
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 10
SHUFFLE_OPTIONS = True
SAVE_STEPS = ARGS.save_steps

EXPERIMENT_VARIANTS = list(dict.fromkeys(ARGS.variants))
VARIANT_SPECS = {
    "raw":                   {"external": [],                 "target": "raw"},
    "clean_aug":             {"external": [],                 "target": "clean_aug"},
    "kreta_raw":             {"external": ["kreta"],          "target": "raw"},
    "mtvqa_raw":             {"external": ["mtvqa"],          "target": "raw"},
    "mtvqa_kreta_raw":       {"external": ["mtvqa", "kreta"], "target": "raw"},
    "mtvqa_kreta_clean_aug": {"external": ["mtvqa", "kreta"], "target": "clean_aug"},
    "mtvqa_kreta_clean_aug_v2":       {"external": ["mtvqa", "kreta"], "target": "clean_aug_v2"},
    "mtvqa_kreta_clean_aug_v2_aug3x": {"external": ["mtvqa", "kreta"], "target": "clean_aug_v2",
                                       "img_aug": ["compression", "affine"]},
}
TRANSFER_MODE = ARGS.transfer_mode
EXT_EPOCHS = ARGS.ext_epochs
EXT_LR = ARGS.ext_lr if ARGS.ext_lr is not None else LR

DEV_KEEP_PATTERNS = ARGS.dev_keep_patterns
DEV_V2_DROP_PATTERNS = ARGS.dev_v2_drop_patterns
AUG_SCOPE = ARGS.aug_scope
JPEG_QUALITY = ARGS.jpeg_quality
AFFINE_TRANSLATE = ARGS.affine_translate
AFFINE_SHEAR_DEG = ARGS.affine_shear_deg
FILL_COLOR = (255, 255, 255)
NUM_WORKERS = ARGS.num_workers
HARD_REVERSE_AUG = True
DEV_VOTE_COLS = None

KRETA_REPO = "tabtoyou/KRETA"
KRETA_SPLIT = "test"
KRETA_SYSTEMS = None
MTVQA_REPO = "ByteDance/MTVQA"
MTVQA_SPLITS = ARGS.mtvqa_splits
MTVQA_LANGS = ["KO", "KR"]
MTVQA_MAX_ANSWER_CHARS = 40
MTVQA_DISTRACTOR_POOL = 50

SYSTEM_PROMPT = (
    "당신은 이미지를 보고 질문에 답하는 시각 질의응답 도우미입니다. "
    "a, b, c, d 중 정확히 한 글자로만 답하고, 설명은 하지 마세요."
)
USER_INSTRUCTION = "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
PROMPT_REPEAT = ARGS.prompt_repeat
PROMPT_REPEAT_SEP = "\n\n"

INFER_BATCH_SIZE = ARGS.infer_batch_size or _P["infer_batch_size"]
N_TTA = ARGS.n_tta
TTA_SHIFTS = [0, 2, 1, 3]
SAVE_TEST_SUBMISSION = not ARGS.no_test
SKIP_DONE = not ARGS.no_skip_done

if BATCH_SIZE * GRAD_ACCUM != 16:
    print(f"[warn] 실효 배치 {BATCH_SIZE * GRAD_ACCUM} ≠ 16 → 기존 tl8bbf16 run과 직접 비교 불가")
USE_WANDB = not ARGS.no_wandb and not ARGS.prepare_only and ARGS.benchmark_steps == 0
WANDB_ENTITY = "min9lessmin9team"
WANDB_PROJECT = "stv"
WANDB_GROUP = ARGS.wandb_group or f"transfer-{_P['tag']}-{PRECISION_TAG}-{TRANSFER_MODE}-rep{PROMPT_REPEAT}-tta{N_TTA}"
WANDB_RUN_PREFIX = f"tl{_P['tag']}{PRECISION_TAG}"

# 로그 파일로 리다이렉트해도 즉시 기록되도록
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
INTERACTIVE = sys.stderr.isatty()


# =====================================================================
# 2. 경로 / 환경 변수
# =====================================================================
import zipfile

ON_RUNPOD = bool(os.environ.get("RUNPOD_POD_ID")) or Path("/workspace").is_dir()

_HERE = Path(__file__).resolve().parent
REPO_ROOT = next(
    (d for d in [Path.cwd(), *Path.cwd().parents, _HERE, *_HERE.parents] if (d / "eda" / "split_validation.py").exists()),
    Path.cwd(),
)

ZIP_PATH = Path(os.environ.get("STV_DATA_ZIP", "/workspace/stv/data.zip" if ON_RUNPOD else str(REPO_ROOT / "data.zip")))
DATA_ROOT = Path(os.environ.get("STV_DATA_ROOT", "/root/data" if ON_RUNPOD else str(REPO_ROOT / "data")))
RAW_BASE_DIR = DATA_ROOT / "raw"
SPLIT_DIR = DATA_ROOT / "split"

CLEAN_DIR_ENV = os.environ.get("STV_CLEAN_DIR")
DEV_CSV_ENV = os.environ.get("STV_DEV_CSV")
DEV_AUDIT_ENV = os.environ.get("STV_DEV_AUDIT_CSV")
DEV_DIR_ENV = os.environ.get("STV_DEV_DIR")

EXT_DIR = Path(os.environ.get("STV_EXT_DIR", "/workspace/stv/external" if ON_RUNPOD else str(REPO_ROOT / "data" / "external")))
EXT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path(ARGS.output_dir or os.environ.get(
    "STV_OUTPUT_DIR",
    f"/workspace/stv/outputs/transfer_{ARGS.preset}_{PRECISION_TAG}_v2" if ON_RUNPOD
    else str(REPO_ROOT / "output" / f"transfer_{ARGS.preset}_{PRECISION_TAG}_v2"),
))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface" if ON_RUNPOD else str(REPO_ROOT / ".hf-cache"))
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHOICES = ["a", "b", "c", "d"]

print("repo  :", REPO_ROOT)
print("data  :", DATA_ROOT)
print("ext   :", EXT_DIR)
print("output:", OUTPUT_DIR)
print("model :", MODEL_NAME, f"| {PRECISION_TAG} | batch {BATCH_SIZE}x{GRAD_ACCUM} | infer batch {INFER_BATCH_SIZE}")
print("variants:", EXPERIMENT_VARIANTS)

# ---- 설정 가드: 같은 OUTPUT_DIR에 다른 설정의 결과가 섞이지 않게 함 ----
#   (SKIP_DONE이 이전 설정으로 만든 결과·LoRA를 재사용하는 것을 방지)
import json as _json
RUN_CONFIG = {
    "model": MODEL_NAME, "load_in_4bit": LOAD_IN_4BIT, "max_pixels": MAX_PIXELS, "max_seq_len": MAX_SEQ_LEN,
    "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "finetune_vision": FINETUNE_VISION,
    "epochs": EPOCHS, "ext_epochs": EXT_EPOCHS, "lr": LR, "ext_lr": EXT_LR,
    "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM, "transfer_mode": TRANSFER_MODE,
    "prompt_repeat": PROMPT_REPEAT, "n_tta": N_TTA, "dev_keep_patterns": DEV_KEEP_PATTERNS,
    "mtvqa_splits": MTVQA_SPLITS, "seed": SEED,
    "dev_v2_drop_patterns": DEV_V2_DROP_PATTERNS, "aug_scope": AUG_SCOPE, "jpeg_quality": JPEG_QUALITY,
    "affine_translate": AFFINE_TRANSLATE, "affine_shear_deg": AFFINE_SHEAR_DEG,
}
_cfg_path = OUTPUT_DIR / "run_config.json"
if not ARGS.prepare_only and ARGS.benchmark_steps == 0:
    if _cfg_path.exists():
        _old = _json.loads(_cfg_path.read_text(encoding="utf-8"))
        _diff = {k: (_old.get(k), v) for k, v in RUN_CONFIG.items() if _old.get(k) != v}
        if _diff:
            print(f"[ERROR] {OUTPUT_DIR}에 다른 설정으로 만든 결과가 있습니다 (기존 → 현재):")
            for k, (a, b) in _diff.items():
                print(f"  {k}: {a} → {b}")
            print("기존 결과를 지우거나(rm -rf) --output-dir로 다른 폴더를 지정하세요.")
            sys.exit(2)
    else:
        _cfg_path.write_text(_json.dumps(RUN_CONFIG, ensure_ascii=False, indent=1), encoding="utf-8")


# =====================================================================
# 3. import & 시드
# =====================================================================
import unsloth
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import gc, time, json, ast, random, re, io, base64, math, shutil, traceback, zlib
from collections import Counter
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, SequentialSampler
from tqdm.auto import tqdm as _tqdm
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import set_seed
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from peft.utils import load_peft_weights
from transformers.trainer_utils import get_last_checkpoint

if USE_WANDB:
    import wandb


def tqdm(*a, **k):
    """로그 파일(비대화형)에서는 진행 표시를 30초 간격으로만 갱신."""
    k.setdefault("mininterval", 0.1 if INTERACTIVE else 30)
    return _tqdm(*a, **k)


assert torch.cuda.is_available(), "CUDA GPU가 필요합니다."


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
print("GPU:", torch.cuda.get_device_name())

# =====================================================================
# 4. 데이터 준비
# =====================================================================
RAW_BASE_DIR.mkdir(parents=True, exist_ok=True)
if not list(RAW_BASE_DIR.rglob("train.csv")):
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"dataset zip not found: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(RAW_BASE_DIR)

RAW_DATA_DIR = sorted(RAW_BASE_DIR.rglob("train.csv"))[0].parent

sys.path.insert(0, str(REPO_ROOT / "eda"))
from split_validation import ensure_split
ensure_split(RAW_DATA_DIR, SPLIT_DIR)

train_df = pd.read_csv(SPLIT_DIR / "train.csv")
valid_df = pd.read_csv(SPLIT_DIR / "validation.csv")
test_df = pd.read_csv(RAW_DATA_DIR / "test.csv")
sample_sub = pd.read_csv(RAW_DATA_DIR / "sample_submission.csv")

REQUIRED_COLS = {"id", "path", "question", "a", "b", "c", "d"}
KEEP_COLS = ["id", "path", "question", "a", "b", "c", "d", "answer", "_img", "_src", "_aug"]


def normalize_df(df, name, has_answer=True):
    need = REQUIRED_COLS | ({"answer"} if has_answer else set())
    missing = need - set(df.columns)
    assert not missing, f"[{name}] missing columns: {sorted(missing)}"
    df = df.copy()
    df["id"] = df["id"].astype(str)
    if has_answer and "answer" in df.columns:
        df["answer"] = df["answer"].astype(str).str.strip().str.lower()
    return df


def attach_img_path(df, roots, name):
    roots = [Path(r) for r in roots if r is not None]

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
        f"[{name}] 이미지 {int(miss.sum())}개를 찾지 못함. roots={[str(r) for r in roots]}, "
        f"예: {df.loc[miss, 'path'].head(3).tolist()}"
    )
    return df


def norm_key(v):
    """'dev/dev_0001.jpg', 'dev_0001.jpg', 'dev_0001' → 'dev_0001'"""
    return Path(str(v)).stem.lower()


train_df = attach_img_path(normalize_df(train_df, "train"), [RAW_DATA_DIR], "train")
valid_df = attach_img_path(normalize_df(valid_df, "valid"), [RAW_DATA_DIR], "valid")
test_df = attach_img_path(normalize_df(test_df, "test", has_answer=False), [RAW_DATA_DIR], "test")

for name, df in [("train", train_df), ("valid", valid_df)]:
    assert df["answer"].isin(CHOICES).all(), f"[{name}] answer는 a/b/c/d만 허용"
assert set(train_df["id"]).isdisjoint(set(valid_df["id"]))

VALID_IDS = set(valid_df["id"])
VALID_KEYS = set(valid_df["id"].map(norm_key))
VALID_IMGS = set(valid_df["_img"])


def drop_valid_leakage(df, name):
    """validation과 id(source_id 우선) 또는 이미지 파일이 겹치는 행 제거."""
    src = df["source_id"].astype(str) if "source_id" in df.columns else df["id"].astype(str)
    by_id = src.map(norm_key).isin(VALID_KEYS).to_numpy()
    by_img = df["_img"].isin(VALID_IMGS).to_numpy()
    drop = by_id | by_img
    if drop.any():
        print(f"[{name}] validation leakage 제거: id {int(by_id.sum())} | image {int(by_img.sum())}")
    return df.loc[~drop].reset_index(drop=True)


print(f"train={len(train_df)} | valid={len(valid_df)} | test={len(test_df)}")

# =====================================================================
# 5-1. clean_aug
# =====================================================================
EDIT_ACTIONS = {"delete", "mark_hard", "edit_option"}
REV_MAP = {"a": "d", "b": "c", "c": "b", "d": "a"}


CLEAN_FILES = [f"{s}_{n}.csv" for s in ("train", "dev") for n in ("changes", "delete_list", "hard_list")]


def find_clean_dir():
    """정제 CSV 6개는 train.csv / dev.csv와 같은 폴더(RAW_DATA_DIR)에 둡니다. STV_CLEAN_DIR로 덮어쓰기 가능."""
    d = Path(CLEAN_DIR_ENV) if CLEAN_DIR_ENV else RAW_DATA_DIR
    missing = [f for f in CLEAN_FILES if not (d / f).exists()]
    assert not missing, f"정제 CSV가 {d}에 없습니다: {missing} (train.csv, dev.csv와 같은 폴더에 두세요)"
    return d


def same_value(a, b):
    a, b = str(a).strip(), str(b).strip()
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def load_change_lists(split):
    read = lambda n: pd.read_csv(CLEAN_DIR / f"{split}_{n}.csv", dtype=str, keep_default_na=False)
    changes, dele, hard = read("changes"), read("delete_list"), read("hard_list")

    unknown = set(changes["action"]) - EDIT_ACTIONS
    assert not unknown, f"[{split}] 알 수 없는 action: {unknown}"

    ch_del = set(changes.loc[changes["action"] == "delete", "id"].map(norm_key))
    ch_hard = set(changes.loc[changes["action"] == "mark_hard", "id"].map(norm_key))
    del_ids, hard_ids = set(dele["id"].map(norm_key)), set(hard["id"].map(norm_key))
    if ch_del != del_ids or ch_hard != hard_ids:
        print(f"⚠ [{split}] changes.csv와 *_list.csv 목록이 다름 → 합집합 사용")
    del_ids |= ch_del
    hard_ids |= ch_hard
    assert del_ids.isdisjoint(hard_ids), f"[{split}] 삭제와 hard에 동시에 있는 id가 있습니다."

    edits = changes.loc[changes["action"] == "edit_option", ["id", "field", "before", "after"]].copy()
    edits["key"] = edits["id"].map(norm_key)
    edits["field"] = edits["field"].str.strip().str.lower()
    assert edits["field"].isin(CHOICES).all(), f"[{split}] edit_option field는 a~d만 허용"

    ref = pd.concat([dele, hard])[["id", "answer"]]
    ref_answer = dict(zip(ref["id"].map(norm_key), ref["answer"].str.strip().str.lower()))
    print(f"[{split}] 삭제 {len(del_ids)} | hard {len(hard_ids)} | 선지 수정 {len(edits)} (문항 {edits['key'].nunique()})")
    return {"delete": del_ids, "hard": hard_ids, "edits": edits, "ref_answer": ref_answer}


def apply_edits(df, edits, name):
    df = df.copy()
    keys = df["id"].map(norm_key)
    n_ok, n_missing = 0, 0
    for e in edits.to_dict("records"):
        m = (keys == e["key"]).to_numpy()
        if not m.any():
            n_missing += 1
            continue
        cur = df.loc[m, e["field"]].iloc[0]
        assert same_value(cur, e["before"]), (
            f"[{name}] {e['id']} 보기 {e['field']}: 현재값 {cur!r} != before {e['before']!r}"
        )
        df.loc[m, e["field"]] = e["after"]
        n_ok += 1
    print(f"[{name}] 선지 수정 적용 {n_ok}건 | 대상 행 없음 {n_missing}건")
    return df


def reverse_options(df):
    """보기 abcd → dcba 재배치 + 정답 재매핑."""
    rev = df.copy()
    rev[["a", "b", "c", "d"]] = df[["d", "c", "b", "a"]].to_numpy()
    rev["answer"] = df["answer"].map(REV_MAP)
    rev["source_id"] = df["id"].astype(str)
    rev["id"] = df["id"].astype(str) + "__rev"
    rev["_src"] = df["_src"].astype(str) + "_hardrev"
    return rev


def check_ref_answer(df, ref_answer, name):
    keys = df["id"].map(norm_key)
    has = keys.isin(ref_answer.keys())
    if has.any():
        diff = (df.loc[has, "answer"].values != keys[has].map(ref_answer).values).sum()
        print(f"[{name}] 정제 목록 answer와 불일치: {int(diff)}/{int(has.sum())}")


# ---------------------------------------------------------------------
# train 부분
# ---------------------------------------------------------------------
def build_clean_train():
    cl = load_change_lists("train")
    listed = cl["delete"] | cl["hard"] | set(cl["edits"]["key"])
    print(f"[train] 목록 id 중 validation 소속(미적용): {len(listed & VALID_KEYS)}")

    df = train_df.assign(_src="train")
    check_ref_answer(df, cl["ref_answer"], "train")
    df = apply_edits(df, cl["edits"], "train")

    keys = df["id"].map(norm_key)
    drop = keys.isin(cl["delete"])
    df = df.loc[~drop].reset_index(drop=True)

    parts = [df]
    if HARD_REVERSE_AUG:
        hard = df.loc[df["id"].map(norm_key).isin(cl["hard"])]
        parts.append(reverse_options(hard))
    out = pd.concat(parts, ignore_index=True)
    print(f"[train] 삭제 {int(drop.sum())} | hard 복사 {len(out) - len(df)} | 결과 {len(out)}")
    return out


# ---------------------------------------------------------------------
# dev 부분
# ---------------------------------------------------------------------
VOTE_COL_TEMPLATES = ["vote_{c}", "votes_{c}", "{c}_vote", "{c}_votes", "count_{c}", "{c}_count", "n_{c}", "{c}_cnt", "cnt_{c}"]
VOTE_STR_COLS = ["human_votes", "votes", "vote", "vote_dist", "vote_counts", "distribution", "dist"]


def find_file(env_value, names, label):
    if env_value:
        p = Path(env_value)
        assert p.exists(), f"{label} 경로 없음: {p}"
        return p
    for base in (DATA_ROOT, REPO_ROOT):
        for name in names:
            hits = sorted(base.rglob(name))
            if hits:
                return hits[0]
    return None


ANNOTATOR_COL_RE = re.compile(r"^answer\d+$")   # dev.csv의 answer1 ~ answer5 (작업자별 선택)


def detect_vote_spec(df):
    annot = sorted((c for c in df.columns if ANNOTATOR_COL_RE.match(str(c))), key=lambda c: int(c[6:]))
    if annot:
        return "annot", annot
    if DEV_VOTE_COLS:
        missing = set(DEV_VOTE_COLS.values()) - set(df.columns)
        assert not missing, f"DEV_VOTE_COLS 컬럼 없음: {sorted(missing)}"
        return "cols", DEV_VOTE_COLS
    for t in VOTE_COL_TEMPLATES:
        cols = {c: t.format(c=c) for c in CHOICES}
        if all(v in df.columns for v in cols.values()):
            return "cols", cols
    for col in VOTE_STR_COLS:
        if col in df.columns:
            return "str", col
    return None


def parse_votes(row, mode, spec):
    """보기 a~d 순서의 득표수 [int×4]."""
    if mode == "annot":
        letters = []
        for c in spec:
            v = row[c]
            if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
                continue                                   # 응답 없는 작업자는 제외
            t = str(v).strip().lower()
            t = {"1": "a", "2": "b", "3": "c", "4": "d", "1.0": "a", "2.0": "b", "3.0": "c", "4.0": "d"}.get(t, t)
            assert t in CHOICES, f"{row['id']} {c}: 해석할 수 없는 응답 {v!r}"
            letters.append(t)
        return [letters.count(c) for c in CHOICES]
    if mode == "cols":
        return [int(float(row[spec[c]])) if pd.notna(row[spec[c]]) and str(row[spec[c]]).strip() else 0 for c in CHOICES]
    v = row[spec]
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return [0, 0, 0, 0]
    if isinstance(v, str):
        s = v.strip()
        if "|" in s:                                   # "A|B|B|A|B"
            letters = [t.strip().lower() for t in s.split("|") if t.strip()]
            return [letters.count(c) for c in CHOICES]
        try:
            v = json.loads(s)
        except json.JSONDecodeError:
            v = ast.literal_eval(s)
    if isinstance(v, dict):
        v = {str(k).strip().lower(): int(n) for k, n in v.items()}
        return [v.get(c, 0) for c in CHOICES]
    if isinstance(v, (list, tuple)) and len(v) == 4:
        return [int(n) for n in v]
    raise ValueError(f"득표 형식 해석 불가: {row[spec]!r}")


def vote_pattern(votes):
    return ":".join(str(n) for n in sorted((n for n in votes if n > 0), reverse=True))


def load_dev_votes():
    dev_csv = Path(DEV_CSV_ENV) if DEV_CSV_ENV else RAW_DATA_DIR / "dev.csv"
    if not dev_csv.exists():
        dev_csv = find_file(None, ["dev.csv"], "dev CSV")
    assert dev_csv is not None, "dev.csv를 찾지 못함. STV_DEV_CSV로 지정하세요."
    dev_dir = Path(DEV_DIR_ENV) if DEV_DIR_ENV else dev_csv.parent
    dev = normalize_df(pd.read_csv(dev_csv), "dev", has_answer=False)
    dev["_key"] = dev["id"].map(norm_key)

    spec = detect_vote_spec(dev)
    if spec is None:
        audit_csv = Path(DEV_AUDIT_ENV) if DEV_AUDIT_ENV else RAW_DATA_DIR / "dev_audit.csv"
        if not audit_csv.exists():
            audit_csv = find_file(None, ["dev_audit.csv"], "dev audit CSV")
        assert audit_csv is not None, (
            f"dev.csv에 투표 컬럼이 없고 dev_audit.csv도 없습니다. STV_DEV_AUDIT_CSV 또는 DEV_VOTE_COLS를 지정하세요.\n"
            f"dev columns: {list(dev.columns)}"
        )
        audit = pd.read_csv(audit_csv)
        audit["_key"] = audit["id"].map(norm_key)
        extra = [c for c in audit.columns if c not in dev.columns]
        dev = dev.merge(audit[["_key", *extra]], on="_key", how="left", validate="one_to_one")
        spec = detect_vote_spec(dev)
        assert spec is not None, f"audit 병합 후에도 투표 컬럼을 찾지 못함: {list(dev.columns)}"
        print(f"[dev] audit 병합: {audit_csv}")

    mode, col = spec
    votes = np.array([parse_votes(r, mode, col) for _, r in dev.iterrows()])
    dev["_vote_pattern"] = [vote_pattern(v) for v in votes]
    dev["_human_answer"] = [CHOICES[int(np.argmax(v))] for v in votes]
    dev["_top_unique"] = [(v == v.max()).sum() == 1 and v.max() > 0 for v in votes]
    print(f"[dev] {dev_csv} | rows={len(dev)} | vote spec={mode}:{col}")
    return dev, dev_dir


def build_clean_dev(drop_unedited_patterns=()):
    dev, dev_dir = load_dev_votes()
    print("[dev] 투표 패턴 분포:\n" + dev["_vote_pattern"].value_counts().to_string())

    sel = dev.loc[dev["_vote_pattern"].isin(DEV_KEEP_PATTERNS)].copy()
    assert sel["_top_unique"].all(), "선택 패턴에 최다 득표 동률이 있습니다. DEV_KEEP_PATTERNS 확인"
    sel["answer"] = sel["_human_answer"]
    sel["_src"] = "dev_" + sel["_vote_pattern"]
    print(f"[dev] 패턴 {DEV_KEEP_PATTERNS} 선택: {len(sel)}/{len(dev)}")

    cl = load_change_lists("dev")
    listed = cl["delete"] | cl["hard"] | set(cl["edits"]["key"])
    print(f"[dev] 목록 id 중 패턴 필터로 이미 제외된 수: {len(listed - set(sel['_key']))}")
    check_ref_answer(sel, cl["ref_answer"], "dev")

    if drop_unedited_patterns:
        miss = set(drop_unedited_patterns) - set(DEV_KEEP_PATTERNS)
        assert not miss, f"--dev-v2-drop-patterns {sorted(miss)}가 DEV_KEEP_PATTERNS에 없습니다"
        edited = cl["hard"] | set(cl["edits"]["key"])
        in_pat = sel["_vote_pattern"].isin(drop_unedited_patterns)
        v2_drop = in_pat & ~sel["_key"].isin(edited)
        print(f"[dev v2] 패턴 {list(drop_unedited_patterns)}: {int(in_pat.sum())}개 중 "
              f"편집 유지 {int((in_pat & ~v2_drop).sum())} | 미편집 제거 {int(v2_drop.sum())}")
        sel = sel.loc[~v2_drop].reset_index(drop=True)

    sel = apply_edits(sel, cl["edits"], "dev")
    drop = sel["_key"].isin(cl["delete"])
    sel = sel.loc[~drop].reset_index(drop=True)

    sel = attach_img_path(sel, [dev_dir, RAW_DATA_DIR, DATA_ROOT], "dev")
    sel = drop_valid_leakage(sel, "dev")

    parts = [sel]
    if HARD_REVERSE_AUG:
        parts.append(reverse_options(sel.loc[sel["_key"].isin(cl["hard"])]))
    out = pd.concat(parts, ignore_index=True)
    print(f"[dev] 삭제 {int(drop.sum())} | hard 복사 {len(out) - len(sel)} | 결과 {len(out)}")
    return out


def build_target(name):
    if name == "raw":
        return train_df.assign(_src="train")
    if name in ("clean_aug", "clean_aug_v2"):
        global CLEAN_DIR
        CLEAN_DIR = find_clean_dir()
        print("clean dir:", CLEAN_DIR)
        drop = DEV_V2_DROP_PATTERNS if name == "clean_aug_v2" else ()
        return pd.concat([build_clean_train(), build_clean_dev(drop)], ignore_index=True)
    raise ValueError(f"unknown target: {name}")

# =====================================================================
# 5-2. 외부 데이터 (KRETA / MTVQA)
# =====================================================================
def safe_name(s):
    return re.sub(r"[^0-9A-Za-z_\-]", "_", str(s))


def finalize_ext(df, name):
    df = normalize_df(df, name)
    df = attach_img_path(df, [EXT_DIR], name)
    df["_src"] = name
    for c in CHOICES:
        df[c] = df[c].astype(str)
    assert df["answer"].isin(CHOICES).all()
    return drop_valid_leakage(df, name)


# ---------------------------------------------------------------------
# KRETA
# ---------------------------------------------------------------------
def load_kreta():
    cache = EXT_DIR / "kreta" / "kreta.csv"
    img_dir = EXT_DIR / "kreta" / "images"
    if not cache.exists():
        img_dir.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(KRETA_REPO, split=KRETA_SPLIT)
        rows, skipped, seen = [], Counter(), set()
        for r in tqdm(ds, desc="KRETA"):
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
                Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").save(p, quality=95)
            rows.append({
                "id": f"kreta_{rid}", "path": str(p.relative_to(EXT_DIR)),
                "question": str(r["question"]).strip(),
                **{c: str(o).strip() for c, o in zip(CHOICES, opts)},
                "answer": ans.lower(),
                "topic_difficulty": r.get("topic_difficulty"),
                "image_type": r.get("image_type"),
            })
        pd.DataFrame(rows).to_csv(cache, index=False, encoding="utf-8-sig")
        print(f"[kreta] 저장 {len(rows)} | 제외 {dict(skipped)}")

    df = pd.read_csv(cache, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if KRETA_SYSTEMS:
        df = df[df["topic_difficulty"].isin(KRETA_SYSTEMS)]
    return finalize_ext(df, "kreta")


# ---------------------------------------------------------------------
# MTVQA (KO) → 4지선다
# ---------------------------------------------------------------------
def mt_norm(s):
    return re.sub(r"[\s\W_]+", "", str(s)).lower()


def has_digit(s):
    return any(ch.isdigit() for ch in str(s))


def load_mtvqa_qa():
    cache = EXT_DIR / "mtvqa" / "mtvqa_ko_qa.csv"
    img_dir = EXT_DIR / "mtvqa" / "images"
    if not cache.exists():
        img_dir.mkdir(parents=True, exist_ok=True)
        langs = {l.upper() for l in MTVQA_LANGS}
        recs = []
        for split in MTVQA_SPLITS:
            ds = load_dataset(MTVQA_REPO, split=split)
            idx = [i for i, l in enumerate(ds["lang"]) if str(l).upper() in langs]
            print(f"[mtvqa] {split}: {len(idx)} images (lang in {sorted(langs)})")
            for r in tqdm(ds.select(idx), desc=f"MTVQA {split}"):
                img_id = f"{split}_{safe_name(r['id'])}"
                p = img_dir / f"{img_id}.jpg"
                if not p.exists():
                    r["image"].convert("RGB").save(p, quality=95)
                qa = r["qa_pairs"]
                if isinstance(qa, str):
                    qa = ast.literal_eval(qa)
                for k, item in enumerate(qa):
                    recs.append({
                        "img_id": img_id, "k": k, "path": str(p.relative_to(EXT_DIR)),
                        "question": " ".join(str(item.get("question", "")).split()),
                        "answer_text": " ".join(str(item.get("answer", "")).split()),
                    })
        pd.DataFrame(recs).to_csv(cache, index=False, encoding="utf-8-sig")
    return pd.read_csv(cache, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def build_mtvqa_mc(qa):
    n0 = len(qa)
    qa = qa[(qa["question"].str.len() > 0) & (qa["answer_text"].str.len() > 0)
            & (qa["answer_text"].str.len() <= MTVQA_MAX_ANSWER_CHARS)].copy()
    qa["norm"] = qa["answer_text"].map(mt_norm)
    qa = qa[qa["norm"].str.len() > 0].reset_index(drop=True)

    uniq = qa.drop_duplicates("norm")
    pool = {True: [], False: []}
    for a, n in zip(uniq["answer_text"], uniq["norm"]):
        pool[has_digit(a)].append((a, n, len(a)))
    by_img = qa.groupby("img_id")["answer_text"].apply(list).to_dict()

    def ok(n, used, gold_n):
        return n and n not in used and n not in gold_n and gold_n not in n

    rows, n_skip, n_same_img = [], 0, 0
    for i, r in enumerate(qa.to_dict("records")):
        rng = random.Random(SEED * 7919 + i)
        gold, gold_n = r["answer_text"], r["norm"]
        opts, used = [gold], {gold_n}

        same = list(by_img[r["img_id"]])
        rng.shuffle(same)
        for a in same:
            n = mt_norm(a)
            if ok(n, used, gold_n):
                opts.append(a); used.add(n); n_same_img += 1
            if len(opts) == 4:
                break

        if len(opts) < 4:
            cands = [c for c in pool[has_digit(gold)] if ok(c[1], used, gold_n)]
            cands.sort(key=lambda c: abs(c[2] - len(gold)))
            near = cands[:MTVQA_DISTRACTOR_POOL]
            rng.shuffle(near)
            for a, n, _ in near:
                if n not in used:
                    opts.append(a); used.add(n)
                if len(opts) == 4:
                    break

        if len(opts) < 4:
            n_skip += 1
            continue
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        shown = [opts[j] for j in order]
        rows.append({
            "id": f"mtvqa_{r['img_id']}_{r['k']}", "path": r["path"], "question": r["question"],
            **dict(zip(CHOICES, shown)), "answer": CHOICES[order.index(0)],
        })

    print(f"[mtvqa] QA {n0} → 필터 후 {len(qa)} → 4지선다 {len(rows)} (오답 부족 제외 {n_skip}) | "
          f"같은 이미지 오답 비율 {n_same_img / max(1, 3 * len(rows)):.1%}")
    return pd.DataFrame(rows)


def load_mtvqa():
    return finalize_ext(build_mtvqa_mc(load_mtvqa_qa()), "mtvqa")


EXT_LOADERS = {"kreta": load_kreta, "mtvqa": load_mtvqa}

# =====================================================================
# 5-3. variant별 학습 단계
# =====================================================================
def finalize(df, name):
    df = df.copy()
    if "_src" not in df.columns:
        df["_src"] = name
    if "_aug" not in df.columns:
        df["_aug"] = "none"
    df = df[KEEP_COLS].reset_index(drop=True)
    assert df["answer"].isin(CHOICES).all(), f"[{name}] answer 오류"
    for c in CHOICES:
        assert (df[c].astype(str).str.strip() != "").all(), f"[{name}] 빈 보기 {c}"
    assert not df["id"].duplicated().any(), f"[{name}] 중복 id"
    return df


need_targets = sorted({VARIANT_SPECS[v]["target"] for v in EXPERIMENT_VARIANTS})
need_ext = sorted({x for v in EXPERIMENT_VARIANTS for x in VARIANT_SPECS[v]["external"]})

TARGET_DFS = {}
for t in need_targets:
    print(f"\n===== target: {t} =====")
    TARGET_DFS[t] = finalize(build_target(t), t)

EXT_DFS = {}
for x in need_ext:
    print(f"\n===== external: {x} =====")
    EXT_DFS[x] = finalize(EXT_LOADERS[x](), x)


def shuffled(df):
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def is_ordered(variant):
    """이미지 증강 variant는 [원본 블록 → 증강 블록] 순서를 지켜야 하므로 순차 샘플링."""
    return bool(VARIANT_SPECS[variant].get("img_aug"))


def expand_aug(df, augs):
    """[원본 전체(셔플)] + [증강 사본들을 섞어서 셔플] → 행 수 (1 + len(augs))배."""
    if not augs:
        return shuffled(df)
    copies = [df.assign(id=df["id"].astype(str) + f"__{a}", _aug=a, _src=df["_src"].astype(str) + f"+{a}")
              for a in augs]
    return pd.concat([shuffled(df), shuffled(pd.concat(copies, ignore_index=True))], ignore_index=True)


def build_stages(variant):
    spec = VARIANT_SPECS[variant]
    augs = spec.get("img_aug", [])
    ext_augs = augs if AUG_SCOPE == "all" else []
    target = TARGET_DFS[spec["target"]]
    if not spec["external"]:
        return [("target", expand_aug(target, augs))]
    ext = pd.concat([EXT_DFS[x] for x in spec["external"]], ignore_index=True)
    if TRANSFER_MODE == "sequential":
        return [("external", expand_aug(ext, ext_augs)), ("target", expand_aug(target, augs))]
    if TRANSFER_MODE == "mix":
        return [("mix", expand_aug(pd.concat([ext, target], ignore_index=True), augs))]
    raise ValueError(f"unknown TRANSFER_MODE: {TRANSFER_MODE}")


STAGES = {v: build_stages(v) for v in EXPERIMENT_VARIANTS}

summary = pd.DataFrame([
    {
        "variant": v, "stage": s, "rows": len(df),
        **{f"ans_{c}": int((df["answer"] == c).sum()) for c in CHOICES},
        "aug": df["_aug"].value_counts().to_dict(),
        "sources": df["_src"].value_counts().to_dict(),
    }
    for v, stages in STAGES.items() for s, df in stages
])
print("\n[stage summary]\n" + summary.to_string(index=False))
summary.to_csv(OUTPUT_DIR / "stage_summary.csv", index=False, encoding="utf-8-sig")

if ARGS.prepare_only:
    print("\n--prepare-only: 데이터 구성만 확인하고 종료")
    sys.exit(0)

# =====================================================================
# 6. 모델 로드 & LoRA
# =====================================================================
model, processor = FastVisionModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit=LOAD_IN_4BIT,
    use_gradient_checkpointing=GRADIENT_CHECKPOINTING,
    max_seq_length=MAX_SEQ_LEN,
)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=FINETUNE_VISION,
    finetune_language_layers=FINETUNE_LANGUAGE,
    finetune_attention_modules=FINETUNE_ATTENTION,
    finetune_mlp_modules=FINETUNE_MLP,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    random_state=SEED,
)
model.print_trainable_parameters()

tokenizer = processor.tokenizer
processor.image_processor.size = {"shortest_edge": int(MIN_PIXELS), "longest_edge": int(MAX_PIXELS)}

CHOICE_IDS = [tokenizer.convert_tokens_to_ids(c) for c in CHOICES]
assert all(tokenizer.decode([t]) == c for t, c in zip(CHOICE_IDS, CHOICES))

init_state = {k: v.detach().to("cpu", copy=True) for k, v in get_peft_model_state_dict(model).items()}
print(f"init LoRA tensors: {len(init_state)} | GPU memory: {torch.cuda.memory_allocated() / 2**30:.2f} GB")

# =====================================================================
# 7. 프롬프트 / Dataset / Collator
# =====================================================================
def apply_image_aug(img, kind, key):
    """학습 사본 전용 증강 (항상 적용). id 기반 RNG → worker 수/재실행과 무관하게 동일 변환."""
    if kind in (None, "", "none"):
        return img
    rng = random.Random(zlib.crc32(str(key).encode("utf-8")) ^ SEED)
    if kind == "compression":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        out.load()
        return out
    if kind == "affine":
        w, h = img.size
        tx = rng.uniform(-AFFINE_TRANSLATE, AFFINE_TRANSLATE) * w
        ty = rng.uniform(-AFFINE_TRANSLATE, AFFINE_TRANSLATE) * h
        sx = math.tan(math.radians(rng.uniform(-AFFINE_SHEAR_DEG, AFFINE_SHEAR_DEG)))
        sy = math.tan(math.radians(rng.uniform(-AFFINE_SHEAR_DEG, AFFINE_SHEAR_DEG)))
        return img.transform(img.size, Image.Transform.AFFINE, (1.0, sx, tx, sy, 1.0, ty),
                             resample=Image.Resampling.BICUBIC, fillcolor=FILL_COLOR)
    raise ValueError(f"unknown image aug: {kind}")


def load_image(row):
    img = Image.open(row["_img"]).convert("RGB")
    return apply_image_aug(img, row.get("_aug", "none"), row["id"])   # valid/test에는 _aug 없음 → 원본


def build_messages(row, answer=None, options=None, image=None):
    if options is None:
        options = [row[c] for c in CHOICES]
    if image is None:
        image = load_image(row)
    body = (
        f"{row['question']}\n"
        f"(a) {options[0]}\n(b) {options[1]}\n(c) {options[2]}\n(d) {options[3]}\n\n"
        f"{USER_INSTRUCTION}"
    )
    prompt = PROMPT_REPEAT_SEP.join([body] * PROMPT_REPEAT)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


class VQADataset(Dataset):
    def __init__(self, df, shuffle_options=False, seed=SEED):
        self.df = df.reset_index(drop=True)
        self.shuffle_options = shuffle_options
        self.seed = seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        options = [row[c] for c in CHOICES]
        gold = CHOICES.index(str(row["answer"]).strip().lower())
        order = list(range(4))
        if self.shuffle_options:
            random.Random(self.seed * 1_000_003 + i).shuffle(order)
        shown = [options[j] for j in order]
        return {"messages": build_messages(row, answer=CHOICES[order.index(gold)], options=shown)}


base_collator = UnslothVisionDataCollator(
    model, processor, resize="max", max_seq_length=MAX_SEQ_LEN,
    train_on_responses_only=True,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)


def collator(features):
    batch = base_collator(features)
    labels = batch["labels"]
    keep = torch.zeros_like(labels, dtype=torch.bool)
    for token_id in CHOICE_IDS:
        keep |= labels.eq(token_id)
    batch["labels"] = torch.where(keep, labels, torch.full_like(labels, -100))
    return batch


def text_len(df):
    return df[["question", *CHOICES]].astype(str).apply(lambda r: sum(map(len, r)), axis=1)


all_stage_dfs = [df for stages in STAGES.values() for _, df in stages]
check_df = pd.concat([df.assign(_len=text_len(df)) for df in all_stage_dfs]).nlargest(4, "_len")
check = collator([VQADataset(check_df)[i] for i in range(len(check_df))])
seq_lens = check["attention_mask"].sum(1).tolist()
for ids, labels, L in zip(check["input_ids"], check["labels"], seq_lens):
    kept = ids[labels != -100]
    print(f"seq_len={L} | loss 대상 {tokenizer.decode(kept)!r}")
    assert len(kept) == 1 and tokenizer.decode(kept) in CHOICES, "정답 토큰이 잘렸거나 마스킹 오류 → MAX_SEQ_LEN 확인"
print(f"max seq_len: {max(seq_lens)} / MAX_SEQ_LEN={MAX_SEQ_LEN}")
del check, check_df

# =====================================================================
# 8. 추론 (TTA)
# =====================================================================
@torch.inference_mode()
def predict_probs(df, batch_size=INFER_BATCH_SIZE, n_tta=N_TTA, desc="predict"):
    assert 1 <= n_tta <= len(TTA_SHIFTS)
    shifts = TTA_SHIFTS[:n_tta]
    probs = np.zeros((len(df), 4), dtype=np.float32)

    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    FastVisionModel.for_inference(model)
    try:
        for start in tqdm(range(0, len(df), batch_size), desc=desc):
            rows = [r for _, r in df.iloc[start:start + batch_size].iterrows()]
            images = [load_image(r) for r in rows]
            acc = np.zeros((len(rows), 4), dtype=np.float32)
            for s in shifts:
                order = [(j + s) % 4 for j in range(4)]
                texts = [
                    processor.apply_chat_template(
                        build_messages(r, options=[r[CHOICES[o]] for o in order], image=img),
                        tokenize=False, add_generation_prompt=True,
                    )
                    for r, img in zip(rows, images)
                ]
                enc = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**enc, logits_to_keep=1).logits[:, -1, CHOICE_IDS]
                acc[:, order] += torch.softmax(logits.float(), dim=-1).cpu().numpy()
            probs[start:start + len(rows)] = acc / len(shifts)
    finally:
        tokenizer.padding_side = old_side
    return probs


def accuracy(df, probs):
    gold = df["answer"].map(CHOICES.index).to_numpy()
    return float((probs.argmax(axis=1) == gold).mean())

# =====================================================================
# 9. 단계별 학습 (checkpoint 재개 지원) / variant 실험
# =====================================================================
def sft_args(output_dir, epochs, lr, max_steps=-1, save=True, report=True):
    return SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        optim=OPTIM,
        max_grad_norm=MAX_GRAD_NORM,
        bf16=True,
        tf32=True,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps" if (save and SAVE_STEPS > 0) else "no",
        save_steps=max(SAVE_STEPS, 1),
        save_total_limit=1,
        report_to="wandb" if (report and USE_WANDB) else "none",
        disable_tqdm=not INTERACTIVE,   # 로그 파일에는 logging_steps마다 loss dict만 기록
        seed=SEED,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=MAX_SEQ_LEN,
        dataloader_num_workers=NUM_WORKERS,
    )


def stage_hparams(stage):
    return (EXT_EPOCHS, EXT_LR) if stage == "external" else (EPOCHS, LR)


class OrderedSFTTrainer(SFTTrainer):
    """데이터프레임 순서 그대로 학습 (원본 블록 → 증강 블록)."""
    def _get_train_sampler(self, *args, **kwargs):
        ds = args[0] if args else kwargs.get("train_dataset", None)
        return SequentialSampler(ds if ds is not None else self.train_dataset)


def make_trainer(df, args, ordered=False):
    cls = OrderedSFTTrainer if ordered else SFTTrainer
    trainer = cls(
        model=model,
        processing_class=tokenizer,
        data_collator=collator,
        train_dataset=VQADataset(df, shuffle_options=SHUFFLE_OPTIONS),
        args=args,
    )
    if ordered:
        # 순차 샘플러가 실제로 쓰이는지 확인 (Unsloth 패치로 무시되는 경우 방지)
        bs = getattr(trainer.get_train_dataloader(), "batch_sampler", None)
        if bs is None:
            print("    [warn] batch_sampler 확인 불가 — 순서 검증 생략")
        else:
            first = []
            for b in bs:
                first.extend(b)
                if len(first) >= 3 * BATCH_SIZE:
                    break
            assert first == list(range(len(first))), f"순차 샘플링이 아님: {first[:12]}"
            print(f"    ordered sampler OK: {first[:8]} ...")
    return trainer


def train_stage(variant, stage, df):
    sdir = OUTPUT_DIR / f"{variant}__{stage}"
    adapter_dir, marker, trainer_dir = sdir / "adapter", sdir / "done.json", sdir / "trainer"

    if SKIP_DONE and marker.exists():
        set_peft_model_state_dict(model, load_peft_weights(str(adapter_dir)))
        info = json.loads(marker.read_text(encoding="utf-8"))
        print(f"[skip] {variant}/{stage}: 완료된 단계 → 저장된 LoRA 로드 ({adapter_dir})")
        return info

    epochs, lr = stage_hparams(stage)
    print(f"\n--- [{variant}] stage={stage} | rows={len(df)} | epochs={epochs} | lr={lr} | "
          f"batch={BATCH_SIZE}x{GRAD_ACCUM}")
    print("    sources:", df["_src"].value_counts().to_dict())

    FastVisionModel.for_training(model)
    trainer = make_trainer(df, sft_args(trainer_dir, epochs, lr), ordered=is_ordered(variant))
    last_ckpt = get_last_checkpoint(str(trainer_dir)) if (SKIP_DONE and trainer_dir.is_dir()) else None
    if last_ckpt:
        print(f"    checkpoint에서 재개: {last_ckpt}")

    t0 = time.time()
    res = trainer.train(resume_from_checkpoint=last_ckpt)
    info = {"stage": stage, "rows": len(df), "loss": round(float(res.training_loss), 4),
            "minutes": round((time.time() - t0) / 60, 1)}

    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    marker.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(trainer_dir, ignore_errors=True)   # 단계 완료 후 중간 checkpoint 삭제
    if USE_WANDB:
        wandb.log({f"stage/{stage}/train_loss": info["loss"], f"stage/{stage}/minutes": info["minutes"]})

    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"    done: loss={info['loss']} | {info['minutes']} min")
    return info


def result_path(variant):
    return OUTPUT_DIR / f"result_{variant}.json"


def is_done(variant):
    need = [result_path(variant), OUTPUT_DIR / f"probs_valid_{variant}.npy"]
    if SAVE_TEST_SUBMISSION:
        need.append(OUTPUT_DIR / f"submission_{variant}.csv")
    return all(p.exists() for p in need)


def run_experiment(variant):
    if SKIP_DONE and is_done(variant):
        print(f"[skip] {variant}: 결과 파일이 이미 있음")
        return json.loads(result_path(variant).read_text(encoding="utf-8"))

    print("\n" + "=" * 80 + f"\nEXPERIMENT: {variant}  ({time.strftime('%Y-%m-%d %H:%M:%S')})\n" + "=" * 80)
    seed_everything()
    set_peft_model_state_dict(model, init_state)
    stages = STAGES[variant]

    if USE_WANDB:
        wandb.init(
            entity=WANDB_ENTITY, project=WANDB_PROJECT, group=WANDB_GROUP,
            name=f"{WANDB_RUN_PREFIX}-{variant}", reinit=True,
            config={
                "variant": variant, "spec": VARIANT_SPECS[variant], "transfer_mode": TRANSFER_MODE,
                "preset": ARGS.preset, "stages": {s: len(df) for s, df in stages},
                "stage_sources": {s: df["_src"].value_counts().to_dict() for s, df in stages},
                "dev_keep_patterns": DEV_KEEP_PATTERNS, "hard_reverse_aug": HARD_REVERSE_AUG,
                "mtvqa_splits": MTVQA_SPLITS, "model": MODEL_NAME, "load_in_4bit": LOAD_IN_4BIT,
                "prompt_repeat": PROMPT_REPEAT, "n_tta": N_TTA,
                "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "max_seq_len": MAX_SEQ_LEN,
                "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "finetune_vision": FINETUNE_VISION,
                "epochs": EPOCHS, "ext_epochs": EXT_EPOCHS, "learning_rate": LR, "ext_learning_rate": EXT_LR,
                "batch_size": BATCH_SIZE, "gradient_accumulation": GRAD_ACCUM,
                "shuffle_options": SHUFFLE_OPTIONS, "seed": SEED,
                "img_aug": VARIANT_SPECS[variant].get("img_aug", []),
                "aug_scope": AUG_SCOPE if VARIANT_SPECS[variant].get("img_aug") else None,
                "jpeg_quality": JPEG_QUALITY, "affine_translate": AFFINE_TRANSLATE,
                "affine_shear_deg": AFFINE_SHEAR_DEG, "dev_v2_drop_patterns": DEV_V2_DROP_PATTERNS,
            },
        )

    stage_logs = [train_stage(variant, s, df) for s, df in stages]
    adapter_dir = OUTPUT_DIR / f"{variant}__{stages[-1][0]}" / "adapter"

    t1 = time.time()
    valid_probs = predict_probs(valid_df, desc=f"{variant} validation")
    infer_sec = (time.time() - t1) / len(valid_df)
    valid_acc = accuracy(valid_df, valid_probs)
    np.save(OUTPUT_DIR / f"probs_valid_{variant}.npy", valid_probs)
    print(f"[{variant}] validation accuracy (TTA={N_TTA}): {valid_acc:.4f}")
    if USE_WANDB:
        wandb.log({"validation/accuracy": valid_acc})

    if SAVE_TEST_SUBMISSION:
        test_probs = predict_probs(test_df, desc=f"{variant} test")
        submission = pd.DataFrame({
            "id": test_df["id"].values,
            "answer": [CHOICES[i] for i in test_probs.argmax(axis=1)],
        })
        assert len(submission) == len(sample_sub)
        assert (submission["id"].astype(str).values == sample_sub["id"].astype(str).values).all()
        assert submission["answer"].isin(CHOICES).all()
        np.save(OUTPUT_DIR / f"probs_test_{variant}.npy", test_probs)
        submission.to_csv(OUTPUT_DIR / f"submission_{variant}.csv", index=False)
        print("saved:", OUTPUT_DIR / f"submission_{variant}.csv")

    if USE_WANDB:
        wandb.finish()

    result = {
        "variant": variant,
        "stages": " → ".join(f"{s['stage']}({s['rows']})" for s in stage_logs),
        "total_train_rows": sum(s["rows"] for s in stage_logs),
        "valid_acc": valid_acc,
        "train_loss_last": stage_logs[-1]["loss"],
        "train_min": round(sum(s["minutes"] for s in stage_logs), 1),
        "infer_sec_per_sample": round(infer_sec, 3),
        "adapter_dir": str(adapter_dir),
    }
    result_path(variant).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    gc.collect()
    torch.cuda.empty_cache()
    return result


# =====================================================================
# 속도 측정 모드
# =====================================================================
def run_benchmark(n_steps):
    variant = next((v for v in EXPERIMENT_VARIANTS if is_ordered(v)), EXPERIMENT_VARIANTS[0])
    stage, df = STAGES[variant][0]
    if is_ordered(variant):   # 증강(이미지 변환) 비용이 포함되도록 증강 블록부터 측정
        df = df[df["_aug"] != "none"].reset_index(drop=True)
    epochs, lr = stage_hparams(stage)
    print(f"\n[benchmark] {variant}/{stage}: {n_steps} step 학습 (batch={BATCH_SIZE}x{GRAD_ACCUM})")
    set_peft_model_state_dict(model, init_state)
    FastVisionModel.for_training(model)
    trainer = make_trainer(df, sft_args(OUTPUT_DIR / "_benchmark", epochs, lr,
                                         max_steps=n_steps, save=False, report=False),
                           ordered=is_ordered(variant))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    sec_step = (time.time() - t0) / n_steps
    peak = torch.cuda.max_memory_allocated() / 2**30
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(OUTPUT_DIR / "_benchmark", ignore_errors=True)

    n_inf = min(32, len(valid_df))
    t1 = time.time()
    predict_probs(valid_df.iloc[:n_inf], desc="benchmark infer")
    sec_sample = (time.time() - t1) / n_inf
    set_peft_model_state_dict(model, init_state)

    print(f"[benchmark] {sec_step:.1f} s/step | 학습 peak GPU {peak:.1f} GB | 추론(TTA={N_TTA}) {sec_sample:.2f} s/sample")
    print("  (첫 step의 컴파일·캐시 시간이 포함되어 실제보다 약간 크게 나옵니다)")
    n_infer = len(valid_df) + (len(test_df) if SAVE_TEST_SUBMISSION else 0)
    infer_h = sec_sample * n_infer / 3600
    total = 0.0
    for v in EXPERIMENT_VARIANTS:
        train_h = 0.0
        for s, sdf in STAGES[v]:
            ep, _ = stage_hparams(s)
            steps = math.ceil(math.ceil(len(sdf) / BATCH_SIZE) / GRAD_ACCUM * ep)
            train_h += steps * sec_step / 3600
        total += train_h + infer_h
        print(f"  {v:<24} 학습 {train_h:5.1f} h + 추론 {infer_h:4.1f} h = {train_h + infer_h:5.1f} h")
    print(f"  전체 예상: {total:.1f} h")


# =====================================================================
# 10. 실행
# =====================================================================
if ARGS.benchmark_steps > 0:
    run_benchmark(ARGS.benchmark_steps)
    sys.exit(0)

results, failed = [], []
for v in EXPERIMENT_VARIANTS:
    try:
        results.append(run_experiment(v))
    except Exception:
        print(f"\n[ERROR] {v} 실패:\n{traceback.format_exc()}")
        failed.append(v)
        if USE_WANDB and wandb.run is not None:
            wandb.finish(exit_code=1)
        gc.collect()
        torch.cuda.empty_cache()

if results:
    comparison = pd.DataFrame(results)
    if "raw" in set(comparison["variant"]):
        raw_acc = float(comparison.loc[comparison["variant"] == "raw", "valid_acc"].iloc[0])
        comparison["delta_vs_raw_pp"] = ((comparison["valid_acc"] - raw_acc) * 100).round(2)
    comparison["valid_acc"] = comparison["valid_acc"].round(4)
    comparison.to_csv(OUTPUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    print("\n[comparison]\n" + comparison.drop(columns=["adapter_dir"]).to_string(index=False))
    print("saved:", OUTPUT_DIR / "comparison.csv")
    done = [r["variant"] for r in results]
    print(f"\n문항 단위 비교:\npython compare_valid_predictions.py --probs-dir {OUTPUT_DIR} "
          f"--variants {' '.join(done)} --out-dir {OUTPUT_DIR / 'diff_report'}")

if failed:
    print(f"\n실패한 variant: {failed} → 같은 명령으로 재실행하면 완료된 부분은 건너뛰고 이어서 진행합니다.")
    sys.exit(1)
print(f"\nALL DONE ({time.strftime('%Y-%m-%d %H:%M:%S')})")
