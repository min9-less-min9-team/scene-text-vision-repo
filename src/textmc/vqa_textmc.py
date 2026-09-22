# -*- coding: utf-8 -*-
"""
Text-in-image multiple-choice VQA pipeline (SSAFY AI Challenge style).

Distilled from the two previous SSAFY winners:
  * MinjuJangg/ai_chall          (15기 2회차 1위)  choice-CE LoRA, choice-order TTA, margin ensemble
  * kimkihyun1/ssafy-ai-challenge-vqa (16기 1회차 1위)  Qwen3.5-27B LoRA, answer-only loss, TTA4, uncertainty metrics
and adapted for text/signboard questions (Korean OCR-heavy images).

Modes
  dryrun   : pandas/PIL only. Validates CSV columns, builds prompts, prints stats. Works on a laptop without torch.
  smoke    : tiny train + tiny predict on GPU to catch version/OOM issues (5 min).
  zeroshot : no training; TTA inference with the base model -> first submission fast.
  train    : LoRA fine-tuning (+ validation accuracy if --valid-size > 0).
  infer    : load adapter and predict test with TTA (supports --lora-off, --subset-ids, --crop-mode, hi-res).
  all      : train then infer.

Typical Colab usage
  !python vqa_textmc.py --mode dryrun  --root /content/drive/MyDrive/ssafy-16-2
  !python vqa_textmc.py --mode smoke   --root ... --model-id Qwen/Qwen3.5-27B
  !python vqa_textmc.py --mode zeroshot --root ... --model-id Qwen/Qwen3.5-27B --tta-perms 4 --out-dir runs/zs27b
  !python vqa_textmc.py --mode all     --root ... --model-id Qwen/Qwen3.5-27B --epochs 1 --lr 1e-4 --out-dir runs/ft27b

Outputs (in --out-dir)
  adapter/                       LoRA adapter (train)
  valid_predictions.csv          per-sample validation predictions with uncertainty (train, if valid split)
  test_tta_cache/*.npz           per-permutation probability cache (resumable)
  test_probs.npz                 averaged probs + per-perm runs + ids
  test_predictions_detailed.csv  id, answer, p_a..p_d, confidence, margin, entropy, agreement, unc_score, qtype
  submission.csv                 id order/columns matched to sample_submission.csv when present
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
CHOICES = ["a", "b", "c", "d"]

# Latin-square permutations: every original option appears once in every displayed slot.
PERM_BANK = [
    (0, 1, 2, 3),
    (1, 3, 0, 2),
    (2, 0, 3, 1),
    (3, 2, 1, 0),
]

# ----------------------------------------------------------------------------------------------
# Prompts (Korean text-VQA specific). Keep train and inference prompts identical.
# ----------------------------------------------------------------------------------------------
SYSTEM_PROMPT_TEXT = (
    "당신은 간판, 안내판, 현수막, 메뉴판, 표지판 등 이미지 속 글자를 정확히 읽고 질문에 답하는 전문가입니다. "
    "반드시 이미지에 실제로 보이는 글자와 그 위치 관계만 근거로 판단하세요. "
    "질문이 특정 층, 위치, 방향, 색상, 업종 같은 조건을 제시하면 그 조건에 맞는 간판의 글자만 찾고, "
    "조건에 맞지 않는 다른 간판이나 주변 글자는 무시하세요. "
    "보기의 글자가 이미지의 글자와 한 글자라도 다르면 오답일 수 있으므로 철자, 숫자, 띄어쓰기를 정확히 대조하세요. "
    "글자가 작거나 흐리면 그 부분을 확대해서 본다고 생각하고 다시 확인하세요. "
    "정답은 a, b, c, d 중 정확히 하나의 소문자 한 글자로만 답하세요."
)

SYSTEM_PROMPT_GENERIC = (
    "You are a highly accurate visual multiple-choice question answering model. "
    "Use the image and the answer choices. Return exactly one lowercase letter: a, b, c, or d."
)

QTYPE_RULES: List[Tuple[str, List[str]]] = [
    ("floor", ["층"]),
    ("phone", ["전화", "번호", "연락처"]),
    ("price", ["가격", "얼마", "원인", "원입니까", "원 인가"]),
    ("time", ["시간", "영업", "오픈", "마감", "시까지", "시부터", "휴무"]),
    ("count", ["몇 개", "몇개", "개수", "갯수", "총 몇"]),
    ("color", ["색"]),
    ("location", ["어디", "위치", "옆에", "왼쪽", "오른쪽", "위에", "아래에", "사이에"]),
    ("name", ["이름", "상호", "가게", "점포", "매장", "업체", "무엇이라고", "뭐라고", "적힌", "적혀", "쓰여", "써진", "쓰인"]),
]

QTYPE_HINTS: Dict[str, str] = {
    "floor": "층 표시(1F, 2층, B1 등)를 먼저 찾고, 같은 층에 붙어 있는 간판의 글자만 읽으세요. 건물의 위아래 순서로 층을 세어 확인하세요.",
    "phone": "숫자를 한 자리씩 대조하세요. 지역번호, 하이픈 위치, 자릿수까지 정확히 비교하세요.",
    "price": "숫자와 단위(원, 천원, 만원)를 정확히 읽고 메뉴명과 짝이 맞는 가격인지 확인하세요.",
    "time": "숫자와 시/분, 오전/오후, 요일 표기를 정확히 읽으세요.",
    "count": "질문 조건에 맞는 대상만 세고 중복해서 세지 마세요.",
    "color": "질문 대상 글자나 간판 자체의 색만 보고 배경이나 조명은 무시하세요.",
    "location": "기준 대상을 먼저 찾고 그 주변 간판의 상대 위치를 확인하세요.",
    "name": "질문 조건(층, 위치, 업종)에 맞는 간판 하나를 특정한 뒤 그 간판의 글자를 보기와 한 글자씩 대조하세요.",
    "other": "질문 조건에 맞는 글자를 찾아 보기와 정확히 대조하세요.",
}


def detect_qtype(question: str) -> str:
    q = str(question)
    for name, keys in QTYPE_RULES:
        if any(k in q for k in keys):
            return name
    return "other"


def build_user_prompt(question: str, options: Sequence[str], ocr_text: str = "", use_hints: bool = True) -> str:
    lines = [f"질문: {str(question).strip()}"]
    if use_hints:
        lines += ["", "판단 기준:", f"- {QTYPE_HINTS[detect_qtype(question)]}"]
    if ocr_text and str(ocr_text).strip() and str(ocr_text).strip().lower() != "nan":
        lines += ["", "[자동 OCR 참고 텍스트 (오류가 있을 수 있음, 이미지가 우선)]", str(ocr_text).strip()[:600]]
    lines.append("")
    for label, text in zip(CHOICES, options):
        lines.append(f"({label}) {text}")
    lines += ["", "이미지의 글자를 정확히 읽고 질문 조건에 맞는 보기를 고르세요. 정답은 a, b, c, d 중 한 글자만 출력하세요."]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Data handling: column detection, answer normalization, image resolution, group split
# ----------------------------------------------------------------------------------------------
ID_CANDIDATES = ["id", "ID", "Id", "image_id", "sample_id", "qid"]
PATH_CANDIDATES = ["path", "image_path", "img_path", "image", "img", "filename", "file", "file_name", "image_name"]
QUESTION_CANDIDATES = ["question", "query", "Question", "질문", "text"]
ANSWER_CANDIDATES = ["answer", "label", "Answer", "정답", "target"]
CHOICE_CANDIDATE_SETS = [
    ["a", "b", "c", "d"],
    ["A", "B", "C", "D"],
    ["choice_a", "choice_b", "choice_c", "choice_d"],
    ["option_a", "option_b", "option_c", "option_d"],
    ["opt_a", "opt_b", "opt_c", "opt_d"],
    ["option1", "option2", "option3", "option4"],
    ["choice1", "choice2", "choice3", "choice4"],
    ["1", "2", "3", "4"],
]


def _first_present(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


@dataclass
class Schema:
    id_col: str
    path_col: str
    question_col: str
    choice_cols: List[str]
    answer_col: Optional[str]


def detect_schema(df: pd.DataFrame, args) -> Schema:
    cols = list(df.columns)
    id_col = args.id_col or _first_present(cols, ID_CANDIDATES)
    path_col = args.path_col or _first_present(cols, PATH_CANDIDATES)
    q_col = args.question_col or _first_present(cols, QUESTION_CANDIDATES)
    ans_col = args.answer_col or _first_present(cols, ANSWER_CANDIDATES)
    if args.choice_cols:
        choice_cols = [c.strip() for c in args.choice_cols.split(",")]
    else:
        choice_cols = None
        for cand in CHOICE_CANDIDATE_SETS:
            if all(c in cols for c in cand):
                choice_cols = cand
                break
    missing = [n for n, v in [("id", id_col), ("path", path_col), ("question", q_col), ("choices", choice_cols)] if not v]
    if missing:
        raise ValueError(
            f"Could not detect columns {missing} in {cols}. "
            "Pass --id-col/--path-col/--question-col/--choice-cols a,b,c,d explicitly."
        )
    return Schema(id_col, path_col, q_col, list(choice_cols), ans_col)


def normalize_answer(raw: Any, options: Sequence[str]) -> Optional[str]:
    """Map answer to 'a'..'d'. Accepts a/A, 1-4, 0-3 (if --answer-zero-based), or option text."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    low = s.lower()
    if low in CHOICES:
        return low
    if re.fullmatch(r"\(?[abcd]\)?", low):
        return low.strip("()")
    if s in {"1", "2", "3", "4"}:
        return CHOICES[int(s) - 1]
    for label, text in zip(CHOICES, options):
        if str(text).strip() == s:
            return label
    return None


def standardize_df(df: pd.DataFrame, schema: Schema, is_train: bool) -> pd.DataFrame:
    out = pd.DataFrame()
    out["id"] = df[schema.id_col].astype(str)
    out["path"] = df[schema.path_col].astype(str)
    out["question"] = df[schema.question_col].astype(str)
    for label, col in zip(CHOICES, schema.choice_cols):
        out[label] = df[col].astype(str).str.strip()
    if is_train:
        if schema.answer_col is None:
            raise ValueError("Train CSV has no answer column; pass --answer-col")
        answers = []
        for raw, opts in zip(df[schema.answer_col].tolist(), out[CHOICES].values.tolist()):
            answers.append(normalize_answer(raw, opts))
        out["answer"] = answers
        bad = out["answer"].isna().sum()
        if bad:
            print(f"[warn] {bad} train rows have unparseable answers and will be dropped")
        out = out[out["answer"].notna()].reset_index(drop=True)
    out["qtype"] = out["question"].map(detect_qtype)
    return out


def resolve_image_path(raw: str, roots: Sequence[Path]) -> Optional[Path]:
    p = Path(str(raw))
    cands: List[Path] = []
    if p.is_absolute():
        cands.append(p)
    for r in roots:
        cands += [r / p, r / p.name, r / "train" / p.name, r / "test" / p.name, r / "images" / p.name]
    for c in cands:
        if c.exists():
            return c
    return None


def attach_paths(df: pd.DataFrame, roots: Sequence[Path]) -> pd.DataFrame:
    resolved = [resolve_image_path(p, roots) for p in df["path"].tolist()]
    df = df.copy()
    df["abs_path"] = [str(p) if p else "" for p in resolved]
    missing = int((df["abs_path"] == "").sum())
    if missing:
        print(f"[warn] {missing}/{len(df)} images not found. Examples: {df.loc[df['abs_path']=='', 'path'].head(3).tolist()}")
    return df


def load_image(path: str) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def file_hash(path: str, nbytes: int = 1 << 20) -> str:
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            h.update(f.read(nbytes))
    except Exception:
        h.update(str(path).encode())
    return h.hexdigest()


def group_split(df: pd.DataFrame, valid_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split so that the same image never appears in both train and validation."""
    if valid_size <= 0:
        return df.reset_index(drop=True), df.iloc[0:0].copy()
    keys = df["abs_path"].where(df["abs_path"] != "", df["path"]).map(file_hash)
    groups = keys.unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_valid = max(1, int(round(len(groups) * valid_size)))
    valid_groups = set(groups[:n_valid])
    is_valid = keys.isin(valid_groups)
    return df[~is_valid].reset_index(drop=True), df[is_valid].reset_index(drop=True)


def make_crop(image: Image.Image, mode: str, fraction: float = 0.72) -> Image.Image:
    w, h = image.size
    cw, ch = int(w * fraction), int(h * fraction)
    if mode == "full":
        return image
    if mode == "top":
        return image.crop((0, 0, w, int(h * 0.6)))
    if mode == "bottom":
        return image.crop((0, int(h * 0.4), w, h))
    boxes = {
        "center": ((w - cw) // 2, (h - ch) // 2),
        "tl": (0, 0),
        "tr": (w - cw, 0),
        "bl": (0, h - ch),
        "br": (w - cw, h - ch),
    }
    x, y = boxes[mode]
    return image.crop((x, y, x + cw, y + ch))


CROP_SETS = {
    "none": ["full"],
    "five": ["center", "tl", "tr", "bl", "br"],
    "topbottom": ["top", "bottom"],
}


# ----------------------------------------------------------------------------------------------
# Torch / model side (lazy imports so dryrun works without torch)
# ----------------------------------------------------------------------------------------------
class ML:
    torch = None
    transformers = None
    peft = None


def load_ml():
    if ML.torch is None:
        import torch  # noqa
        import transformers  # noqa
        import peft  # noqa

        ML.torch, ML.transformers, ML.peft = torch, transformers, peft
        print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
    return ML.torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if ML.torch is not None:
        ML.torch.manual_seed(seed)
        ML.torch.cuda.manual_seed_all(seed)


def apply_template(processor, messages, add_generation_prompt: bool) -> str:
    """Disable thinking mode when the template supports it (Qwen3.5/3.6/3.8, Qwen3-VL)."""
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
        )
    except TypeError:
        try:
            return processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def make_messages(images: Sequence[Image.Image], user_text: str, system_text: str, gold: Optional[str] = None):
    content = [{"type": "image", "image": im} for im in images]
    if len(images) > 1:
        user_text = (
            f"첫 번째 이미지는 원본 전체이고, 나머지 {len(images)-1}개는 같은 이미지의 확대 영역입니다. "
            "확대 영역의 글자를 원본 맥락과 함께 판단하세요.\n\n" + user_text
        )
    content.append({"type": "text", "text": user_text})
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": content},
    ]
    if gold is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": gold}]})
    return msgs


def visual_token_side(processor) -> int:
    ip = getattr(processor, "image_processor", None)
    patch = getattr(ip, "patch_size", None) or 16
    merge = getattr(ip, "merge_size", None) or 2
    return int(patch) * int(merge)


def build_processor(model_id: str, min_tokens: int, max_tokens: int):
    from transformers import AutoProcessor

    tmp = AutoProcessor.from_pretrained(model_id)
    side = visual_token_side(tmp)
    min_pixels, max_pixels = min_tokens * side * side, max_tokens * side * side
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    # Some processors ignore kwargs above; set directly as well.
    ip = getattr(processor, "image_processor", None)
    for attr, val in [("min_pixels", min_pixels), ("max_pixels", max_pixels)]:
        if ip is not None and hasattr(ip, attr):
            setattr(ip, attr, val)
    if ip is not None and hasattr(ip, "size") and isinstance(ip.size, dict):
        if "shortest_edge" in ip.size:
            ip.size["shortest_edge"] = min_pixels
        if "longest_edge" in ip.size:
            ip.size["longest_edge"] = max_pixels
    processor.tokenizer.padding_side = "right"
    print(f"processor={type(processor).__name__} token_side={side}px min_pixels={min_pixels} max_pixels={max_pixels}")
    return processor


def choice_token_ids(processor):
    ids = []
    for c in CHOICES:
        cands = [
            processor.tokenizer.encode(c, add_special_tokens=False),
            processor.tokenizer.encode(f" {c}", add_special_tokens=False),
        ]
        tok = next((t[0] for t in cands if len(t) == 1), None)
        if tok is None:
            raise RuntimeError(f"choice {c!r} is not a single token: {cands}")
        ids.append(int(tok))
    print("choice token ids:", dict(zip(CHOICES, ids)))
    return ids


def load_base_model(model_id: str, precision: str, device_map: Any = None):
    torch = load_ml()
    from transformers import BitsAndBytesConfig

    kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
        "device_map": device_map if device_map is not None else {"": 0},
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    }
    if precision == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=kwargs["torch_dtype"],
        )
    last = None
    for cls_name in ["AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForVision2Seq"]:
        cls = getattr(ML.transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_id, **kwargs)
            print(f"loaded {model_id} via {cls_name} precision={precision}")
            return model
        except Exception as exc:  # noqa
            last = exc
            print(f"[info] {cls_name} failed: {str(exc)[:200]}")
    raise RuntimeError(f"could not load {model_id}") from last


LORA_CANDIDATES = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                   "out_proj", "gate_proj", "up_proj", "down_proj"]
VISION_MARKERS = ("visual", "vision", "vit", "image_encoder", "merger", "patch_embed")


def detect_lora_targets(model, include_vision: bool) -> List[str]:
    torch = ML.torch
    names: List[str] = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in LORA_CANDIDATES:
            continue
        if not isinstance(module, torch.nn.Linear) and "Linear" not in type(module).__name__:
            continue
        if not include_vision and any(m in name.lower() for m in VISION_MARKERS):
            continue
        names.append(name)
    leaves = sorted({n.rsplit(".", 1)[-1] for n in names})
    print(f"LoRA target leaves: {leaves} ({len(names)} modules, include_vision={include_vision})")
    if len(names) == 0:
        raise RuntimeError("no LoRA targets found; inspect model.named_modules()")
    return names  # full names -> exact targeting (safe for vision exclusion)


def attach_lora(model, args, precision: str):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if precision in {"8bit", "4bit"}:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    targets = detect_lora_targets(model, args.lora_vision)
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
                     target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def is_oom(exc: Exception) -> bool:
    torch = ML.torch
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    m = str(exc).lower()
    return "out of memory" in m or "cublas_status_alloc_failed" in m


def cleanup(model=None):
    if model is not None:
        del model
    gc.collect()
    if ML.torch is not None:
        ML.torch.cuda.empty_cache()


def gpu_report(prefix: str):
    torch = ML.torch
    if not torch.cuda.is_available():
        return
    print(f"{prefix} | alloc={torch.cuda.memory_allocated()/2**30:.1f}GB reserved={torch.cuda.memory_reserved()/2**30:.1f}GB "
          f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB")


# ----------------------------------------------------------------------------------------------
# Datasets / collators
# ----------------------------------------------------------------------------------------------
class Ctx:
    """Runtime context shared by datasets (processor, prompts, options)."""
    processor = None
    system_text = SYSTEM_PROMPT_TEXT
    use_hints = True
    ocr_map: Dict[str, str] = {}
    choice_ids: List[int] = []


def sample_images(row, crop_modes: Sequence[str]) -> List[Image.Image]:
    image = load_image(row["abs_path"])
    if crop_modes == ["full"]:
        return [image]
    return [image] + [make_crop(image, m) for m in crop_modes]


class TrainDS:
    def __init__(self, df: pd.DataFrame, shuffle_options: bool):
        self.df = df.reset_index(drop=True)
        self.shuffle_options = shuffle_options

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        options = [str(row[c]) for c in CHOICES]
        gold_orig = CHOICES.index(row["answer"])
        perm = np.random.permutation(4).tolist() if self.shuffle_options else [0, 1, 2, 3]
        shown = [options[j] for j in perm]
        gold = CHOICES[perm.index(gold_orig)]
        text = build_user_prompt(row["question"], shown, Ctx.ocr_map.get(row["id"], ""), Ctx.use_hints)
        images = sample_images(row, ["full"])
        return {"images": images, "prompt_msgs": make_messages(images, text, Ctx.system_text),
                "full_msgs": make_messages(images, text, Ctx.system_text, gold=gold), "gold": gold}


class EvalDS:
    def __init__(self, df: pd.DataFrame, perm: Sequence[int], crop_modes: Sequence[str]):
        self.df = df.reset_index(drop=True)
        self.perm = list(perm)
        self.crop_modes = list(crop_modes)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        options = [str(row[c]) for c in CHOICES]
        shown = [options[j] for j in self.perm]
        text = build_user_prompt(row["question"], shown, Ctx.ocr_map.get(row["id"], ""), Ctx.use_hints)
        images = sample_images(row, self.crop_modes)
        return {"images": images, "msgs": make_messages(images, text, Ctx.system_text), "id": row["id"]}


def _encode(texts: List[str], images_nested: List[List[Image.Image]]):
    flat = [im for ims in images_nested for im in ims]
    return Ctx.processor(text=texts, images=flat, padding=True, return_tensors="pt")


def collate_train(batch, loss_mode: str):
    torch = ML.torch
    if loss_mode == "choice_ce":
        texts = [apply_template(Ctx.processor, b["prompt_msgs"], True) for b in batch]
        enc = _encode(texts, [b["images"] for b in batch])
        enc["choice_labels"] = torch.tensor([CHOICES.index(b["gold"]) for b in batch], dtype=torch.long)
        return enc
    texts = [apply_template(Ctx.processor, b["full_msgs"], False) for b in batch]
    enc = _encode(texts, [b["images"] for b in batch])
    labels = torch.full_like(enc["input_ids"], -100)
    for i, b in enumerate(batch):
        gold_id = Ctx.choice_ids[CHOICES.index(b["gold"])]
        pos = torch.where((enc["input_ids"][i] == gold_id) & enc["attention_mask"][i].bool())[0]
        if len(pos) == 0:
            raise RuntimeError("gold token not found in encoded sequence")
        labels[i, int(pos[-1])] = gold_id
    enc["labels"] = labels
    return enc


def collate_eval(batch):
    texts = [apply_template(Ctx.processor, b["msgs"], True) for b in batch]
    enc = _encode(texts, [b["images"] for b in batch])
    return enc, [b["id"] for b in batch]


def last_positions(attention_mask):
    return attention_mask.sum(dim=1).long() - 1


def choice_logits_from(out_logits, attention_mask, choice_ids_t):
    idx = last_positions(attention_mask).to(out_logits.device)
    idx = idx.clamp(max=out_logits.shape[1] - 1)
    b = ML.torch.arange(out_logits.shape[0], device=out_logits.device)
    return out_logits[b, idx][:, choice_ids_t.to(out_logits.device)].float()


def to_device(enc, device):
    torch = ML.torch
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in enc.items()}


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------
def train_lora(model, train_df: pd.DataFrame, args, out_dir: Path):
    torch = ML.torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    device = next(model.parameters()).device
    ds = TrainDS(train_df, shuffle_options=not args.no_shuffle_options)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=lambda b: collate_train(b, args.loss))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(dl) / args.grad_accum)
    total_steps = max(1, int(math.ceil(args.epochs * steps_per_epoch)))
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * args.warmup_ratio), total_steps)
    choice_ids_t = torch.tensor(Ctx.choice_ids, dtype=torch.long)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    max_micro = int(math.ceil(args.epochs * len(dl)))
    print(f"train samples={len(ds)} micro-batches/epoch={len(dl)} optimizer steps={total_steps} epochs={args.epochs}")
    model.train()
    opt.zero_grad(set_to_none=True)
    step_micro, running, seen, t0 = 0, 0.0, 0, time.time()
    done = False
    epoch = 0
    while not done:
        epoch += 1
        for enc in dl:
            step_micro += 1
            enc = to_device(enc, device)
            try:
                with torch.autocast("cuda", dtype=dtype):
                    if args.loss == "choice_ce":
                        labels = enc.pop("choice_labels")
                        out = model(**enc, use_cache=False)
                        logits = choice_logits_from(out.logits, enc["attention_mask"], choice_ids_t)
                        loss = torch.nn.functional.cross_entropy(logits, labels.to(logits.device),
                                                                 label_smoothing=args.label_smoothing)
                    else:
                        out = model(**enc, use_cache=False)
                        loss = out.loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss {loss}")
                (loss / args.grad_accum).backward()
            except Exception as exc:
                if is_oom(exc):
                    gpu_report("OOM")
                    raise RuntimeError("CUDA OOM during training: lower --max-visual-tokens, use --precision 8bit/4bit, "
                                       "or --batch-size 1") from exc
                raise
            running += float(loss.detach()); seen += 1
            if step_micro % args.grad_accum == 0 or step_micro == max_micro:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            if step_micro % args.log_every == 0:
                el = time.time() - t0
                print(f"ep{epoch} micro {step_micro}/{max_micro} loss={running/seen:.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={el/60:.1f}m eta={(el/step_micro)*(max_micro-step_micro)/60:.1f}m", flush=True)
            del out, loss, enc
            if step_micro >= max_micro:
                done = True
                break
        if args.save_every_epoch and not done:
            model.save_pretrained(out_dir / f"adapter_epoch{epoch}")
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    print("adapter saved:", adapter_dir, f"| train loss={running/max(seen,1):.4f}")
    return adapter_dir


# ----------------------------------------------------------------------------------------------
# Inference with TTA + uncertainty
# ----------------------------------------------------------------------------------------------
def perms_for(n: int) -> List[Tuple[int, ...]]:
    bank = list(PERM_BANK)
    if n <= 4:
        return bank[:n]
    rest = [p for p in itertools.permutations(range(4)) if p not in bank]
    return (bank + rest)[:n]


def predict_probs(model, df: pd.DataFrame, perm: Sequence[int], crop_modes: Sequence[str], args, desc: str) -> np.ndarray:
    torch = ML.torch
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    ds = EvalDS(df, perm, crop_modes)
    dl = DataLoader(ds, batch_size=args.infer_batch_size, shuffle=False, num_workers=args.num_workers,
                    collate_fn=collate_eval)
    choice_ids_t = torch.tensor(Ctx.choice_ids, dtype=torch.long)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.eval()
    shown_probs, t0 = [], time.time()
    with torch.inference_mode():
        for bi, (enc, ids) in enumerate(dl, 1):
            enc = to_device(enc, device)
            with torch.autocast("cuda", dtype=dtype):
                out = model(**enc, use_cache=False)
            logits = choice_logits_from(out.logits, enc["attention_mask"], choice_ids_t)
            shown_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            if bi % args.log_every == 0:
                el = time.time() - t0
                print(f"{desc} {bi}/{len(dl)} elapsed={el/60:.1f}m eta={(el/bi)*(len(dl)-bi)/60:.1f}m", flush=True)
            del out, logits, enc
    shown = np.concatenate(shown_probs, axis=0)
    orig = np.zeros_like(shown)
    for shown_j, orig_j in enumerate(perm):
        orig[:, orig_j] = shown[:, shown_j]
    return orig


def predict_tta(model, df: pd.DataFrame, args, cache_dir: Optional[Path], tag: str, crop_modes=("full",)) -> Dict[str, Any]:
    runs = []
    ids = df["id"].tolist()
    for pi, perm in enumerate(perms_for(args.tta_perms), 1):
        pkey = "".join(map(str, perm))
        cache = cache_dir / f"{tag}_perm{pkey}_crop{'-'.join(crop_modes)}.npz" if cache_dir else None
        if cache is not None and cache.exists():
            z = np.load(cache, allow_pickle=True)
            if z["ids"].tolist() == ids:
                print(f"[cache] {cache.name}")
                runs.append(z["probs"]); continue
            print(f"[cache] id mismatch, recomputing {cache.name}")
        probs = predict_probs(model, df, perm, crop_modes, args, desc=f"{tag} perm{pi}/{args.tta_perms}")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, probs=probs, ids=np.array(ids, dtype=object))
        runs.append(probs)
    runs_arr = np.stack(runs, axis=0)
    return {"runs": runs_arr, "avg": runs_arr.mean(axis=0), "ids": ids}


def uncertainty(runs: np.ndarray) -> Dict[str, np.ndarray]:
    avg = runs.mean(axis=0)
    s = np.sort(avg, axis=1)
    margin = s[:, -1] - s[:, -2]
    conf = s[:, -1]
    ent = -np.sum(avg * np.log(np.clip(avg, 1e-12, 1)), axis=1) / math.log(4)
    votes = runs.argmax(axis=2)  # [T, N]
    agree = np.array([np.bincount(votes[:, i], minlength=4).max() / runs.shape[0] for i in range(runs.shape[1])])
    score = 0.40 * (1 - margin) + 0.30 * ent + 0.15 * (1 - conf) + 0.15 * (1 - agree)
    return {"confidence": conf, "margin": margin, "entropy": ent, "agreement": agree, "unc_score": score}


def detailed_frame(df: pd.DataFrame, res: Dict[str, Any]) -> pd.DataFrame:
    avg, runs = res["avg"], res["runs"]
    unc = uncertainty(runs)
    out = pd.DataFrame({"id": res["ids"], "answer": np.array(CHOICES)[avg.argmax(axis=1)]})
    for j, c in enumerate(CHOICES):
        out[f"p_{c}"] = avg[:, j]
    for k, v in unc.items():
        out[k] = v
    out["qtype"] = df["qtype"].values
    out["question"] = df["question"].values
    return out


def write_submission(pred_df: pd.DataFrame, sample_path: Optional[Path], out_path: Path):
    sub = pred_df[["id", "answer"]].copy()
    if sample_path and sample_path.exists():
        sample = pd.read_csv(sample_path)
        id_col = sample.columns[0]
        ans_col = sample.columns[1] if len(sample.columns) > 1 else "answer"
        sample[id_col] = sample[id_col].astype(str)
        merged = sample[[id_col]].merge(sub.rename(columns={"id": id_col}), on=id_col, how="left")
        if merged["answer"].isna().any():
            missing = merged.loc[merged["answer"].isna(), id_col].head(5).tolist()
            raise RuntimeError(f"submission ids missing predictions, e.g. {missing}")
        # Preserve original id dtype if sample ids were numeric
        merged = merged.rename(columns={"answer": ans_col})
        if pd.api.types.is_numeric_dtype(pd.read_csv(sample_path)[id_col]):
            merged[id_col] = pd.to_numeric(merged[id_col])
        # Match answer letter case used in sample file if it looks like uppercase letters
        sample_vals = sample[ans_col].astype(str).str.strip() if ans_col in sample.columns else pd.Series(dtype=str)
        if len(sample_vals) and sample_vals.str.fullmatch(r"[ABCD]").mean() > 0.5:
            merged[ans_col] = merged[ans_col].str.upper()
        merged.to_csv(out_path, index=False)
    else:
        sub.to_csv(out_path, index=False)
    print("submission written:", out_path)


def accuracy_report(det: pd.DataFrame, gold: pd.Series, name: str):
    correct = det["answer"].values == gold.values
    print(f"[{name}] accuracy = {correct.mean():.4f} ({correct.sum()}/{len(correct)})")
    by = pd.DataFrame({"qtype": det["qtype"], "ok": correct}).groupby("qtype")["ok"].agg(["mean", "count"])
    print(by.to_string())
    # Accuracy on the most uncertain quartile: tells whether unc_score is informative.
    q = det["unc_score"].quantile(0.75)
    hard = det["unc_score"] >= q
    print(f"[{name}] accuracy on top-25% uncertain = {correct[hard.values].mean():.4f} | rest = {correct[~hard.values].mean():.4f}")


# ----------------------------------------------------------------------------------------------
# Main modes
# ----------------------------------------------------------------------------------------------
def load_frames(args) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[Path]]:
    root = Path(args.root)
    train_csv = Path(args.train_csv) if args.train_csv else root / "train.csv"
    test_csv = Path(args.test_csv) if args.test_csv else root / "test.csv"
    sample_csv = Path(args.sample_csv) if args.sample_csv else root / "sample_submission.csv"
    roots = [Path(args.image_root)] if args.image_root else []
    roots += [root, root / "images", train_csv.parent]
    train_df = None
    if train_csv.exists():
        raw = pd.read_csv(train_csv)
        schema = detect_schema(raw, args)
        print("train schema:", schema)
        train_df = attach_paths(standardize_df(raw, schema, True), roots)
        if args.max_train_samples:
            train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)), random_state=args.seed).reset_index(drop=True)
    test_df = None
    if test_csv.exists():
        raw = pd.read_csv(test_csv)
        schema = detect_schema(raw, args)
        test_df = attach_paths(standardize_df(raw, schema, False), roots)
        if args.max_test_samples:
            test_df = test_df.head(args.max_test_samples).reset_index(drop=True)
    if args.ocr_csv:
        ocr = pd.read_csv(args.ocr_csv)
        Ctx.ocr_map = dict(zip(ocr.iloc[:, 0].astype(str), ocr.iloc[:, 1].astype(str)))
        print(f"OCR hints loaded for {len(Ctx.ocr_map)} ids")
    if args.subset_ids and test_df is not None:
        keep = set(Path(args.subset_ids).read_text(encoding="utf-8").split())
        test_df = test_df[test_df["id"].isin(keep)].reset_index(drop=True)
        print(f"test subset: {len(test_df)} rows")
    return train_df, test_df, sample_csv


def maybe_copy_local(df: Optional[pd.DataFrame], dst_root: Optional[str]):
    if df is None or not dst_root:
        return df
    dst = Path(dst_root)
    dst.mkdir(parents=True, exist_ok=True)
    new_paths = []
    for p in df["abs_path"].tolist():
        if not p:
            new_paths.append(p); continue
        target = dst / Path(p).name
        if not target.exists():
            shutil.copy2(p, target)
        new_paths.append(str(target))
    df = df.copy(); df["abs_path"] = new_paths
    return df


def mode_dryrun(args):
    train_df, test_df, sample_csv = load_frames(args)
    for name, df in [("train", train_df), ("test", test_df)]:
        if df is None:
            print(f"{name}: (missing)"); continue
        print(f"\n=== {name}: {len(df)} rows ===")
        print("qtype distribution:\n", df["qtype"].value_counts().to_string())
        if "answer" in df:
            print("answer distribution:\n", df["answer"].value_counts(normalize=True).sort_index().round(3).to_string())
        found = int((df["abs_path"] != "").sum())
        print(f"images resolved: {found}/{len(df)}")
        sizes = []
        for p in df.loc[df["abs_path"] != "", "abs_path"].head(50):
            try:
                with Image.open(p) as im:
                    sizes.append(im.size)
            except Exception as e:
                print("bad image", p, e)
        if sizes:
            ws, hs = zip(*sizes)
            print(f"image size sample: w[{min(ws)}..{max(ws)}] h[{min(hs)}..{max(hs)}] median={int(np.median(ws))}x{int(np.median(hs))}")
    if train_df is not None:
        tr, va = group_split(train_df, args.valid_size, args.seed)
        print(f"\ngroup split: train={len(tr)} valid={len(va)} (valid_size={args.valid_size})")
        row = tr.iloc[0]
        print("\n=== sample prompt ===")
        print("SYSTEM:", SYSTEM_PROMPT_TEXT if args.prompt_style == "text" else SYSTEM_PROMPT_GENERIC)
        print("USER:\n" + build_user_prompt(row["question"], [row[c] for c in CHOICES], Ctx.ocr_map.get(row["id"], ""), not args.no_hints))
        print("GOLD:", row["answer"])
    print("\nsample_submission:", sample_csv, "exists" if sample_csv.exists() else "missing")
    print("dryrun OK")


def build_model_for_training(args, train_df: pd.DataFrame):
    torch = load_ml()
    attempts = ["bf16", "8bit"] if args.precision == "auto" else [args.precision]
    last = None
    for prec in attempts:
        model = None
        try:
            model = load_base_model(args.model_id, prec)
            model = attach_lora(model, args, prec)
            # preflight one micro-batch forward/backward
            ds = TrainDS(train_df.head(4), shuffle_options=True)
            enc = collate_train([ds[0]], args.loss)
            enc = to_device(enc, next(model.parameters()).device)
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                if args.loss == "choice_ce":
                    labels = enc.pop("choice_labels")
                    out = model(**enc, use_cache=False)
                    loss = torch.nn.functional.cross_entropy(
                        choice_logits_from(out.logits, enc["attention_mask"], torch.tensor(Ctx.choice_ids)), labels.to(out.logits.device))
                else:
                    out = model(**enc, use_cache=False); loss = out.loss
            loss.backward(); model.zero_grad(set_to_none=True)
            print(f"preflight ok precision={prec} loss={float(loss):.4f}")
            gpu_report("preflight")
            del out, loss, enc
            gc.collect(); torch.cuda.empty_cache()
            return model, prec
        except Exception as exc:
            last = exc
            cleanup(model)
            if is_oom(exc) and args.precision == "auto" and prec != attempts[-1]:
                print(f"[warn] OOM at {prec}; falling back to next precision")
                continue
            raise
    raise RuntimeError("model build failed") from last


def load_model_for_inference(args):
    torch = load_ml()
    from peft import PeftModel

    prec = "bf16" if args.precision == "auto" else args.precision
    model = load_base_model(args.model_id, prec)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        print("adapter loaded:", args.adapter)
    model.eval()
    return model


def run_inference(model, test_df: pd.DataFrame, args, out_dir: Path, sample_csv: Optional[Path], tag: str):
    crop_modes = CROP_SETS[args.crop_mode]
    cache_dir = out_dir / "test_tta_cache"
    ctx = model.disable_adapter() if (args.lora_off and hasattr(model, "disable_adapter")) else None
    if ctx is not None:
        print("inference with LoRA adapter DISABLED (base model)")
        with ctx:
            res = predict_tta(model, test_df, args, cache_dir, tag + "_loraoff", crop_modes)
    else:
        res = predict_tta(model, test_df, args, cache_dir, tag, crop_modes)
    np.savez_compressed(out_dir / f"{tag}_probs.npz", avg=res["avg"], runs=res["runs"], ids=np.array(res["ids"], dtype=object))
    det = detailed_frame(test_df, res)
    det.to_csv(out_dir / f"{tag}_predictions_detailed.csv", index=False, encoding="utf-8-sig")
    write_submission(det, sample_csv, out_dir / f"submission_{tag}.csv")
    print("answer distribution:\n", det["answer"].value_counts(normalize=True).sort_index().round(3).to_string())
    print(f"mean margin={det['margin'].mean():.3f} | low-margin(<0.15) count={(det['margin']<0.15).sum()}")
    return det


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["dryrun", "smoke", "zeroshot", "train", "infer", "all"])
    p.add_argument("--root", default=".", help="folder with train.csv/test.csv/sample_submission.csv and images")
    p.add_argument("--train-csv"); p.add_argument("--test-csv"); p.add_argument("--sample-csv"); p.add_argument("--image-root")
    p.add_argument("--id-col"); p.add_argument("--path-col"); p.add_argument("--question-col"); p.add_argument("--answer-col")
    p.add_argument("--choice-cols", help="comma separated 4 columns, e.g. a,b,c,d")
    p.add_argument("--ocr-csv", help="csv with columns id, ocr_text to append as hint")
    p.add_argument("--copy-local", help="copy images to this local dir first (Colab: /content/data)")
    p.add_argument("--out-dir", default="runs/default")
    p.add_argument("--model-id", default="Qwen/Qwen3.5-27B")
    p.add_argument("--adapter", help="adapter dir for infer (default: <out-dir>/adapter if exists)")
    p.add_argument("--precision", default="auto", choices=["auto", "bf16", "8bit", "4bit"])
    p.add_argument("--min-visual-tokens", type=int, default=256)
    p.add_argument("--max-visual-tokens", type=int, default=1280, help="1024 = winner default; 1280-1792 for small text; 768 if OOM")
    p.add_argument("--prompt-style", default="text", choices=["text", "generic"])
    p.add_argument("--no-hints", action="store_true", help="drop qtype hint line from the prompt")
    p.add_argument("--loss", default="choice_ce", choices=["choice_ce", "answer_lm"])
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--no-shuffle-options", action="store_true")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--infer-batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lora-r", type=int, default=16); p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-vision", action="store_true", help="also adapt vision tower linear layers")
    p.add_argument("--valid-size", type=float, default=0.1, help="0 = train on everything (final run)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tta-perms", type=int, default=4, help="1..24 choice-order permutations")
    p.add_argument("--crop-mode", default="none", choices=list(CROP_SETS))
    p.add_argument("--lora-off", action="store_true", help="infer with adapter disabled (base model view for blending)")
    p.add_argument("--subset-ids", help="text file of test ids (one per line) to restrict inference (rerun of uncertain items)")
    p.add_argument("--max-train-samples", type=int, default=0); p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Ctx.system_text = SYSTEM_PROMPT_TEXT if args.prompt_style == "text" else SYSTEM_PROMPT_GENERIC
    Ctx.use_hints = not args.no_hints
    (out_dir / "run_args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    if args.mode == "dryrun":
        mode_dryrun(args); return

    torch = load_ml()
    assert torch.cuda.is_available(), "CUDA GPU required for this mode"
    print("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f}GB")
    set_seed(args.seed)
    train_df, test_df, sample_csv = load_frames(args)
    train_df = maybe_copy_local(train_df, args.copy_local and str(Path(args.copy_local) / "train"))
    test_df = maybe_copy_local(test_df, args.copy_local and str(Path(args.copy_local) / "test"))
    Ctx.processor = build_processor(args.model_id, args.min_visual_tokens, args.max_visual_tokens)
    Ctx.choice_ids = choice_token_ids(Ctx.processor)

    if args.mode == "smoke":
        args.max_train_samples = args.max_train_samples or 16
        args.epochs = min(args.epochs, 0.5) if args.epochs else 0.5
        args.tta_perms = 2
        train_df = train_df.head(16)
        test_sub = (test_df if test_df is not None else train_df).head(8)
        model, prec = build_model_for_training(args, train_df)
        train_lora(model, train_df, args, out_dir)
        run_inference(model, test_sub, args, out_dir, None, "smoke")
        print("SMOKE OK"); return

    if args.mode == "zeroshot":
        model = load_model_for_inference(args)
        if train_df is not None and args.valid_size > 0:
            _, va = group_split(train_df, args.valid_size, args.seed)
            va = va.head(args.max_test_samples or 400)
            det = run_inference(model, va, args, out_dir, None, "zeroshot_valid")
            accuracy_report(det, va["answer"], "zeroshot valid")
        if test_df is not None:
            run_inference(model, test_df, args, out_dir, sample_csv, "zeroshot")
        return

    if args.mode in {"train", "all"}:
        assert train_df is not None, "train.csv not found"
        tr, va = group_split(train_df, args.valid_size, args.seed)
        print(f"train={len(tr)} valid={len(va)}")
        model, prec = build_model_for_training(args, tr)
        train_lora(model, tr, args, out_dir)
        if len(va):
            det = run_inference(model, va, args, out_dir, None, "valid")
            accuracy_report(det, va["answer"], "valid")
            det["gold"] = va["answer"].values
            det.to_csv(out_dir / "valid_predictions.csv", index=False, encoding="utf-8-sig")
        if args.mode == "all" and test_df is not None:
            run_inference(model, test_df, args, out_dir, sample_csv, "test")
        return

    if args.mode == "infer":
        if not args.adapter and (out_dir / "adapter").exists():
            args.adapter = str(out_dir / "adapter")
        model = load_model_for_inference(args)
        assert test_df is not None, "test.csv not found"
        tag = "test"
        if args.crop_mode != "none":
            tag += f"_crop{args.crop_mode}"
        if args.subset_ids:
            tag += "_subset"
        if args.lora_off:
            tag += "_loraoff"
        run_inference(model, test_df, args, out_dir, sample_csv, tag)


def _utf8_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


if __name__ == "__main__":
    _utf8_console()
    main()
