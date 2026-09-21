from unsloth import FastVisionModel  # isort: skip

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import Config, parse_config
from .data import CHOICES, build_messages, load_split, subsample
from .model import load_model


def load_image(data_dir: str, path: str) -> Image.Image:
    return Image.open(os.path.join(data_dir, path)).convert("RGB")


@torch.inference_mode()
def predict_batch(model, processor, rows: list, data_dir: str) -> np.ndarray:
    """[B,4] probabilities over a/b/c/d from the next-token logits of one left-padded batch."""
    choice_ids = [processor.tokenizer.encode(c, add_special_tokens=False)[0] for c in CHOICES]
    # enable_thinking=False: Qwen3 계열이 <think>로 시작하지 않고 바로 답 글자를 내도록 (다른 템플릿은 무시)
    texts = [
        processor.apply_chat_template(
            build_messages(row, data_dir, with_answer=False),
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        for row in rows
    ]
    images = [load_image(data_dir, row["path"]) for row in rows]
    # left padding이라 모든 샘플의 마지막 위치(-1)가 답 글자 자리
    processor.tokenizer.padding_side = "left"
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True).to(model.device)
    # autocast: bf16 로드(--no-load-in-4bit) 시 float32 pixel_values와 dtype이 어긋나지 않도록
    # logits_to_keep=1: 마지막 위치만 vocab으로 투영 (전체 [B,T,V] logits는 배치 추론에서 VRAM을 가장 많이 먹음)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**inputs, logits_to_keep=1).logits[:, -1, choice_ids]
    return logits.float().softmax(-1).cpu().numpy()


def predict_probs(model, processor, df: pd.DataFrame, data_dir: str, batch_size: int = 8) -> np.ndarray:
    """[N,4] probabilities, in df order."""
    rows = [row for _, row in df.iterrows()]
    probs = []
    for start in range(0, len(rows), batch_size):
        probs.append(predict_batch(model, processor, rows[start : start + batch_size], data_dir))
        done = start + batch_size
        if done // 200 > start // 200:
            print(f"{min(done, len(rows))}/{len(rows)}", flush=True)
    return np.concatenate(probs)


def infer(cfg: Config):
    df = subsample(load_split(cfg, cfg.split), cfg.max_infer_samples, cfg.seed)

    adapter = cfg.adapter_dir or cfg.output_dir
    model, processor = load_model(cfg.model_id if adapter == "none" else adapter, cfg.load_in_4bit)
    FastVisionModel.for_inference(model)

    probs = predict_probs(model, processor, df, cfg.data_dir, cfg.infer_batch_size)
    pred = [CHOICES[k] for k in probs.argmax(1)]

    if "answer" in df.columns:
        acc = (df["answer"].astype(str).str.strip().str.lower() == pd.Series(pred)).mean()
        print(f"{cfg.split} accuracy: {acc:.4f} (n={len(df)})")

    os.makedirs(cfg.output_dir, exist_ok=True)
    ids = df["id"] if "id" in df.columns else df.index
    sub_path = os.path.join(cfg.output_dir, f"submission_{cfg.split}.csv")
    pd.DataFrame({"id": ids, "answer": pred}).to_csv(sub_path, index=False)
    # 앙상블용 확률 (README 규약: 행 = csv 순서)
    np.savez(os.path.join(cfg.output_dir, f"{cfg.split}_probs.npz"), avg=probs, ids=np.asarray(ids))
    print("Saved:", sub_path)


def main():
    infer(parse_config())


if __name__ == "__main__":
    main()
