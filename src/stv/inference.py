from unsloth import FastVisionModel  # isort: skip

import os

import numpy as np
import pandas as pd
import torch

from . import tracking
from .config import Config, parse_config
from .data import CHOICES, build_messages, is_numeric_options, load_image, load_split, subsample
from .model import apply_init_adapter_config, add_lora, load_adapter_weights, load_model, set_image_budget

TTA_SHIFTS = [0, 2, 1, 3]  # 보기 순환 이동량 (2 = 순서를 반 바퀴 돌림)


def choice_ids(processor) -> list[int]:
    ids = [processor.tokenizer.convert_tokens_to_ids(c) for c in CHOICES]
    assert all(processor.tokenizer.decode([t]) == c for t, c in zip(ids, CHOICES))
    return ids


@torch.inference_mode()
def predict_batch(model, processor, rows: list, cfg: Config, n_tta: int = 1) -> np.ndarray:
    """[B,4] 원래 보기 a~d 순서의 확률. 답을 쓰기 직전 위치에서 a/b/c/d 토큰 로짓만 비교 (파싱 실패 없음, forward 1회).
    n_tta>1이면 보기를 순환 이동한 프롬프트로도 예측해 원래 자리로 되돌려 평균 (위치 편향 제거)."""
    ids = choice_ids(processor)
    images = [load_image(cfg.data_dir, r["path"]) for r in rows]
    processor.tokenizer.padding_side = "left"  # 모든 샘플의 마지막 위치(-1)가 답 글자 자리
    acc = np.zeros((len(rows), 4))
    for s in TTA_SHIFTS[:n_tta]:
        order = [(j + s) % 4 for j in range(4)]  # order[j] = j번째 자리에 보여줄 원래 보기
        texts = [
            processor.apply_chat_template(
                build_messages(img, str(r["question"]), [str(r[CHOICES[o]]) for o in order], variant=cfg.prompt_variant),
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for r, img in zip(rows, images)
        ]
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
        # logits_to_keep=1: 마지막 위치만 vocab으로 투영 (전체 [B,T,V] logits가 배치 추론 VRAM의 대부분)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**inputs, logits_to_keep=1).logits[:, -1, ids]
        acc[:, order] += torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return acc / n_tta


def predict_probs(model, processor, df: pd.DataFrame, cfg: Config, n_tta: int = 1, max_pixels: int = 0,
                  cache_path: str = "", cache_key: str = "", desc: str = "predict") -> np.ndarray:
    """[N,4] probabilities, in df order. cache_path를 주면 중간 결과를 저장 → 끊겨도 같은 설정이면 이어서 실행."""
    max_pixels = max_pixels or cfg.infer_max_pixels
    probs, done = np.zeros((len(df), 4)), np.zeros(len(df), dtype=bool)
    key = f"{cache_key}|prompt{cfg.prompt_variant}|tta{n_tta}|px{max_pixels}-{cfg.min_pixels}|n{len(df)}|{df['id'].iloc[0]}|{df['id'].iloc[-1]}"
    if cache_path and os.path.exists(cache_path):
        with np.load(cache_path) as saved:
            if "key" in saved.files and str(saved["key"]) == key:
                probs, done = saved["probs"], saved["done"]
        print(f"[{desc}] 저장된 진행분 {done.sum()}/{len(df)} 이어서 실행", flush=True)

    old_size, old_side, was_training = dict(processor.image_processor.size), processor.tokenizer.padding_side, model.training
    set_image_budget(processor, max_pixels, cfg.min_pixels)
    model.eval()
    try:
        rows = [row for _, row in df.iterrows()]
        todo = np.where(~done)[0]
        for n_batch, start in enumerate(range(0, len(todo), cfg.infer_batch_size)):
            idx = todo[start : start + cfg.infer_batch_size]
            probs[idx] = predict_batch(model, processor, [rows[i] for i in idx], cfg, n_tta)
            done[idx] = True
            if n_batch % 50 == 49:
                print(f"[{desc}] {done.sum()}/{len(df)}", flush=True)
                if cache_path:
                    np.savez(cache_path, probs=probs, done=done, key=np.array(key))
    finally:  # 중간에 멈춰도 학습용 설정으로 되돌림
        processor.image_processor.size = old_size
        processor.tokenizer.padding_side = old_side
        if was_training:
            model.train()
    return probs


def evaluate(df: pd.DataFrame, probs: np.ndarray, name: str = "valid", show_wrong: int = 5) -> float:
    """정확도 + 숫자형/텍스트형 보기 분리. dev는 동률 문항(answer NaN)을 제외하고 채점."""
    has_label = df["answer"].notna().values
    df, probs = df[has_label], probs[has_label]
    gold = df["answer"].str.strip().str.lower().map(CHOICES.index).values
    pred = probs.argmax(1)
    correct = pred == gold
    numeric = is_numeric_options(df)
    print(
        f"[{name}] 정확도 {correct.mean():.4f} ({correct.sum()}/{len(df)})"
        f" | 숫자형 보기 {correct[numeric].mean():.3f} (n={numeric.sum()})"
        f" | 텍스트형 보기 {correct[~numeric].mean():.3f} (n={(~numeric).sum()})"
    )
    print("   예측 분포:", dict(zip(CHOICES, np.bincount(pred, minlength=4).tolist())))
    for (_, r), pr in list(zip(df[~correct].iterrows(), pred[~correct]))[:show_wrong]:
        print(f"   ✗ {r['id']} | {str(r['question'])[:40]} | 예측 {CHOICES[pr]}={str(r[CHOICES[pr]])[:15]} / 정답 {r['answer']}={str(r[r['answer']])[:15]}")
    return float(correct.mean())


def save_outputs(df: pd.DataFrame, probs: np.ndarray, cfg: Config, split: str, tag: str = "") -> str:
    """submission_{split}{tag}.csv + {split}{tag}_probs.npz (팀 규약: avg [N,4], ids, 행 = csv 순서)"""
    os.makedirs(cfg.output_dir, exist_ok=True)
    pred = [CHOICES[k] for k in probs.argmax(1)]
    sub_path = os.path.join(cfg.output_dir, f"submission_{split}{tag}.csv")
    pd.DataFrame({"id": df["id"].values, "answer": pred}).to_csv(sub_path, index=False)
    np.savez(os.path.join(cfg.output_dir, f"{split}{tag}_probs.npz"), avg=probs, ids=df["id"].values)
    return sub_path


def check_submission(df: pd.DataFrame, probs: np.ndarray, cfg: Config):
    sample = pd.read_csv(os.path.join(cfg.data_dir, "sample_submission.csv"))
    if len(df) == len(sample):
        assert (df["id"].values == sample["id"].values).all(), "id 순서/개수가 sample_submission과 다름"
    dist = np.bincount(probs.argmax(1), minlength=4) / len(df)
    print("예측 분포:", dict(zip(CHOICES, dist.round(3).tolist())))
    assert dist.max() < 0.5, "한 글자로 예측이 쏠림 → 모델/프롬프트 점검 필요"


def load_for_inference(cfg: Config):
    """adapter_dir(기본 output_dir)의 학습된 어댑터, 없으면 init_adapter, "none"이면 원본 모델."""
    adapter = cfg.adapter_dir or cfg.output_dir
    init_dir = apply_init_adapter_config(cfg)
    if adapter != "none" and os.path.exists(os.path.join(adapter, "adapter_model.safetensors")):
        model, processor = load_model(adapter, cfg)
        print("어댑터 로드:", adapter)
    elif init_dir:
        model, processor = load_model(cfg.model_id, cfg)
        model = add_lora(model, cfg)
        load_adapter_weights(model, init_dir)
        print("시작 어댑터로 추론:", cfg.init_adapter)
    else:
        assert adapter == "none", f"{adapter}에 학습된 어댑터가 없습니다. 먼저 학습하거나 --adapter-dir none(zero-shot)"
        model, processor = load_model(cfg.model_id, cfg)
        print("zero-shot:", cfg.model_id)
    FastVisionModel.for_inference(model)
    return model, processor


def infer(cfg: Config):
    df = subsample(load_split(cfg, cfg.split), cfg.max_infer_samples, cfg.seed)
    # WSL/Windows 드라이버는 VRAM이 차면 OOM 대신 시스템 메모리로 넘겨 수십 배 느려짐 → 상한을 걸어 바로 OOM이 나게 함
    torch.cuda.set_per_process_memory_fraction(0.95)
    model, processor = load_for_inference(cfg)

    os.makedirs(cfg.output_dir, exist_ok=True)
    cache_path = os.path.join(cfg.output_dir, f"cache_{cfg.split}.npz")
    adapter = cfg.adapter_dir or cfg.output_dir
    weights = os.path.join(adapter, "adapter_model.safetensors")
    cache_key = f"{adapter}|{os.path.getmtime(weights) if os.path.exists(weights) else cfg.init_adapter or cfg.model_id}"
    probs = predict_probs(model, processor, df, cfg, cfg.n_tta, cache_path=cache_path, cache_key=cache_key, desc=cfg.split)
    if os.path.exists(cache_path):
        os.remove(cache_path)

    if "answer" in df.columns:
        acc = evaluate(df, probs, name=f"{cfg.split} TTA{cfg.n_tta}")
        tracking.init(cfg, "infer")
        tracking.summary({f"{cfg.split}/acc_tta{cfg.n_tta}": acc, f"{cfg.split}/n": len(df)})
        tracking.finish()
    else:
        check_submission(df, probs, cfg)
    print("Saved:", save_outputs(df, probs, cfg, cfg.split))


def main():
    infer(parse_config())


if __name__ == "__main__":
    main()
