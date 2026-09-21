"""Find the largest inference batch size that fits: grows the batch until CUDA OOM.

    uv run stv-bench --config configs/sample.toml --adapter-dir none

Each batch size is measured on the worst case (the largest images of the split, which produce the most
visual tokens) for peak VRAM, and on random batches for throughput.
"""

import time

import torch
from PIL import Image

from .config import Config, parse_config
from .data import load_split
from .inference import load_for_inference, predict_batch
from .model import set_image_budget

BATCH_SIZES = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
MAX_BATCH = BATCH_SIZES[-1]
TIMED_BATCHES = 2
# WSL/Windows 드라이버는 VRAM이 차면 OOM 대신 시스템 메모리로 넘겨 수십 배 느려진 채 멈춘 것처럼 보임.
# 할당 상한을 걸어 그 전에 OOM이 나게 함
MEMORY_FRACTION = 0.92


def bench(cfg: Config):
    df = load_split(cfg, cfg.split)
    # PIL은 헤더만 읽으므로 전체 스캔도 몇 초면 끝남
    pixels = [Image.open(f"{cfg.data_dir}/{p}").size for p in df["path"]]
    df = df.assign(pixels=[w * h for w, h in pixels])
    worst = [row for _, row in df.sort_values("pixels", ascending=False).head(MAX_BATCH).iterrows()]
    random_rows = [row for _, row in df.sample(n=min(len(df), MAX_BATCH * TIMED_BATCHES), random_state=cfg.seed).iterrows()]

    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
    model, processor = load_for_inference(cfg)
    set_image_budget(processor, cfg.infer_max_pixels, cfg.min_pixels)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"model loaded: {torch.cuda.memory_allocated() / 2**30:.2f} GiB / {total:.1f} GiB", flush=True)
    print(f"{'batch':>6} {'peak GiB (worst case)':>22} {'img/s (random)':>15}", flush=True)

    predict_batch(model, processor, worst[:1], cfg)  # warmup
    best, last_ok = None, None
    for batch_size in BATCH_SIZES:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            predict_batch(model, processor, worst[:batch_size], cfg)
            peak = torch.cuda.max_memory_allocated() / 2**30

            torch.cuda.synchronize()
            start = time.perf_counter()
            for k in range(TIMED_BATCHES):
                predict_batch(model, processor, random_rows[k * batch_size : (k + 1) * batch_size], cfg)
            torch.cuda.synchronize()
            speed = TIMED_BATCHES * batch_size / (time.perf_counter() - start)
        except torch.OutOfMemoryError:
            print(f"{batch_size:>6} {'OOM':>22}", flush=True)
            break
        print(f"{batch_size:>6} {peak:>22.2f} {speed:>15.2f}", flush=True)
        last_ok = batch_size
        if best is None or speed > best[1]:
            best = (batch_size, speed)

    if best:
        print(f"\nfastest: --infer-batch-size {best[0]} ({best[1]:.2f} img/s, 이미지 디코딩·전처리 시간 포함)")
        print(f"largest that fits: {last_ok} (VRAM {MEMORY_FRACTION:.0%} 상한, 가장 큰 이미지들로 잰 최악의 경우)")


def main():
    bench(parse_config())


if __name__ == "__main__":
    main()
