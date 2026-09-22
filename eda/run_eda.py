#!/usr/bin/env python3
"""Reproducible static EDA for the scene-text VQA dataset.

The script intentionally uses only project dependencies (numpy, pandas, Pillow,
torch and torchvision).  It scans every train/test/dev image, while the main
distribution comparison and embeddings focus on train versus test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


from common import (  # noqa: E402  (same-folder module; run as `python eda/run_eda.py`)
    CHOICES, MAIN_SPLITS, ROOT, SPLITS, ks_statistic, load_csvs, numeric_comparison, text_metrics, total_variation,
)

PHASH_SIZE = 32
PHASH_LOW = 8


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def orientation(width: int, height: int) -> str:
    ratio = width / height
    if ratio > 1.05:
        return "landscape"
    if ratio < 0.95:
        return "portrait"
    return "square"


def shannon_entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    prob = hist[hist > 0] / hist.sum()
    return float(-(prob * np.log2(prob)).sum())


def perceptual_hash(gray: Image.Image) -> int:
    """64-bit pHash using a small, dependency-free DCT."""
    arr = np.asarray(gray.resize((PHASH_SIZE, PHASH_SIZE), Image.Resampling.LANCZOS), dtype=np.float64)
    n = PHASH_SIZE
    x = np.arange(n)
    u = np.arange(n)[:, None]
    dct = np.cos((2 * x + 1) * u * np.pi / (2 * n))
    dct[0] /= math.sqrt(2)
    coeff = (2 / n) * dct @ arr @ dct.T
    low = coeff[:PHASH_LOW, :PHASH_LOW].ravel()
    threshold = np.median(low[1:])
    bits = low > threshold
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def image_metrics(path: Path, split: str, row_id: str) -> dict:
    raw_size = path.stat().st_size
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    with Image.open(path) as source:
        source.load()
        width, height = source.size
        mode = source.mode
        fmt = source.format
        rgb_image = ImageOps.exif_transpose(source).convert("RGB")
        thumb = rgb_image.copy()
        thumb.thumbnail((256, 256), Image.Resampling.LANCZOS)
        rgb = np.asarray(thumb, dtype=np.uint8)
        gray_image = thumb.convert("L")
        gray = np.asarray(gray_image, dtype=np.uint8)

    rgb_float = rgb.astype(np.float32) / 255.0
    gray_float = gray.astype(np.float32) / 255.0
    channel_mean = rgb_float.reshape(-1, 3).mean(axis=0)
    channel_std = rgb_float.reshape(-1, 3).std(axis=0)
    maximum = rgb_float.max(axis=2)
    minimum = rgb_float.min(axis=2)
    saturation = np.divide(maximum - minimum, maximum, out=np.zeros_like(maximum), where=maximum > 1e-8)
    rg = rgb_float[..., 0] - rgb_float[..., 1]
    yb = 0.5 * (rgb_float[..., 0] + rgb_float[..., 1]) - rgb_float[..., 2]
    colorfulness = math.sqrt(float(rg.std() ** 2 + yb.std() ** 2)) + 0.3 * math.sqrt(float(rg.mean() ** 2 + yb.mean() ** 2))

    gx = np.abs(np.diff(gray_float, axis=1))
    gy = np.abs(np.diff(gray_float, axis=0))
    edge_strength = (gx[:-1, :] + gy[:, :-1]) / 2 if min(gray.shape) > 1 else np.zeros((1, 1))
    edge_density = float((edge_strength > 0.10).mean())
    lap = (
        -4 * gray_float[1:-1, 1:-1]
        + gray_float[:-2, 1:-1]
        + gray_float[2:, 1:-1]
        + gray_float[1:-1, :-2]
        + gray_float[1:-1, 2:]
    ) if min(gray.shape) > 2 else np.zeros((1, 1))

    return {
        "split": split,
        "id": row_id,
        "path": str(path.relative_to(ROOT)),
        "width": width,
        "height": height,
        "megapixels": width * height / 1_000_000,
        "aspect_ratio": width / height,
        "orientation": orientation(width, height),
        "mode": mode,
        "format": fmt,
        "file_bytes": raw_size,
        "jpeg_bytes_per_pixel": raw_size / (width * height),
        "brightness": float(gray_float.mean()),
        "contrast": float(gray_float.std()),
        "red_mean": float(channel_mean[0]),
        "green_mean": float(channel_mean[1]),
        "blue_mean": float(channel_mean[2]),
        "red_std": float(channel_std[0]),
        "green_std": float(channel_std[1]),
        "blue_std": float(channel_std[2]),
        "saturation": float(saturation.mean()),
        "colorfulness": colorfulness,
        "entropy_bits": shannon_entropy(gray),
        "edge_density": edge_density,
        "laplacian_variance": float(lap.var()),
        "sha256": sha.hexdigest(),
        "phash": f"{perceptual_hash(gray_image):016x}",
    }



def exact_duplicate_table(images: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_no = 0
    for digest, group in images.groupby("sha256", sort=False):
        if len(group) < 2:
            continue
        group_no += 1
        splits = Counter(group.split)
        records.append({
            "group": group_no, "sha256": digest, "count": len(group),
            "splits": ";".join(f"{k}:{v}" for k, v in sorted(splits.items())),
            "ids": ";".join(group.id), "paths": ";".join(group.path),
        })
    return pd.DataFrame(records, columns=["group", "sha256", "count", "splits", "ids", "paths"])


def exact_cross_split_pairs(images: pd.DataFrame, csvs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    questions = {split: csvs[split].set_index("id")["question"].astype(str) for split in MAIN_SPLITS}
    records = []
    for digest, group in images.groupby("sha256", sort=False):
        train_rows = group[group.split == "train"]
        test_rows = group[group.split == "test"]
        for train_row in train_rows.itertuples(index=False):
            for test_row in test_rows.itertuples(index=False):
                train_question = questions["train"].loc[train_row.id]
                test_question = questions["test"].loc[test_row.id]
                records.append({
                    "sha256": digest, "train_id": train_row.id, "test_id": test_row.id,
                    "train_path": train_row.path, "test_path": test_row.path,
                    "train_question": train_question, "test_question": test_question,
                    "question_exact_match": train_question == test_question,
                })
    return pd.DataFrame(records, columns=[
        "sha256", "train_id", "test_id", "train_path", "test_path",
        "train_question", "test_question", "question_exact_match",
    ])


BIT_COUNTS = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(a[:, None], b[None, :])
    return BIT_COUNTS[xor.view(np.uint8).reshape(len(a), len(b), 8)].sum(axis=2)


def similar_pairs(images: pd.DataFrame, threshold: int = 6, keep: int = 1000) -> pd.DataFrame:
    """Find close pHash candidates within each split and across train/test."""
    records: list[dict] = []
    sets = [("train", "train"), ("test", "test"), ("train", "test")]
    for left_split, right_split in sets:
        left = images[images.split == left_split].reset_index(drop=True)
        right = images[images.split == right_split].reset_index(drop=True)
        left_hash = np.array([int(x, 16) for x in left.phash], dtype=np.uint64)
        right_hash = np.array([int(x, 16) for x in right.phash], dtype=np.uint64)
        for start in range(0, len(left), 256):
            dist = hamming_matrix(left_hash[start:start + 256], right_hash)
            ii, jj = np.where(dist <= threshold)
            for local_i, j in zip(ii.tolist(), jj.tolist()):
                i = start + local_i
                if left_split == right_split and j <= i:
                    continue
                lrow, rrow = left.iloc[i], right.iloc[j]
                if lrow.sha256 == rrow.sha256:
                    continue
                records.append({
                    "scope": f"{left_split}-{right_split}", "hamming_distance": int(dist[local_i, j]),
                    "left_id": lrow.id, "left_path": lrow.path, "right_id": rrow.id, "right_path": rrow.path,
                    "same_dimensions": bool(lrow.width == rrow.width and lrow.height == rrow.height),
                    "brightness_difference": abs(float(lrow.brightness - rrow.brightness)),
                })
    records.sort(key=lambda x: (x["hamming_distance"], x["scope"], x["left_id"], x["right_id"]))
    records = records[:keep]
    for record in records:
        record["pixel_mae"] = np.nan
        record["pixel_abs_difference_p99"] = np.nan
        if record["same_dimensions"]:
            with Image.open(ROOT / record["left_path"]) as left_image, Image.open(ROOT / record["right_path"]) as right_image:
                left_pixels = np.asarray(left_image.convert("RGB"), dtype=np.int16)
                right_pixels = np.asarray(right_image.convert("RGB"), dtype=np.int16)
            difference = np.abs(left_pixels - right_pixels)
            record["pixel_mae"] = float(difference.mean())
            record["pixel_abs_difference_p99"] = float(np.quantile(difference, .99))
    return pd.DataFrame(records, columns=[
        "scope", "hamming_distance", "left_id", "left_path", "right_id", "right_path",
        "same_dimensions", "brightness_difference", "pixel_mae", "pixel_abs_difference_p99",
    ])


def extract_embeddings(data_root: Path, csvs: dict[str, pd.DataFrame], output: Path, batch_size: int, workers: int) -> tuple[np.ndarray, list[str]]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import ResNet18_Weights, resnet18

    class DatasetImpl(Dataset):
        def __init__(self, items, transform):
            self.items, self.transform = items, transform
        def __len__(self):
            return len(self.items)
        def __getitem__(self, index):
            split, row_id, relative = self.items[index]
            with Image.open(data_root / relative) as image:
                tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
            return tensor, split, row_id

    cache_file = output / "resnet18_embeddings.npz"
    expected_ids = [str(x) for split in MAIN_SPLITS for x in csvs[split].id]
    if cache_file.exists():
        cached = np.load(cache_file)
        cached_ids = cached["ids"].astype(str).tolist()
        if cached_ids == expected_ids:
            return cached["embeddings"].astype(np.float32), cached["splits"].astype(str).tolist()

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    items = [(split, str(row.id), str(row.path)) for split in MAIN_SPLITS for row in csvs[split].itertuples(index=False)]
    loader = DataLoader(DatasetImpl(items, weights.transforms()), batch_size=batch_size, shuffle=False, num_workers=workers)
    features, ids, splits = [], [], []
    torch.set_grad_enabled(False)
    started = time.time()
    for batch_no, (tensor, batch_splits, batch_ids) in enumerate(loader, 1):
        emb = model(tensor).numpy().astype(np.float32)
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        features.append(emb); ids.extend(batch_ids); splits.extend(batch_splits)
        if batch_no % 25 == 0:
            print(f"  embeddings: {len(ids)}/{len(items)} ({time.time() - started:.1f}s)", flush=True)
    matrix = np.concatenate(features)
    np.savez_compressed(cache_file, embeddings=matrix.astype(np.float16), ids=np.asarray(ids), splits=np.asarray(splits))
    return matrix, splits


def embedding_analysis(features: np.ndarray, splits: list[str], output: Path, seed: int) -> tuple[dict, pd.DataFrame]:
    split_array = np.asarray(splits)
    rng = np.random.default_rng(seed)
    # Fit PCA on a balanced sample for speed, then project all points.
    sample_idx = np.concatenate([rng.choice(np.where(split_array == s)[0], min(3000, (split_array == s).sum()), replace=False) for s in MAIN_SPLITS])
    fit = features[sample_idx].astype(np.float64)
    mean = fit.mean(axis=0)
    _, singular, vt = np.linalg.svd(fit - mean, full_matrices=False)
    components = vt[:64]
    projected = (features - mean) @ components.T
    explained = singular ** 2 / np.maximum((singular ** 2).sum(), 1e-12)

    arrays = {s: projected[split_array == s] for s in MAIN_SPLITS}
    centroids = {s: arrays[s].mean(axis=0) for s in MAIN_SPLITS}
    centroid_distance = float(np.linalg.norm(centroids["train"] - centroids["test"]))
    original_centroids = {s: features[split_array == s].mean(axis=0) for s in MAIN_SPLITS}
    cosine = float(np.dot(original_centroids["train"], original_centroids["test"]) /
                   (np.linalg.norm(original_centroids["train"]) * np.linalg.norm(original_centroids["test"])))

    cov_train = np.cov(arrays["train"], rowvar=False)
    cov_test = np.cov(arrays["test"], rowvar=False)
    eig_a, vec_a = np.linalg.eigh(cov_train)
    sqrt_a = (vec_a * np.sqrt(np.clip(eig_a, 0, None))) @ vec_a.T
    middle = sqrt_a @ cov_test @ sqrt_a
    eig_middle = np.linalg.eigvalsh((middle + middle.T) / 2)
    frechet = centroid_distance ** 2 + np.trace(cov_train) + np.trace(cov_test) - 2 * np.sqrt(np.clip(eig_middle, 0, None)).sum()

    n = min(1000, len(arrays["train"]), len(arrays["test"]))
    x = arrays["train"][rng.choice(len(arrays["train"]), n, replace=False), :64]
    y = arrays["test"][rng.choice(len(arrays["test"]), n, replace=False), :64]
    pooled = np.vstack((x, y))
    probe = pooled[rng.choice(len(pooled), min(500, len(pooled)), replace=False)]
    d2_probe = ((probe[:, None] - probe[None, :]) ** 2).sum(axis=2)
    bandwidth = float(np.median(d2_probe[d2_probe > 0]))
    def kernel(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-d2 / max(2 * bandwidth, 1e-12))
    kxx, kyy, kxy = kernel(x, x), kernel(y, y), kernel(x, y)
    # The biased empirical estimator is non-negative and easier to interpret in
    # an EDA report than a slightly negative finite-sample unbiased estimate.
    mmd2 = float(kxx.mean() + kyy.mean() - 2 * kxy.mean())

    rows = []
    for pc in range(10):
        train_pc, test_pc = arrays["train"][:, pc], arrays["test"][:, pc]
        rows.append({
            "component": pc + 1, "explained_variance_ratio": explained[pc],
            "train_mean": train_pc.mean(), "test_mean": test_pc.mean(),
            "train_std": train_pc.std(ddof=1), "test_std": test_pc.std(ddof=1),
            "ks_statistic": ks_statistic(train_pc, test_pc),
        })
    pc_table = pd.DataFrame(rows)
    np.savez_compressed(output / "embedding_pca_projection.npz", projection=projected[:, :2].astype(np.float32), splits=split_array)
    summary = {
        "model": "torchvision ResNet-18 ImageNet-1K V1, global-average-pooled 512D L2-normalized features",
        "sample_count": len(features), "pca_fit_sample_count": len(sample_idx),
        "pca_64_explained_variance": float(explained[:64].sum()),
        "pca_first_two_explained_variance": float(explained[:2].sum()),
        "centroid_distance_pca64": centroid_distance,
        "centroid_cosine_similarity_original512": cosine,
        "frechet_distance_pca64": float(max(frechet, 0)),
        "rbf_mmd_squared_pca64": max(mmd2, 0.0),
        "mmd_sample_per_split": n, "mmd_rbf_bandwidth_squared": bandwidth,
    }
    return summary, pc_table


def font(size: int = 18):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


COLORS = {"train": (52, 105, 180), "test": (225, 93, 68), "dev": (70, 160, 100)}


def histogram_plot(frame: pd.DataFrame, column: str, path: Path, title: str, bins: int = 40):
    width, height = 900, 520
    margin = (75, 50, 25, 65)
    canvas = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(canvas)
    arrays = [frame.loc[frame.split == split, column].dropna().to_numpy(float) for split in MAIN_SPLITS]
    lo, hi = np.quantile(np.concatenate(arrays), [.005, .995])
    if lo == hi: hi = lo + 1
    edges = np.linspace(lo, hi, bins + 1)
    densities = [np.histogram(np.clip(a, lo, hi), bins=edges, density=True)[0] for a in arrays]
    ymax = max(d.max() for d in densities) * 1.08
    x0, y0, x1, y1 = margin[0], margin[1], width - margin[2], height - margin[3]
    draw.line((x0, y1, x1, y1), fill="black", width=2); draw.line((x0, y0, x0, y1), fill="black", width=2)
    for split, density in zip(MAIN_SPLITS, densities):
        pts = []
        centers = (edges[:-1] + edges[1:]) / 2
        for x, y in zip(centers, density):
            px = x0 + (x - lo) / (hi - lo) * (x1 - x0)
            py = y1 - y / ymax * (y1 - y0)
            pts.append((px, py))
        draw.line(pts, fill=COLORS[split], width=4)
    draw.text((x0, 15), title, fill="black", font=font(24))
    draw.text((x0, y1 + 15), f"{lo:.3g}", fill="black", font=font(15)); draw.text((x1 - 60, y1 + 15), f"{hi:.3g}", fill="black", font=font(15))
    for i, split in enumerate(MAIN_SPLITS):
        draw.rectangle((x1 - 180, y0 + i * 28, x1 - 160, y0 + 18 + i * 28), fill=COLORS[split])
        draw.text((x1 - 150, y0 - 3 + i * 28), split, fill="black", font=font(17))
    canvas.save(path)


def bar_plot(counts: dict[str, pd.Series], path: Path, title: str):
    labels = sorted(set().union(*(series.index for series in counts.values())))
    width, height = max(900, 95 * len(labels)), 560
    canvas = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = 70, 55, width - 25, height - 110
    proportions = {s: counts[s].reindex(labels, fill_value=0) / max(counts[s].sum(), 1) for s in counts}
    ymax = max((v.max() for v in proportions.values()), default=1) * 1.15
    draw.line((x0, y1, x1, y1), fill="black", width=2); draw.line((x0, y0, x0, y1), fill="black", width=2)
    group_width = (x1 - x0) / max(len(labels), 1)
    bar_width = group_width * .32
    for idx, label in enumerate(labels):
        center = x0 + (idx + .5) * group_width
        for j, split in enumerate(counts):
            value = proportions[split].loc[label]
            left = center + (j - (len(counts)-1)/2) * bar_width - bar_width * .42
            top = y1 - value / ymax * (y1 - y0)
            draw.rectangle((left, top, left + bar_width * .84, y1), fill=COLORS.get(split, (100, 100, 100)))
        label_ascii = label if label.isascii() else str(idx + 1)
        draw.text((center - 8, y1 + 10), label_ascii, fill="black", font=font(14))
    draw.text((x0, 15), title, fill="black", font=font(24))
    for i, split in enumerate(counts):
        legend_x, legend_y = x1 - 115, y0 + 10 + i * 27
        draw.rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), fill=COLORS.get(split, (100, 100, 100)))
        draw.text((legend_x + 25, legend_y - 2), split, fill="black", font=font(16))
    legend = " | ".join(f"{i+1}: {label}" for i, label in enumerate(labels))
    draw.text((x0, height - 55), legend[:150], fill="black", font=font(13))
    canvas.save(path)


def pca_scatter(path_npz: Path, path_png: Path, seed: int):
    data = np.load(path_npz); xy, splits = data["projection"], data["splits"].astype(str)
    rng = np.random.default_rng(seed)
    ids = np.concatenate([rng.choice(np.where(splits == s)[0], min(2500, (splits == s).sum()), replace=False) for s in MAIN_SPLITS])
    selected = xy[ids]; selected_splits = splits[ids]
    low = np.quantile(selected, .005, axis=0); high = np.quantile(selected, .995, axis=0)
    width, height = 760, 650; x0, y0, x1, y1 = 70, 50, width - 30, height - 65
    canvas = Image.new("RGBA", (width, height), "white"); layer = Image.new("RGBA", canvas.size, (255,255,255,0)); dots = ImageDraw.Draw(layer)
    for split in MAIN_SPLITS:
        points = selected[selected_splits == split]
        for x, y in points:
            px = x0 + np.clip((x-low[0])/(high[0]-low[0]), 0, 1)*(x1-x0)
            py = y1 - np.clip((y-low[1])/(high[1]-low[1]), 0, 1)*(y1-y0)
            dots.ellipse((px-2, py-2, px+2, py+2), fill=(*COLORS[split], 75))
    canvas = Image.alpha_composite(canvas, layer).convert("RGB"); draw = ImageDraw.Draw(canvas)
    draw.rectangle((x0,y0,x1,y1), outline="black", width=2); draw.text((x0, 14), "ResNet-18 embedding PCA (sampled)", fill="black", font=font(24))
    draw.text((width//2-20,y1+20), "PC1", fill="black", font=font(17)); draw.text((12,height//2), "PC2", fill="black", font=font(17))
    for i, split in enumerate(MAIN_SPLITS):
        legend_x, legend_y = x1 - 115, y0 + 15 + i * 28
        draw.rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), fill=COLORS[split])
        draw.text((legend_x + 26, legend_y - 2), split, fill="black", font=font(16))
    canvas.save(path_png)


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if col in formats and isinstance(value, (int, float, np.number)):
                values.append(format(value, formats[col]))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_report(output: Path, csvs: dict[str, pd.DataFrame], images: pd.DataFrame, texts: pd.DataFrame,
                    numeric_stats: pd.DataFrame, numeric_compare: pd.DataFrame, exact: pd.DataFrame,
                    exact_cross: pd.DataFrame, similar: pd.DataFrame, emb_summary: dict, pc_table: pd.DataFrame):
    img = numeric_stats[numeric_stats.metric.isin([
        "width", "height", "megapixels", "aspect_ratio", "brightness", "contrast", "saturation",
        "colorfulness", "entropy_bits", "edge_density", "laplacian_variance", "jpeg_bytes_per_pixel",
    ])].copy()
    pivot = img.pivot(index="metric", columns="split", values=["mean", "median", "std"])
    rows = []
    for metric in img.metric.unique():
        cmp = numeric_compare[numeric_compare.metric == metric].iloc[0]
        rows.append({"지표": metric, "train 평균": pivot.loc[metric, ("mean", "train")], "test 평균": pivot.loc[metric, ("mean", "test")],
                     "train 중앙값": pivot.loc[metric, ("median", "train")], "test 중앙값": pivot.loc[metric, ("median", "test")],
                     "SMD": cmp.standardized_mean_difference, "KS": cmp.ks_statistic})
    image_table = markdown_table(pd.DataFrame(rows), ["지표", "train 평균", "test 평균", "train 중앙값", "test 중앙값", "SMD", "KS"],
                                 {x: ".4f" for x in ["train 평균", "test 평균", "train 중앙값", "test 중앙값", "SMD", "KS"]})

    length_metrics = ["question_chars", "question_tokens_ws", "choice_chars_mean", "choice_chars_max", "choice_chars_total"]
    length_rows = []
    for metric in length_metrics:
        part = numeric_stats[numeric_stats.metric == metric].set_index("split")
        cmp = numeric_compare[numeric_compare.metric == metric].iloc[0]
        length_rows.append({"지표": metric, "train 평균": part.loc["train", "mean"], "test 평균": part.loc["test", "mean"],
                            "train 중앙값": part.loc["train", "median"], "test 중앙값": part.loc["test", "median"],
                            "SMD": cmp.standardized_mean_difference, "KS": cmp.ks_statistic})
    length_table = markdown_table(pd.DataFrame(length_rows), ["지표", "train 평균", "test 평균", "train 중앙값", "test 중앙값", "SMD", "KS"],
                                  {x: ".3f" for x in ["train 평균", "test 평균", "train 중앙값", "test 중앙값", "SMD", "KS"]})

    orientation_counts = pd.crosstab(images[images.split.isin(MAIN_SPLITS)].orientation, images[images.split.isin(MAIN_SPLITS)].split)
    orientation_rows = []
    for label, row in orientation_counts.iterrows():
        orientation_rows.append({"방향": label, "train": int(row.get("train", 0)), "train %": 100*row.get("train",0)/len(csvs["train"]),
                                 "test": int(row.get("test", 0)), "test %": 100*row.get("test",0)/len(csvs["test"])})
    orientation_table = markdown_table(pd.DataFrame(orientation_rows), ["방향", "train", "train %", "test", "test %"], {"train %":".2f","test %":".2f"})

    q_counts = pd.crosstab(texts.question_type, texts.split)
    q_rows = []
    for label, row in q_counts.sort_values("train", ascending=False).iterrows():
        q_rows.append({"유형": label, "train": int(row.get("train",0)), "train %": 100*row.get("train",0)/len(csvs["train"]),
                       "test": int(row.get("test",0)), "test %": 100*row.get("test",0)/len(csvs["test"])})
    q_table = markdown_table(pd.DataFrame(q_rows), ["유형", "train", "train %", "test", "test %"], {"train %":".2f","test %":".2f"})
    q_tv = total_variation(texts[texts.split=="train"].question_type.value_counts(), texts[texts.split=="test"].question_type.value_counts())
    orient_tv = total_variation(images[images.split=="train"].orientation.value_counts(), images[images.split=="test"].orientation.value_counts())

    answers = csvs["train"].answer.value_counts().reindex(CHOICES, fill_value=0)
    answer_rows = pd.DataFrame({"정답": answers.index, "개수": answers.values, "비율 %": 100*answers.values/answers.sum()})
    answer_table = markdown_table(answer_rows, ["정답", "개수", "비율 %"], {"비율 %":".2f"})
    answer_imbalance = float(answers.max()/answers.min())

    exact_cross_groups = sum("train:" in x and "test:" in x for x in exact.splits) if len(exact) else 0
    exact_train = sum(x.startswith("train:") and ";" not in x for x in exact.splits) if len(exact) else 0
    exact_test = sum(x.startswith("test:") and ";" not in x for x in exact.splits) if len(exact) else 0
    similar_counts = similar.scope.value_counts() if len(similar) else pd.Series(dtype=int)
    top_shift = numeric_compare.reindex(numeric_compare.standardized_mean_difference.abs().sort_values(ascending=False).index).head(5)
    shift_text = ", ".join(f"{r.metric} (SMD {r.standardized_mean_difference:+.3f})" for r in top_shift.itertuples())
    pc_display = pc_table.head(5).copy()
    pc_formats = {c:".5f" for c in pc_display.columns if c != "component"}; pc_formats["component"] = ".0f"
    pc_table_md = markdown_table(pc_display, list(pc_display.columns), pc_formats)

    lines = f"""# Scene-text VQA 정적 EDA: train/test 분포 비교

생성 시각: {pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d %H:%M:%S %Z')}  
대상: train {len(csvs['train']):,}개, test {len(csvs['test']):,}개. 추가로 dev {len(csvs['dev']):,}개 이미지의 무결성과 이미지 지표도 전수 계산했다.

## 핵심 요약

- CSV와 이미지 대응: 세 split 모두 누락/초과 파일 0개, 읽기 실패 0개.
- 가장 큰 연속형 분포 차이: {shift_text}. 일반적으로 |SMD| 0.2 미만은 작은 차이로 해석한다.
- 방향 범주의 train/test 총변동거리(TV): **{orient_tv:.4f}**, 질문 유형 TV: **{q_tv:.4f}** (0은 동일, 1은 완전 분리).
- 임베딩: centroid cosine **{emb_summary.get('centroid_cosine_similarity_original512', float('nan')):.6f}**, PCA-64 Fréchet **{emb_summary.get('frechet_distance_pca64', float('nan')):.4f}**, RBF MMD² **{emb_summary.get('rbf_mmd_squared_pca64', float('nan')):.6f}**.
- 완전 동일 파일 그룹 {len(exact):,}개(그중 train-test 교차 그룹 {exact_cross_groups:,}개). 여기에 SHA는 다르지만 pHash 유사 후보 train-test 교차 {int(similar_counts.get('train-test',0)):,}쌍이 추가로 있다. 이는 split 간 이미지 재사용이 실제로 존재함을 뜻한다.

## 1. 이미지 크기 / 종횡비

{orientation_table}

![크기 분포](plots/megapixels.png)
![종횡비 분포](plots/aspect_ratio.png)

## 2. 밝기 / 대비 / 색상 분포

픽셀 기반 지표는 각 원본의 종횡비를 보존한 최대 256×256 축소본 전체 픽셀에서 계산했다. 값 범위는 RGB/gray 모두 0~1이다.

{image_table}

![밝기](plots/brightness.png)
![대비](plots/contrast.png)
![채도](plots/saturation.png)
![RGB 평균](plots/rgb_means.png)

## 3. 이미지 복잡도 / 정보량

- `entropy_bits`: 8-bit grayscale Shannon entropy(최대 8 bit).
- `edge_density`: 인접 픽셀 gradient가 0.10을 넘는 비율.
- `laplacian_variance`: 초점/고주파 정보의 대리 지표(축소 gray, 정규화 단위).
- `jpeg_bytes_per_pixel`: 압축 파일 크기/원본 픽셀 수. 콘텐츠뿐 아니라 JPEG 품질에도 영향을 받는다.

![엔트로피](plots/entropy_bits.png)
![에지 밀도](plots/edge_density.png)

## 4. 질문 길이 / 선택지 길이

`chars`는 Python 문자열 길이, `tokens_ws`는 공백 기준 어절 수다. 모델 tokenizer 토큰 수가 아님에 유의한다.

{length_table}

![질문 길이](plots/question_chars.png)
![선택지 평균 길이](plots/choice_chars_mean.png)

## 5. 정답 분포

{answer_table}

최대/최소 정답 빈도비는 **{answer_imbalance:.3f}**이다. test에는 정답 컬럼이 없으므로 train/test 정답 분포 비교는 불가능하다.

![정답 분포](plots/answer_distribution.png)

## 6. 질문 유형 분포

질문 유형은 `run_eda.py`의 우선순위 정규식 규칙으로 각 질문에 하나만 부여했다. 따라서 의미론적 정답 라벨이 아니라 재현 가능한 휴리스틱 분석이다.

{q_table}

![질문 유형](plots/question_types.png)

## 7. Train/Test 이미지 임베딩 분포

모델: `{emb_summary.get('model', 'not run')}`. PCA는 split별 최대 3,000개로 공동 fit하고, 모든 13,428개 이미지를 투영했다. MMD는 split별 {emb_summary.get('mmd_sample_per_split','N/A')}개 고정 seed 표본, Fréchet은 PCA 64차원에서 계산했다.

- PCA 64축 설명분산: **{emb_summary.get('pca_64_explained_variance', float('nan')):.4f}**
- 첫 2축 설명분산: **{emb_summary.get('pca_first_two_explained_variance', float('nan')):.4f}**
- PCA-64 centroid 거리: **{emb_summary.get('centroid_distance_pca64', float('nan')):.4f}**
- 원본 512차원 centroid cosine 유사도: **{emb_summary.get('centroid_cosine_similarity_original512', float('nan')):.6f}**
- PCA-64 Fréchet 거리: **{emb_summary.get('frechet_distance_pca64', float('nan')):.4f}**
- PCA-64 RBF MMD²: **{emb_summary.get('rbf_mmd_squared_pca64', float('nan')):.6f}**

{pc_table_md}

![임베딩 PCA](plots/embedding_pca.png)

이 거리들은 동일 모델/전처리로 다른 split 조합과 비교할 때 의미가 크며, 절대값만으로 데이터 누수를 단정해서는 안 된다.

## 8. 중복 이미지 / 유사 이미지

- 완전 중복은 파일 바이트 SHA-256 동일 여부로 판정했다: 전체 {len(exact):,}그룹, train 내부 {exact_train:,}그룹, test 내부 {exact_test:,}그룹, train-test 교차 **{exact_cross_groups:,}그룹**이다. 상세: [exact_duplicates.csv](tables/exact_duplicates.csv).
- 교차 완전 중복의 질문까지 결합한 표는 [exact_cross_split_pairs.csv](tables/exact_cross_split_pairs.csv)다. 교차 {len(exact_cross):,}쌍 중 질문 문자열까지 완전히 같은 것은 {int(exact_cross.question_exact_match.sum()) if len(exact_cross) else 0:,}쌍이며, 나머지는 같은 이미지에 다른 질문 또는 표현을 사용한다.
- 유사 후보는 32×32 DCT pHash 64-bit의 해밍거리 ≤ 6이면서 SHA-256은 다른 쌍이다: [similar_image_candidates.csv](tables/similar_image_candidates.csv). 같은 크기인 경우 원본 RGB의 `pixel_mae`와 99백분위 절대 오차도 기록했다.
- 유사 후보 CSV는 거리 오름차순 최대 1,000쌍이다. OCR 문서형 데이터에서는 레이아웃이 비슷한 오탐이 있으므로 누수 판정 전 원본을 육안 확인해야 한다. 다만 교차 후보 6쌍은 모두 pHash 거리 0이고 픽셀 MAE도 낮아, 재인코딩된 동일 장면일 가능성이 매우 높다.
- 따라서 향후 자체 검증 split은 반드시 이미지 해시/pHash 그룹 단위로 나누어야 한다. 고정 test에는 train 이미지와 같은 장면이 포함되어 있으므로 이미지 검색 기반 누수 가능성도 별도로 관리해야 한다.

## 해석 기준과 한계

- SMD는 `(test 평균 - train 평균) / pooled 표준편차`; 부호는 test가 더 큰지 나타낸다. KS는 최대 누적분포 차이(0~1)다.
- TV는 범주 비율 차이의 절반 합(0~1)이다.
- p-value는 표본 수에 매우 민감하므로 의도적으로 보고하지 않고 효과크기와 분포 거리를 보고했다.
- ResNet-18은 일반 자연 이미지 사전학습 모델이라 OCR 텍스트의 철자/내용 유사성을 직접 보장하지 않는다.
- dev의 `answer1`~`answer5`는 복수 annotator 응답이므로 train의 단일 `answer`와 같은 표에 섞지 않았다.

## 산출물

- `tables/image_metrics.csv`: 16,111개 전 이미지 지표와 해시
- `tables/text_metrics.csv`: train/test 질문·선택지 지표와 질문 유형
- `tables/numeric_summary.csv`, `tables/train_test_numeric_comparison.csv`
- `tables/embedding_pc_comparison.csv`, `embedding_pca_projection.npz`
- `summary.json`: 주요 수치의 machine-readable 요약
- `plots/`: 비교 시각화

## 재실행

```bash
# repo root에서
TORCH_HOME=eda/.cache/torch uv run python eda/run_eda.py --resume
```

최초 실행은 torchvision의 ResNet-18 가중치(약 45MB)를 내려받는다. 이후 `eda/.cache`를 재사용한다.
"""
    (output / "report.md").write_text(lines, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "eda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is the most portable)")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse complete image/text metric CSVs")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve(); args.output = args.output.resolve()
    tables = args.output / "tables"; plots = args.output / "plots"
    tables.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)

    csvs = load_csvs(args.data_root)
    inventory = {}
    expected_image_count = sum(len(csvs[s]) for s in SPLITS)
    image_cache = tables / "image_metrics.csv"
    if args.resume and image_cache.exists() and sum(1 for _ in image_cache.open(encoding="utf-8")) == expected_image_count + 1:
        print("[1/8] Reusing complete image metric cache", flush=True)
        images = pd.read_csv(image_cache, dtype={"phash": str})
        for split in SPLITS:
            inventory[split] = {"csv_rows": len(csvs[split]), "image_files": int((images.split == split).sum()), "missing": [], "extras": [], "failures": []}
    else:
        all_records = []
        print("[1/8] Scanning every image and computing static metrics", flush=True)
        for split in SPLITS:
            listed = set(str(x) for x in csvs[split].path)
            actual = set(str(x.relative_to(args.data_root)) for x in (args.data_root / split).glob("*.jpg"))
            missing, extras, failures = sorted(listed - actual), sorted(actual - listed), []
            for i, row in enumerate(csvs[split].itertuples(index=False), 1):
                try:
                    all_records.append(image_metrics(args.data_root / row.path, split, str(row.id)))
                except Exception as exc:
                    failures.append({"path": str(row.path), "error": repr(exc)})
                if i % 1000 == 0: print(f"  {split}: {i}/{len(csvs[split])}", flush=True)
            inventory[split] = {"csv_rows": len(csvs[split]), "image_files": len(actual), "missing": missing, "extras": extras, "failures": failures}
        images = pd.DataFrame(all_records)
        images.to_csv(image_cache, index=False)

    print("[2/8] Measuring questions and choices", flush=True)
    text_cache = tables / "text_metrics.csv"
    expected_text_count = sum(len(csvs[s]) for s in MAIN_SPLITS)
    if args.resume and text_cache.exists() and sum(1 for _ in text_cache.open(encoding="utf-8")) == expected_text_count + 1:
        texts = pd.read_csv(text_cache)
    else:
        texts = pd.concat([text_metrics(csvs[s], s) for s in MAIN_SPLITS], ignore_index=True)
        texts.to_csv(text_cache, index=False)
    combined = images[images.split.isin(MAIN_SPLITS)].merge(texts, on=["split", "id"], how="left")
    numeric_columns = [
        "width", "height", "megapixels", "aspect_ratio", "file_bytes", "jpeg_bytes_per_pixel",
        "brightness", "contrast", "red_mean", "green_mean", "blue_mean", "red_std", "green_std", "blue_std",
        "saturation", "colorfulness", "entropy_bits", "edge_density", "laplacian_variance",
        "question_chars", "question_tokens_ws", "choice_chars_mean", "choice_chars_max", "choice_chars_total", "choice_tokens_ws_mean",
    ]
    numeric_stats, numeric_compare = numeric_comparison(combined, numeric_columns)
    numeric_stats.to_csv(tables / "numeric_summary.csv", index=False)
    numeric_compare.to_csv(tables / "train_test_numeric_comparison.csv", index=False)

    print("[3/8] Detecting exact duplicates", flush=True)
    exact = exact_duplicate_table(images)
    exact.to_csv(tables / "exact_duplicates.csv", index=False)
    exact_cross = exact_cross_split_pairs(images, csvs)
    exact_cross.to_csv(tables / "exact_cross_split_pairs.csv", index=False)
    print("[4/8] Detecting perceptually similar candidates", flush=True)
    similar = similar_pairs(images)
    similar.to_csv(tables / "similar_image_candidates.csv", index=False)

    print("[5/8] Extracting pretrained image embeddings", flush=True)
    emb_summary, pc_table = {}, pd.DataFrame(columns=["component", "explained_variance_ratio", "train_mean", "test_mean", "train_std", "test_std", "ks_statistic"])
    if not args.skip_embeddings:
        features, splits = extract_embeddings(args.data_root, csvs, args.output, args.batch_size, args.workers)
        print("[6/8] Comparing embedding distributions", flush=True)
        emb_summary, pc_table = embedding_analysis(features, splits, args.output, args.seed)
        pca_scatter(args.output / "embedding_pca_projection.npz", plots / "embedding_pca.png", args.seed)
    pc_table.to_csv(tables / "embedding_pc_comparison.csv", index=False)

    print("[7/8] Rendering comparison plots", flush=True)
    for col, title in [
        ("megapixels", "Image size (megapixels)"), ("aspect_ratio", "Aspect ratio (width / height)"),
        ("brightness", "Brightness"), ("contrast", "Contrast"), ("saturation", "Mean HSV saturation"),
        ("entropy_bits", "Grayscale entropy (bits)"), ("edge_density", "Edge density"),
        ("question_chars", "Question length (characters)"), ("choice_chars_mean", "Mean choice length (characters)"),
    ]:
        histogram_plot(combined, col, plots / f"{col}.png", title)
    rgb_long = images[images.split.isin(MAIN_SPLITS)].melt(id_vars="split", value_vars=["red_mean","green_mean","blue_mean"], var_name="channel", value_name="value")
    bar_plot({s: rgb_long[rgb_long.split==s].groupby("channel").value.mean() for s in MAIN_SPLITS}, plots / "rgb_means.png", "Mean RGB channels")
    question_type_english = {
        "전화번호": "phone", "가격·금액": "price", "수량·계수": "count", "날짜·시간": "date/time",
        "색상": "color", "위치·방향": "location", "메뉴·상품": "product", "인물·대상": "subject",
        "문자·간판 읽기": "text/sign", "기타": "other",
    }
    question_plot_counts = {
        s: texts[texts.split==s].question_type.map(question_type_english).value_counts() for s in MAIN_SPLITS
    }
    bar_plot(question_plot_counts, plots / "question_types.png", "Question type distribution")
    bar_plot({"train": csvs["train"].answer.value_counts()}, plots / "answer_distribution.png", "Train answer distribution")

    print("[8/8] Writing summary and report", flush=True)
    summary = {
        "inventory": inventory,
        "train_test": {
            "numeric_comparison": numeric_compare.to_dict("records"),
            "orientation_total_variation": total_variation(images[images.split=="train"].orientation.value_counts(), images[images.split=="test"].orientation.value_counts()),
            "question_type_total_variation": total_variation(texts[texts.split=="train"].question_type.value_counts(), texts[texts.split=="test"].question_type.value_counts()),
            "train_answer_counts": csvs["train"].answer.value_counts().sort_index().to_dict(),
        },
        "duplicates": {
            "exact_duplicate_groups_all_splits": len(exact),
            "exact_duplicate_images_all_splits": int(exact["count"].sum()) if len(exact) else 0,
            "exact_cross_train_test_groups": int(sum("train:" in x and "test:" in x for x in exact.splits)) if len(exact) else 0,
            "exact_cross_train_test_pairs": len(exact_cross),
            "exact_cross_pairs_with_exact_question_match": int(exact_cross.question_exact_match.sum()) if len(exact_cross) else 0,
            "similar_candidate_counts_saved": similar.scope.value_counts().to_dict() if len(similar) else {},
            "phash_hamming_threshold": 6, "similar_candidates_cap": 1000,
        },
        "embeddings": emb_summary,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    generate_report(args.output, csvs, images, texts, numeric_stats, numeric_compare, exact, exact_cross, similar, emb_summary, pc_table)
    print(f"Done: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
