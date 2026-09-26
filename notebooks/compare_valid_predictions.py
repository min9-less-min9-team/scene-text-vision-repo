#!/usr/bin/env python3
"""
variant별 validation 예측 비교 — accuracy가 같아도 어떤 문항에서 갈렸는지 확인.

위치:  <프로젝트 루트>/notebook/compare_valid_predictions.py
데이터: <프로젝트 루트>/data/raw   (train.csv가 하위 어딘가에 있으면 됨)
split : <프로젝트 루트>/data/split/validation.csv (없으면 eda/split_validation.py로 생성)

실행 예:
    cd <프로젝트 루트>/notebook
    python compare_valid_predictions.py --probs-dir ./data_experiments
    python compare_valid_predictions.py --probs-dir ./data_experiments --variants raw dev_aug_human

필요 파일 (--probs-dir 안):
    probs_valid_<variant>.npy   (행 순서 = validation.csv 순서)
    comparison.csv              (선택. 있으면 accuracy 재계산 값과 대조해 행 순서 일치를 검증)

출력 (--out-dir, 기본 ./diff_report):
    summary.csv            variant별 accuracy / NLL / Brier / 정답 확률 평균
    pairwise.csv           variant 쌍별 한쪽만 맞힌 수 / McNemar exact p-value
    disagreements.csv      예측이 갈린 문항 전체 (보기·정답·variant별 예측/확률)
    disagreements.html     위 문항을 이미지와 함께 보는 리포트
"""
import argparse
import html
import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHOICES = ["a", "b", "c", "d"]
DEFAULT_VARIANTS = ["raw", "dev_aug_legacy", "dev_aug_human"]


# ---------------------------------------------------------------------
# 경로 / 데이터
# ---------------------------------------------------------------------
def find_project_root(start: Path) -> Path:
    for d in [start, *start.parents]:
        if (d / "eda" / "split_validation.py").exists() or (d / "data" / "raw").is_dir():
            return d
    return start.parent


def load_validation(root: Path):
    raw_base = root / "data" / "raw"
    split_dir = root / "data" / "split"
    hits = sorted(raw_base.rglob("train.csv"))
    if not hits:
        sys.exit(f"train.csv를 찾지 못함: {raw_base}")
    raw_data_dir = hits[0].parent

    valid_csv = split_dir / "validation.csv"
    if not valid_csv.exists():
        split_py = root / "eda" / "split_validation.py"
        if not split_py.exists():
            sys.exit(f"{valid_csv}도 없고 {split_py}도 없어 validation split을 만들 수 없습니다.")
        sys.path.insert(0, str(split_py.parent))
        from split_validation import ensure_split  # noqa: E402
        print(f"[split] {valid_csv} 없음 → ensure_split 실행")
        ensure_split(raw_data_dir, split_dir)

    df = pd.read_csv(valid_csv)
    df["id"] = df["id"].astype(str)
    df["answer"] = df["answer"].astype(str).str.strip().str.lower()
    assert df["answer"].isin(CHOICES).all(), "validation answer는 a~d만 허용"
    return df, raw_data_dir


def resolve_image(raw_data_dir: Path, rel: str):
    for base in (raw_data_dir, raw_data_dir.parent):
        p = base / str(rel)
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------
# 통계
# ---------------------------------------------------------------------
def mcnemar_exact_p(b: int, c: int) -> float:
    """양측 exact McNemar (이항검정, p=0.5). b, c = 한쪽만 맞힌 문항 수."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs-dir", type=Path, default=here,
                    help="probs_valid_<variant>.npy, comparison.csv가 있는 폴더 (기본: 스크립트 폴더)")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--out-dir", type=Path, default=here / "diff_report")
    ap.add_argument("--root", type=Path, default=None, help="프로젝트 루트 (기본: 자동 탐색)")
    args = ap.parse_args()

    root = (args.root or find_project_root(here)).resolve()
    out_dir = args.out_dir.resolve()
    valid_df, raw_data_dir = load_validation(root)
    n = len(valid_df)
    gold = valid_df["answer"].map(CHOICES.index).to_numpy()
    print(f"root={root}\nvalidation={n} rows | raw={raw_data_dir}")

    # ---- probs 로드 ----
    probs, preds, correct = {}, {}, {}
    for v in args.variants:
        path = args.probs_dir / f"probs_valid_{v}.npy"
        if not path.exists():
            sys.exit(f"없음: {path}")
        p = np.load(path)
        assert p.shape == (n, 4), f"{v}: shape {p.shape} != ({n}, 4) → validation split이 학습 때와 다릅니다."
        probs[v], preds[v] = p, p.argmax(1)
        correct[v] = preds[v] == gold

    # ---- 행 순서 검증: comparison.csv의 valid_acc와 재계산 값 대조 ----
    cmp_path = args.probs_dir / "comparison.csv"
    if cmp_path.exists():
        cmp = pd.read_csv(cmp_path, encoding="utf-8-sig").set_index("variant")
        for v in args.variants:
            if v in cmp.index:
                re_acc, logged = correct[v].mean(), float(cmp.loc[v, "valid_acc"])
                ok = abs(re_acc - logged) < 5e-4
                print(f"[check] {v}: 재계산 {re_acc:.4f} vs 기록 {logged:.4f} → {'OK' if ok else 'MISMATCH'}")
                if not ok:
                    sys.exit("accuracy가 다릅니다. 로컬 validation.csv 순서/내용이 서버와 다릅니다.")
    else:
        print("[check] comparison.csv 없음 → 행 순서 검증 생략")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) variant별 요약: accuracy가 같아도 확신도(NLL/Brier)는 다를 수 있음 ----
    onehot = np.eye(4)[gold]
    rows = []
    for v in args.variants:
        p = np.clip(probs[v], 1e-9, 1)
        p_gold = p[np.arange(n), gold]
        wrong = ~correct[v]
        rows.append({
            "variant": v,
            "correct": int(correct[v].sum()),
            "accuracy": round(float(correct[v].mean()), 4),
            "nll": round(float(-np.log(p_gold).mean()), 4),
            "brier": round(float(((probs[v] - onehot) ** 2).sum(1).mean()), 4),
            "mean_p_gold": round(float(p_gold.mean()), 4),
            "mean_p_gold_on_wrong": round(float(p_gold[wrong].mean()), 4) if wrong.any() else None,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    print("\n[summary]\n" + summary.to_string(index=False))

    # ---- 2) 쌍별 비교: A만 맞힘 / B만 맞힘 ----
    pw = []
    for a, b in itertools.combinations(args.variants, 2):
        a_only = int((correct[a] & ~correct[b]).sum())
        b_only = int((~correct[a] & correct[b]).sum())
        pw.append({
            "A": a, "B": b,
            "pred_diff": int((preds[a] != preds[b]).sum()),
            "A_only_correct": a_only,
            "B_only_correct": b_only,
            "both_wrong_diff_pred": int((~correct[a] & ~correct[b] & (preds[a] != preds[b])).sum()),
            "mcnemar_p": round(mcnemar_exact_p(a_only, b_only), 4),
        })
    pairwise = pd.DataFrame(pw)
    pairwise.to_csv(out_dir / "pairwise.csv", index=False, encoding="utf-8-sig")
    print("\n[pairwise]\n" + pairwise.to_string(index=False))

    # ---- 3) 예측이 갈린 문항 ----
    P = np.stack([preds[v] for v in args.variants])
    idx = np.where((P != P[0]).any(0))[0]

    recs = []
    for i in idx:
        r = valid_df.iloc[i]
        ok = [v for v in args.variants if correct[v][i]]
        rec = {"row": int(i), "id": r["id"], "path": r["path"], "question": r["question"],
               **{c: r[c] for c in CHOICES}, "answer": r["answer"],
               "correct_by": ",".join(ok) if ok else "none"}
        for v in args.variants:
            rec[f"pred_{v}"] = CHOICES[preds[v][i]]
            rec[f"p_gold_{v}"] = round(float(probs[v][i, gold[i]]), 3)
            rec[f"p_pred_{v}"] = round(float(probs[v][i, preds[v][i]]), 3)
        recs.append(rec)
    diff_df = pd.DataFrame(recs)
    if len(diff_df):
        diff_df = diff_df.sort_values(["correct_by", "id"]).reset_index(drop=True)
    diff_df.to_csv(out_dir / "disagreements.csv", index=False, encoding="utf-8-sig")

    print(f"\n[disagreements] {len(diff_df)}/{n}  (correct_by = 해당 문항을 맞힌 variant)")
    if len(diff_df):
        print(diff_df["correct_by"].value_counts().to_string())

    # ---- 4) HTML 리포트 ----
    write_html(out_dir / "disagreements.html", diff_df, args.variants, probs, raw_data_dir, out_dir, summary, pairwise)
    print(f"\nsaved → {out_dir}")


def write_html(path, diff_df, variants, probs, raw_data_dir, out_dir, summary, pairwise):
    e = html.escape
    cards = []
    for _, r in diff_df.iterrows():
        img = resolve_image(raw_data_dir, r["path"])
        img_src = os.path.relpath(img.resolve(), out_dir).replace(os.sep, "/") if img else ""
        head = "".join(f"<th>{e(v)}</th>" for v in variants)
        body = []
        for j, c in enumerate(CHOICES):
            cells = []
            for v in variants:
                pv = probs[v][int(r["row"]), j]
                is_pred = r[f"pred_{v}"] == c
                cls = "pred ok" if is_pred and c == r["answer"] else "pred ng" if is_pred else ""
                cells.append(f'<td class="{cls}">{pv * 100:.1f}%</td>')
            gold_cls = ' class="gold"' if c == r["answer"] else ""
            body.append(f"<tr><td{gold_cls}><b>{c}</b> {e(str(r[c]))}</td>{''.join(cells)}</tr>")
        img_html = f'<img src="{e(img_src)}" loading="lazy">' if img_src else "이미지 없음"
        cards.append(f"""
        <div class="card">
          <div class="img">{img_html}</div>
          <div class="info">
            <div class="meta">{e(r['id'])} · 정답 <b>{r['answer'].upper()}</b> · 맞힌 variant: <b>{e(r['correct_by'])}</b></div>
            <div class="q">{e(str(r['question']))}</div>
            <table><tr><th>보기</th>{head}</tr>{''.join(body)}</table>
          </div>
        </div>""")

    doc = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>validation disagreements</title>
<style>
body{{font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif;margin:20px;background:#f5f7fb;color:#172033}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:24px}}
table{{border-collapse:collapse;font-size:13px;background:#fff}}
th,td{{border:1px solid #dfe3ea;padding:5px 8px;text-align:left}}
.card{{display:grid;grid-template-columns:420px 1fr;gap:14px;background:#fff;border:1px solid #dfe3ea;border-radius:10px;padding:12px;margin:12px 0}}
.img img{{width:100%;max-height:420px;object-fit:contain;background:#eef1f6;cursor:zoom-in}}
.meta{{font-size:13px;color:#5b6576;margin-bottom:6px}} .q{{font-weight:700;margin-bottom:8px}}
.gold{{background:#e9f8ee}} .pred{{font-weight:800}} .ok{{color:#15803d}} .ng{{color:#b91c1c;background:#fff0f0}}
@media(max-width:900px){{.card{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>예측이 갈린 validation 문항 ({len(diff_df)}건)</h1>
<p>굵은 글씨 = 해당 variant의 예측 (초록: 정답, 빨강: 오답). 초록 배경 보기 = 정답.</p>
<h2>요약</h2>{summary.to_html(index=False)}
<h2>쌍별 비교</h2>{pairwise.to_html(index=False)}
<h2>문항</h2>{''.join(cards)}
<script>document.querySelectorAll('.img img').forEach(i=>i.onclick=()=>window.open(i.src))</script>
</body></html>"""
    path.write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    main()
