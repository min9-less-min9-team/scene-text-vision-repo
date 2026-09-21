# EDA

scene-text VQA 데이터(train/dev/test)의 정적 분석입니다. **정리된 결론은 [`docs/EDA.md`](../docs/EDA.md)** 에 있습니다.

| 파일 | 내용 |
|---|---|
| `run_eda.py` | 이미지 EDA 스크립트: 무결성, 크기·밝기·복잡도, 중복/유사 이미지, ResNet-18 임베딩 분포 (+ 기본 텍스트 지표) |
| `report.md` | `run_eda.py`가 생성한 상세 리포트 (표·차트 포함) |
| `text_eda_vqa.ipynb` | 텍스트 EDA 노트북: 길이·질문 유형·정답 편향·dev 합의도·train/test 텍스트 shift·중복 |
| `difficulty_probe.ipynb` | Zero-shot 난이도 probe: Qwen3.5-2B/4B의 a/b/c/d logits로 train/dev 샘플별 `P(correct)`·난이도 tier 산출 → `tables/difficulty_{split}.csv` |
| `difficulty.py` | 위 노트북의 분석 함수: `load_probe`, `difficulty_table`, `probe_summary`, `tier_counts` |
| `common.py` | 공용 헬퍼: `load_csvs`, `text_metrics`, `question_type`, `numeric_comparison`(SMD·KS), `total_variation`, `dev_vote_features` |
| `tables/`, `plots/`, `summary.json` | 원시/요약 표, 차트, 주요 수치 요약 |

## 재실행

```bash
# repo root에서
TORCH_HOME=eda/.cache/torch uv run python eda/run_eda.py            # 전체 (이미지 16,111장 스캔)
TORCH_HOME=eda/.cache/torch uv run python eda/run_eda.py --resume   # 완성된 지표 CSV·임베딩 캐시 재사용

uv sync --group eda    # 노트북용 scipy / scikit-learn / matplotlib / ipykernel
```

최초 실행 시 ImageNet 사전학습 ResNet-18 가중치(약 45MB)를 받습니다. `eda/.cache/`와 원본 512차원 임베딩 캐시(`resnet18_embeddings.npz`)는 Git에서 제외합니다.

## 다른 분석에서 재사용

```python
import sys; sys.path.insert(0, "eda")   # repo root 기준
from common import load_csvs, text_metrics, dev_vote_features

csvs = load_csvs()
votes = dev_vote_features(csvs["dev"])   # vote_count, majority_answer, is_tie, consensus_ratio, vote_entropy
```
