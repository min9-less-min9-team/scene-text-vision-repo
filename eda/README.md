# EDA

scene-text VQA 데이터(train/dev/test)의 정적 분석입니다. **정리된 결론은 [`docs/EDA.md`](../docs/EDA.md)** 에 있습니다.

데이터 경로: 원본은 `data/raw/`(train.csv, test.csv, dev.csv, 이미지 폴더), 분할은 `data/split/`(train.csv 90% / validation.csv 10%). 둘 다 Git에서 제외됩니다.

| 파일 | 내용 |
|---|---|
| `split_validation.py` | `data/raw/train.csv` → `data/split/{train,validation}.csv` (10%, 이미지 SHA-256·pHash 그룹 단위, 정답 층화, seed 42). 이미 있으면 skip, `--force`로 재생성. 검증 id 목록은 `tables/validation_ids.csv`에 버전 관리. `ensure_split()`으로 노트북에서도 호출 |
| `split_eda.ipynb` | train / validation / test 비교 노트북: 정답 분포, 종횡비·픽셀, 밝기·대비·HSV, 질문·선지 길이, 질문 유형 (이미지 지표 캐시 `tables/split_image_metrics.csv`) |
| `run_eda.py` | 이미지 EDA 스크립트: 무결성, 크기·밝기·복잡도, 중복/유사 이미지, ResNet-18 임베딩 분포 (+ 기본 텍스트 지표) |
| `report.md` | `run_eda.py`가 생성한 상세 리포트 (표·차트 포함) |
| `text_eda_vqa.ipynb` | 텍스트 EDA 노트북: 길이·질문 유형·정답 편향·dev 합의도·train/test 텍스트 shift·중복 |
| `difficulty_probe.ipynb` | Zero-shot 난이도 probe(**Colab에서 열어 실행**, 결과는 Drive `MyDrive/stv/outputs/`): Qwen3.5-2B/4B의 a/b/c/d logits로 train/dev 샘플별 `P(correct)`·난이도 tier 산출 → `difficulty_{split}.csv`. 분석 함수(`load_probe`, `difficulty_table`, `probe_summary`, `tier_counts`)도 노트북 안에 포함 |
| `common.py` | 공용 헬퍼: `load_csvs`, `text_metrics`, `question_type`, `numeric_comparison`(SMD·KS), `total_variation`, `dev_vote_features` |
| `tables/`, `plots/`, `summary.json` | 원시/요약 표, 차트, 주요 수치 요약 |

## 재실행

```bash
# repo root에서
uv run python eda/split_validation.py                                # data/raw → data/split (있으면 skip)
TORCH_HOME=eda/.cache/torch uv run python eda/run_eda.py            # 전체 (이미지 16,111장 스캔, data/raw 기준)
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
