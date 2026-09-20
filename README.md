# scene-text-vision

SSAFY 16기 2회차 AI 챌린지(텍스트 이미지 기반 질의응답, 9/21 09:00 ~ 9/28) 팀 레포입니다.
이미지 속 글자를 읽어 4지선다(a/b/c/d) 정답을 고르는 VLM을 만듭니다.

- 대회 페이지: https://www.kaggle.com/competitions/ssafy-16-2-ai-9-21-9-28
- 자원: Colab Pro 1개(27B 전용), RTX 5070 Ti 16GB PC 4대(9B 이하·검증·보조 모델)

## 처음 오면 이 순서로 읽기 (15분)

1. `docs/DAY1_PLAN.md`: 첫날 시간표, 자리별 역할, 첫날 결정 규칙, 옛 기법의 유효성 판정.
2. `docs/PLAN.md`: 직전 대회 1·2위 코드·디스커션 분석, 규정, 모델 선택, 리스크.
3. `docs/PAPERS.md`: 논문 조사와 "아직 안 써본" 신규 접근법 우선순위.
4. Issues 탭: 오늘 할 일은 전부 `task` 이슈로 올라가 있습니다. 하나를 맡으면 assignee를 자기로 바꿉니다.

## 한 줄 전략

**Qwen3.5-27B + 보기 셔플 LoRA(정답 글자 1토큰 학습) + a/b/c/d 로짓 채점 + 보기 순서 TTA**를 골격으로 하고,
원본 해상도에 맞춘 visual token, 불확실 문항만 crop·고해상도 재추론, 계열이 다른 모델(한국어 특화 VLM 포함)의 로그확률 앙상블을 얹습니다.
근거는 16기 1회차 최종 1위·2위 공개 코드입니다(`docs/PLAN.md` 2절).

## 이슈 규약

| 종류 | 언제 | 템플릿 |
|---|---|---|
| **Task** | 실제 맡은 작업 하나 (자리, 완료 기준, 산출물 위치가 있어야 함) | `[Task] ...` |
| **Insight** | 조사·학습·실험에서 배운 것을 팀에 공유 (채택/조건부/폐기 판정 포함) | `[Insight] ...` |

닫은 길(효과 없던 시도)도 Insight로 남깁니다. 같은 막다른 길을 두 번 밟지 않기 위해서입니다.

## 팀 공유 규약 (첫날 확정)

- **검증 분할**: 이미지 단위 그룹 분할, 시드와 id 목록을 Drive에 고정. 모든 모델은 같은 분할로 검증합니다.
- **확률 파일**: `*_probs.npz` (`avg` [N,4], `runs` [T,N,4], `ids`). 행 순서는 `test.csv` 순서. `scripts/vqa_textmc.py`가 이 형식으로 저장합니다.
- **제출 예산**: 하루 20회(1회차 규정 기준, 공개 후 재확인). 제출 담당자 1명. 챔피언과 다른 문항 수 D가 √D 문턱을 넘는 후보만 제출합니다(`scripts/blend_probs.py`가 출력).
- **산출물 위치**: Colab `/content`는 세션 회수 시 사라지므로 adapter·npz·csv는 반드시 Drive에 둡니다.

## 레포 구조

```
docs/        PLAN.md, DAY1_PLAN.md, PAPERS.md
scripts/     vqa_textmc.py(학습·추론·TTA·불확실도), inspect_data.py(데이터 점검), blend_probs.py(확률 결합), ocr_extract.py(선택)
notebooks/   colab_runbook.ipynb (Colab 실행 순서)
.github/     이슈 템플릿(task, insight), 첫날 이슈 원문
```

## 스크립트 요약

```
python scripts/inspect_data.py --root <데이터폴더>                      # 컬럼·해상도·정답 분포·중복 점검
python scripts/vqa_textmc.py --mode dryrun   --root <데이터폴더>          # torch 없이 형식 검증
python scripts/vqa_textmc.py --mode smoke    --root ... --model-id Qwen/Qwen3.5-27B
python scripts/vqa_textmc.py --mode zeroshot --root ... --model-id Qwen/Qwen3.5-27B --tta-perms 2
python scripts/vqa_textmc.py --mode all      --root ... --model-id Qwen/Qwen3.5-27B --epochs 1 --lr 1e-4
python scripts/blend_probs.py --primary A_probs.npz --secondary B_probs.npz --strategy gated --gate-margin 0.15 --w 0.35 --out sub.csv
```

컬럼명이 다르면 `--id-col/--path-col/--question-col/--choice-cols a,b,c,d/--answer-col`로 매핑합니다. 데이터 공개 후 실제 형식에 맞춰 프롬프트와 컬럼 매핑을 조정할 예정입니다.

## PC 환경 (5070 Ti)

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-pc.txt
```

Windows는 `PYTHONUTF8=1` 환경변수를 설정합니다. Colab은 노트북 첫 셀의 설치 명령을 쓰고 런타임을 재시작합니다.
