# scene-text-vision-repo

SSAFY 16기 2회차 AI 챌린지(텍스트 이미지 기반 질의응답, 9/21 09:00 ~ 9/28) 팀 레포입니다.
이미지 속 글자를 읽어 4지선다(a/b/c/d) 정답을 고르는 VLM을 만듭니다.

- 대회 페이지: https://www.kaggle.com/competitions/ssafy-16-2-ai-9-21-9-28
- 자원: Colab Pro 1개(27B 전용), RTX 50xx 16GB PC 4대(9B 이하·검증·보조 모델)

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

- **검증 분할**: 이미지 단위 그룹 분할(`train.csv`의 10%, seed 42 → `--split valid`). 별도로 `dev.csv`(annotator 5명 답의 다수결로 채점 → `--split dev`)를 공통 검증셋으로 씁니다.
- **확률 파일**: `{split}_probs.npz` (`avg` [N,4], `ids`). 행 순서는 해당 csv 순서. `stv.inference`가 이 형식으로 저장합니다.
- **제출 예산**: 하루 20회(1회차 규정 기준, 공개 후 재확인). 제출 담당자 1명. 챔피언과 다른 문항 수 D가 √D 문턱을 넘는 후보만 제출합니다.
- **산출물 위치**: Colab `/content`는 세션 회수 시 사라지므로 adapter·npz·csv는 반드시 Drive에 둡니다(`--output-dir`).

## 레포 구조

```
pyproject.toml, uv.lock   uv 환경 (torch cu128 인덱스)
configs/     sample.toml(기본), lab.toml(Docker lab: 캐시된 모델을 오프라인으로 사용)
src/stv/     config.py    Config dataclass. 모든 필드가 toml 키이자 --kebab-case CLI 플래그
             data.py      csv 로드, 프롬프트/messages 생성, train/valid/dev/test 분할
             model.py     Unsloth 4bit 로드, LoRA 부착
             train.py     SFTTrainer 학습 → LoRA adapter 저장
             inference.py a/b/c/d 다음 토큰 로짓 → 확률, submission csv + probs npz 저장
scripts/     env.sh(환경 감지), setup.sh(환경 구성·데이터 연결), run.sh(train→dev→test), lab_sync.sh(Docker lab으로 코드 복사)
notebooks/   colab.ipynb, lab.ipynb
docs/        PLAN.md, DAY1_PLAN.md, PAPERS.md
.github/     이슈 템플릿(task, insight), 첫날 이슈 원문
```

> `docs/`는 초기 계획 문서라 예전 스크립트(`vqa_textmc.py`, `blend_probs.py` 등)를 언급합니다. 보기 셔플·TTA·확률 결합은 새 구조에 아직 옮기지 않았습니다.

## 데이터

`data/` 아래에 대회 데이터를 그대로 둡니다(git에는 올리지 않습니다).

```
data/train.csv  id,path,question,a,b,c,d,answer          + data/train/*.jpg
data/dev.csv    id,path,question,a,b,c,d,answer1..5      + data/dev/*.jpg
data/test.csv   id,path,question,a,b,c,d                 + data/test/*.jpg
data/sample_submission.csv  id,answer
```

## 실행

설정 우선순위는 **CLI 플래그 > `--config` toml > 기본값**입니다. 전체 옵션은 `src/stv/config.py` 또는 `--help`.

```bash
bash scripts/setup.sh [data.zip | data_dir]      # uv 설치 → uv sync → 데이터 연결 → GPU 확인

uv run stv-train --config configs/sample.toml                          # 200샘플 smoke
uv run stv-train --config configs/sample.toml --max-train-samples 0    # 전체 학습
uv run stv-infer --config configs/sample.toml --split dev              # dev 정확도
uv run stv-infer --config configs/sample.toml --split test             # submission_test.csv, test_probs.npz
uv run stv-infer --config configs/sample.toml --split dev --adapter-dir none   # zero-shot

bash scripts/run.sh configs/sample.toml --max-train-samples 0 --output-dir outputs/full   # train→dev→test 한 번에
```

산출물은 `--output-dir`(기본 `outputs/qwen2_5_vl_3b_lora`)에 adapter, `submission_{split}.csv`, `{split}_probs.npz`로 저장됩니다.

## 환경별 사용법

`scripts/env.sh`가 환경을 감지해 캐시 위치와 python 실행 방식을 정합니다.

| 환경 | 방법 |
|---|---|
| **로컬 PC (RTX 50xx)** | `bash scripts/setup.sh` 후 위 `uv run ...` 명령. torch는 cu128 휠로 고정돼 있습니다. |
| **RunPod** | `bash scripts/setup.sh /workspace/data.zip` → `nohup bash scripts/run.sh configs/sample.toml > run.log 2>&1 &`. HF·uv 캐시는 `/workspace`(영구 볼륨)에 둡니다. |
| **Colab** | `notebooks/colab.ipynb`를 위에서부터 실행. 데이터는 `MyDrive/stv/data.zip`, 결과는 Drive에 저장. private repo라 clone 시 토큰이 필요합니다. |
| **Docker Jupyter lab** (`ssafy-ai`, http://127.0.0.1:8888) | 아래 참고 |

### Docker Jupyter lab

컨테이너에 설치된 런타임(시스템 python + torch/unsloth/trl)을 그대로 쓰고 uv venv는 쓰지 않습니다. 데이터는 컨테이너의 `/workspace`에 있는 것을 씁니다.

1. WSL 터미널에서 `bash scripts/lab_sync.sh` — 코드를 컨테이너 `/workspace/scene-text-vision-repo`로 복사하고 `stv`를 설치합니다. **`.py`/toml을 고칠 때마다 다시 실행**합니다(노트북 셀만 고쳤으면 불필요).
2. VS Code에서 `notebooks/lab.ipynb`를 열고 커널을 *Existing Jupyter Server* → `http://127.0.0.1:8888`(토큰 `lab`)로 선택해 실행합니다.

- 컨테이너의 HF 미러가 닿지 않아 `HF_HUB_OFFLINE=1`로 캐시된 모델만 씁니다(`configs/lab.toml`: `Qwen/Qwen2.5-VL-3B-Instruct`). 다른 모델을 받으려면 `HF_HUB_OFFLINE=0 HF_ENDPOINT=https://huggingface.co`로 실행합니다.
- 결과물은 컨테이너 안에 생깁니다: `docker cp ssafy-ai:/workspace/scene-text-vision-repo/outputs ./outputs`
- 여기서 학습한 adapter는 base 모델 경로가 컨테이너 내부 경로로 기록됩니다. 다른 환경에서 쓰려면 `adapter_config.json`의 `base_model_name_or_path`를 hub id로 바꿉니다.

Windows는 `PYTHONUTF8=1` 환경변수를 설정하고, 스크립트는 WSL 터미널에서 실행합니다.
