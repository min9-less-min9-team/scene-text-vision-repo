# scene-text-vision-repo

SSAFY 16기 2회차 AI 챌린지(텍스트 이미지 기반 질의응답, 9/21 09:00 ~ 9/28) 팀 레포입니다.
이미지 속 글자를 읽어 4지선다(a/b/c/d) 정답을 고르는 VLM을 만듭니다.

- 대회 페이지: https://www.kaggle.com/competitions/ssafy-16-2-ai-9-21-9-28
- 자원: Colab Pro 1개(27B 전용), RTX 5070 Ti 16GB PC 4대(9B 이하·검증·보조 모델)

## 처음 오면 이 순서로 읽기 (15분)

1. `docs/DAY1_PLAN.md`: 첫날 시간표, 자리별 역할, 첫날 결정 규칙, 옛 기법의 유효성 판정.
2. `docs/PLAN.md`: 직전 대회 1·2위 코드·디스커션 분석, 규정, 모델 선택, 리스크.
3. `docs/PAPERS.md`: 논문 조사와 "아직 안 써본" 신규 접근법 우선순위.
4. `docs/EDA.md`: 이미지/텍스트 EDA 결론 (train=test 분포, dev는 모호 문항 집합, train↔test 중복 이미지).
5. Issues 탭: 오늘 할 일은 전부 `task` 이슈로 올라가 있습니다. 하나를 맡으면 assignee를 자기로 바꿉니다.

## 한 줄 전략

**Qwen3-VL-4B(Unsloth 4bit) + 보기 셔플 LoRA(정답 글자 토큰만 loss) + a/b/c/d 로짓 채점 + 보기 순서 TTA**가 현재 베이스라인입니다(`src/stv`, RTX 5060 Ti 16GB 검증: zero-shot 91.6% → 파인튜닝 94.4% → +TTA 95.2%, 검증 500문항).
그 위에 더 큰 모델(8B/27B), 불확실 문항만 crop·고해상도 재추론, 계열이 다른 모델의 로그확률 앙상블을 얹습니다.
근거는 16기 1회차 최종 1위·2위 공개 코드입니다(`docs/PLAN.md` 2절). 원본 노트북은 `notebooks/unsloth_5060ti.ipynb`입니다.

## 이슈 규약

| 종류 | 언제 | 템플릿 |
|---|---|---|
| **Task** | 실제 맡은 작업 하나 (자리, 완료 기준, 산출물 위치가 있어야 함) | `[Task] ...` |
| **Insight** | 조사·학습·실험에서 배운 것을 팀에 공유 (채택/조건부/폐기 판정 포함) | `[Insight] ...` |

닫은 길(효과 없던 시도)도 Insight로 남깁니다. 같은 막다른 길을 두 번 밟지 않기 위해서입니다.

## 팀 공유 규약 (첫날 확정)

- **검증 분할**: `train.csv`에서 500문항(`valid_size=500`, `split_seed=42` → `--split valid`). 모든 실험과 공개 어댑터(`ssafyjinhyeok/KFC`)가 같은 분할을 쓰므로 바꾸지 않습니다. `dev.csv`는 5명 답의 다수결(동률 제외)로 채점하지만 라벨 노이즈가 커서 참고용입니다(`docs/EDA.md`).
- **확률 파일**: `{split}_probs.npz` (`avg` [N,4], `ids`). 행 순서는 해당 csv 순서. `stv-train`(valid)·`stv-infer`가 이 형식으로 저장하고 `stv-ensemble`이 평균합니다.
- **제출 예산**: 하루 20회(1회차 규정 기준, 공개 후 재확인). 제출 담당자 1명. 챔피언과 다른 문항 수 D가 √D 문턱을 넘는 후보만 제출합니다.
- **산출물 위치**: Colab `/content`는 세션 회수 시 사라지므로 adapter·npz·csv는 반드시 Drive에 둡니다(`--output-dir`).

## 레포 구조

```
pyproject.toml, uv.lock   uv 환경 (torch cu128 인덱스)
configs/     sample.toml(기본 설정, 주석 참고)
src/stv/     config.py    Config dataclass. 모든 필드가 toml 키이자 --kebab-case CLI 플래그
             data.py      csv 로드, 프롬프트(변형 a/b/c), 검증 분할, dev 다수결, 보기 셔플 Dataset
             model.py     Unsloth 4bit 로드, LoRA 부착, 시작 어댑터 로드, 픽셀 예산
             train.py     정답 토큰만 loss, eval_steps마다 검증 → best 어댑터 저장, 최종 valid TTA
             inference.py a/b/c/d 로짓 → 확률, 보기 순서 TTA, 이어하기 캐시, submission csv + probs npz
             ensemble.py  실험별 확률 평균 → 검증 점수 + 앙상블 제출 파일
             bench.py     추론 최대 배치 크기 측정
scripts/     env.sh(환경 감지), setup.sh(환경 구성·데이터 연결), run.sh(train→dev→test),
             runpod_launch.py(pod 생성·삭제), runpod_job.sh(pod 안에서 도는 실험), s3.sh(volume 업로드·다운로드)
notebooks/   colab.ipynb, unsloth_5060ti.ipynb(베이스라인 원본 노트북, 단독 실행용)
eda/         run_eda.py(이미지 EDA), text_eda_vqa.ipynb(텍스트 EDA), common.py(공용 헬퍼), report.md·tables·plots
docs/        PLAN.md, DAY1_PLAN.md, PAPERS.md, EDA.md
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

uv run stv-train --config configs/sample.toml --max-train-samples 48 --valid-size 24 --eval-steps 3 --output-dir outputs/quick   # 동작 확인
uv run stv-train --config configs/sample.toml                          # 전체 학습 (검증 500, best 어댑터 저장, 최종 valid TTA)
uv run stv-infer --config configs/sample.toml --split test             # submission_test.csv, test_probs.npz (끊기면 재실행 시 이어서)
uv run stv-infer --config configs/sample.toml --split valid --adapter-dir none   # zero-shot
uv run stv-infer --config configs/sample.toml --init-adapter ssafyjinhyeok/KFC --adapter-dir none --split valid   # 공개 어댑터 그대로
uv run stv-ensemble outputs/qwen3_vl_4b outputs/run_seed1              # 확률 평균 → 검증 점수 + outputs/ensemble/submission_test.csv

bash scripts/run.sh configs/sample.toml --output-dir outputs/full      # train→dev→test 한 번에
```

산출물은 `--output-dir`(기본 `outputs/qwen3_vl_4b`)에 adapter, `log.txt`(loss·검증 정확도), `submission_{split}.csv`, `{split}_probs.npz`로 저장됩니다.
학습 설정을 바꿀 때는 `--output-dir`를 다르게 주세요(어댑터가 덮어써짐). OOM이면 `batch_size 1 / grad_accum 8` → `finetune_vision false` → `train_max_pixels` 축소 → `lora_r 8` 순으로 줄입니다.

## 환경별 사용법

`scripts/env.sh`가 환경을 감지해 캐시 위치와 python 실행 방식을 정합니다.

| 환경 | 방법 |
|---|---|
| **로컬 PC (RTX 50xx)** | `bash scripts/setup.sh` 후 위 `uv run ...` 명령. torch는 cu128 휠로 고정돼 있습니다. |
| **RunPod** | `bash scripts/setup.sh /workspace/data.zip` → `nohup bash scripts/run.sh configs/sample.toml > run.log 2>&1 &`. HF·uv 캐시는 `/workspace`(영구 볼륨)에 둡니다. |
| **Colab** | `notebooks/colab.ipynb`를 위에서부터 실행. 데이터는 `MyDrive/stv/data.zip`, 결과는 Drive에 저장. private repo라 clone 시 토큰이 필요합니다. |

### RunPod 원격 실험 (GPU 대여 → wandb 기록 → Network Volume/S3 저장)

팀원은 `.env`만 본인 키로 채우면 같은 명령으로 돌릴 수 있습니다. 로컬에는 `python3`와 `uv`만 있으면 됩니다(GPU 불필요).

```bash
cp .env.example .env                           # 본인 키 입력 (git에 올라가지 않음)
bash scripts/s3.sh push-data data.zip          # 최초 1회: 데이터를 volume의 stv/data.zip으로 업로드

python scripts/runpod_launch.py run configs/sample.toml                          # pod 생성 → 학습·dev 채점·test 추론 → pod 자동 삭제
python scripts/runpod_launch.py run configs/sample.toml --name r32 -- --lora-r 32 --lora-alpha 32
python scripts/runpod_launch.py list           # 내 pod / stop POD_ID로 수동 삭제

bash scripts/s3.sh log r32                     # 진행 로그(run.log)
bash scripts/s3.sh pull r32                    # 결과를 outputs/{STV_USER}/r32/ 로 내려받기
```

- **동작**: pod가 뜨면 `STV_BRANCH`를 clone해 `scripts/runpod_job.sh`를 실행합니다. 따라서 **설정 파일·코드 변경은 push한 뒤** 실행해야 반영됩니다.
- **저장**: 결과는 volume의 `stv/outputs/{STV_USER}/{실험 이름}/`에 바로 기록됩니다(Network Volume = S3 bucket). HF 모델 캐시도 volume에 남아 다음 실험부터는 다시 받지 않습니다.
- **wandb**: `WANDB_API_KEY`가 있으면 train loss·lr, `valid/acc`(eval_steps마다), 최종 valid/dev 정확도가 run `{STV_USER}/{실험 이름}`에 기록됩니다. 키가 없으면 기록 없이 돕니다(로컬·Colab도 동일).
- **과금**: 실험이 끝나면 성공·실패와 관계없이 pod를 삭제합니다. `--keep`을 쓴 경우에는 직접 `stop` 해야 합니다. volume 보관료는 별도입니다.
- Network Volume은 Secure Cloud의 해당 데이터센터 GPU에만 붙습니다. 재고가 없으면 `RUNPOD_GPU`에 대체 GPU를 쉼표로 나열하세요.

Windows는 `PYTHONUTF8=1` 환경변수를 설정하고, 스크립트는 WSL 터미널에서 실행합니다.
