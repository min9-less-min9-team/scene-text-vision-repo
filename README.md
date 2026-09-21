# Scene-Text VQA — RunPod 실험 가이드

`.env`에 본인 키를 넣고 명령 한 줄을 실행하면, RunPod에서 GPU를 빌려 **학습 → dev 채점 → test 제출 파일 생성**까지 돌린 뒤 pod를 스스로 삭제합니다. 학습 과정은 wandb에 기록되고, 결과는 RunPod Network Volume(S3)에 남습니다.

```
내 PC                          RunPod pod (GPU)                         저장소
─────                          ────────────────                         ──────
runpod_launch.py run ──────▶   repo clone → setup.sh → run.sh   ──────▶ wandb (loss, valid/acc, dev acc)
                               (train → dev → test)             ──────▶ Network Volume /workspace/stv/outputs/…
s3.sh log / pull  ◀──────────────────────── S3 API ◀─────────────────── (= S3 bucket)
                               끝나면 pod 자동 삭제
```

내 PC에는 GPU가 필요 없습니다. `python3`, `bash`, [`uv`](https://docs.astral.sh/uv/)만 있으면 됩니다(Windows는 WSL).

---

## 1. 준비물 (최초 1회)

### 1-1. 각자 발급하는 키

| `.env` 변수 | 발급 위치 | 비고 |
|---|---|---|
| `RUNPOD_API_KEY` | RunPod Console → Settings → **API Keys** → Create | pod 생성·삭제 권한이 필요하므로 Read/Write(또는 All) |
| `RUNPOD_S3_ACCESS_KEY`, `RUNPOD_S3_SECRET_KEY` | RunPod Console → Settings → **S3 API Keys** → Create | access key는 `user_…`, secret은 `rps_…`. secret은 생성 직후에만 보입니다 |
| `WANDB_API_KEY` | https://wandb.ai/authorize | 비워 두면 wandb 기록 없이 실행됩니다 |
| `GITHUB_TOKEN` | GitHub → Settings → Developer settings → Fine-grained token | pod가 private repo를 clone하는 데 씁니다. **Resource owner를 조직 `min9-less-min9-team`으로** 선택(개인 계정으로 두면 403) → Repository access에서 이 repo → Permissions **Contents: Read-only** |
| `HF_TOKEN` (선택) | https://huggingface.co/settings/tokens | gated 모델을 쓰거나 다운로드 제한에 걸릴 때만 |

> **키와 volume은 같은 RunPod 계정에서 만들어야 합니다.** 콘솔 좌상단에서 개인 계정/팀 계정을 전환할 수 있는데, 개인 계정에서 만든 API 키·S3 키로는 팀 계정의 volume이 보이지 않습니다(API 404, S3 `AccessDenied … not allowed for bucket`). 과금도 키를 만든 계정으로 나갑니다.

토큰이 맞는지는 pod를 띄우기 전에 확인할 수 있습니다(`run`도 pod 생성 전에 같은 검사를 합니다).

```bash
git -c credential.helper= ls-remote https://토큰@github.com/min9-less-min9-team/scene-text-vision-repo.git feat/runpod
```

### 1-2. Network Volume (팀에서 1개 만들어 공유하거나 각자 생성)

1. RunPod Console → Storage → **New Network Volume**
2. **S3 API를 지원하는 데이터센터**를 고릅니다(생성 화면에 S3 지원 여부가 표시됩니다. 예: `EU-RO-1`, `EUR-IS-1`, `EU-CZ-1`, `US-KS-2`, `US-CA-2`). 쓰려는 GPU가 그 데이터센터에 있는지도 같이 확인하세요. **volume은 같은 데이터센터의 GPU에만 붙습니다.**
3. 크기는 50GB 정도면 충분합니다(데이터 zip + 모델 캐시 약 10~20GB + 실험당 결과 수백 MB).
4. 생성 후 표시되는 **Volume ID** → `RUNPOD_VOLUME_ID`, 데이터센터 ID → `RUNPOD_DATACENTER`

> volume은 pod가 없어도 용량만큼 보관료가 나갑니다. RunPod 계정이 팀원별로 따로라면 volume도 각자 만들어야 합니다(다른 계정의 volume은 붙일 수 없음).

---

## 2. `.env` 작성

```bash
cp .env.example .env      # .env는 .gitignore에 있어 git에 올라가지 않습니다
```

```ini
# --- 개인 ---
STV_USER=yeseo                    # 영문 이름. 결과 폴더·wandb run 이름·pod 이름에 쓰임
RUNPOD_API_KEY=rpa_xxxxxxxx
WANDB_API_KEY=xxxxxxxx
HF_TOKEN=
GITHUB_TOKEN=github_pat_xxxxxxxx

RUNPOD_S3_ACCESS_KEY=user_xxxxxxxx
RUNPOD_S3_SECRET_KEY=rps_xxxxxxxx

# --- 팀 공용 ---
RUNPOD_VOLUME_ID=abc123xyz        # = S3 bucket 이름
RUNPOD_DATACENTER=EU-RO-1
WANDB_ENTITY=our-team             # wandb 팀 이름. 비우면 개인 계정에 기록
WANDB_PROJECT=stv

# --- pod 사양 ---
RUNPOD_GPU="NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 5090"
RUNPOD_GPU_COUNT=1
RUNPOD_CLOUD=SECURE
RUNPOD_DISK_GB=40
RUNPOD_IMAGE=runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
STV_REPO=min9-less-min9-team/scene-text-vision-repo
STV_BRANCH=feat/runpod
```

| 변수 | 설명 |
|---|---|
| `STV_USER` | 결과가 `stv/outputs/{STV_USER}/{실험 이름}/`에 저장되므로 팀원끼리 겹치지 않게 |
| `RUNPOD_GPU` | RunPod GPU type ID. 쉼표로 여러 개를 쓰면 앞에서부터 재고가 있는 것을 잡습니다. 값에 공백이 있으므로 **따옴표 필수**. 기본은 4090 → 5090. volume이 있는 데이터센터에 있는 GPU만 잡히며, 재고가 없으면 그 데이터센터의 GPU별 재고·가격 표가 출력되므로 거기서 골라 추가하면 됩니다(예: `"NVIDIA RTX PRO 4500 Blackwell"`) |
| `RUNPOD_S3_ENDPOINT`, `RUNPOD_S3_REGION` | (선택) S3 endpoint를 직접 지정. 비우면 `RUNPOD_DATACENTER`로 `https://s3api-{datacenter}.runpod.io`를 만듭니다 |
| `RUNPOD_CLOUD` | Network Volume은 Secure Cloud에서만 붙으므로 `SECURE` 유지 |
| `RUNPOD_DISK_GB` | 컨테이너 디스크. venv(약 15GB)와 압축을 푼 데이터가 들어갑니다 |
| `RUNPOD_IMAGE` | pod 이미지. torch 등은 `uv sync`가 `uv.lock`대로 다시 설치하므로 CUDA 12.8 계열 드라이버가 되는 이미지면 됩니다 |
| `STV_BRANCH` | pod가 clone할 브랜치. 본인 실험 브랜치로 바꿔도 됩니다 |

이미 셸에 export된 환경변수는 `.env`보다 우선합니다. 한 번만 다른 GPU로 돌리고 싶다면:

```bash
RUNPOD_GPU="NVIDIA A100 80GB PCIe" python scripts/runpod_launch.py run configs/big.toml
```

---

## 3. 데이터 업로드 (volume당 1회)

대회 데이터(`train.csv`, `dev.csv`, `test.csv`와 이미지 폴더)를 zip 하나로 묶어 volume에 올립니다.

```bash
bash scripts/s3.sh push-data /path/to/data.zip     # → volume의 stv/data.zip
bash scripts/s3.sh ls stv/                         # 올라갔는지 확인
```

zip 안에 최상위 폴더가 한 겹 더 있어도 됩니다(`setup.sh`가 `train.csv` 위치를 찾아 끌어올림). `aws` CLI가 없으면 `uvx`가 자동으로 받아 실행합니다.

---

## 4. 실험 실행

```bash
# 요청 내용만 미리 보기 (pod를 만들지 않음, 키는 ***로 가려짐)
python scripts/runpod_launch.py run configs/sample.toml --dry-run

# 실행. 실험 이름 = 설정 파일명(sample)
python scripts/runpod_launch.py run configs/sample.toml

# 실험 이름을 지정하고, "--" 뒤에 설정 덮어쓰기 플래그 전달
python scripts/runpod_launch.py run configs/sample.toml --name r32 -- --lora-r 32 --lora-alpha 32
```

실행하면 pod ID, 시간당 요금, 콘솔 링크, 결과 경로가 출력됩니다. **터미널을 닫아도 실험은 계속 돕니다.**

### 처음에는 smoke test부터

```bash
# 1) 학습 없이 zero-shot 추론 5문항: 키·volume·clone·wandb·저장·자동 삭제까지 약 4~8분(첫 실행은 패키지 설치 때문에 더 김), $0.1 안팎
python scripts/runpod_launch.py run configs/sample.toml --name smoke --infer-only -- --split dev --max-infer-samples 5 --n-tta 1
bash scripts/s3.sh log smoke                        # 마지막 줄이 "[job] exit code 0"이면 성공

# 2) 학습까지 포함한 짧은 실행
python scripts/runpod_launch.py run configs/sample.toml --name smoke-train -- --max-train-samples 100 --max-infer-samples 20 --eval-steps 5
```

`--infer-only`는 `stv.inference --adapter-dir none`만 실행합니다(split은 `-- --split dev`처럼 지정, 기본 test). zero-shot 기준 점수를 잴 때도 쓸 수 있습니다.

### GPU 재고가 없을 때

pod를 만들지 못하면 과금 없이 바로 끝나고, 실패 사실이 화면과 `launch.log`(repo root, git 제외)에 남습니다. pod 생성 성공도 같은 파일에 기록됩니다.

```
[실패] yeseo/r32: GPU 재고 없음 (NVIDIA GeForce RTX 4090, NVIDIA GeForce RTX 5090) → pod를 만들지 못했습니다 (과금 없음)
  데이터센터 EU-RO-1 (* = RUNPOD_GPU에 지정한 GPU)
    NVIDIA RTX PRO 4500 Blackwell                   32GB  $0.72/h   재고 Low
  * NVIDIA GeForce RTX 5090                         32GB  $0.99/h   재고 Low
  * NVIDIA GeForce RTX 4090                         24GB  -         재고 없음
```

재고 "Low"는 표시와 달리 생성이 실패할 수 있습니다. 잠시 뒤 다시 시도하거나, 표에 있는 다른 GPU를 한 번만 지정해 실행하세요: `RUNPOD_GPU="NVIDIA RTX PRO 4500 Blackwell" python scripts/runpod_launch.py run …`

### 주의: pod는 GitHub의 코드를 받습니다

pod는 내 PC의 파일이 아니라 **`STV_BRANCH`를 새로 clone**해서 실행합니다. 설정 파일이나 코드를 바꿨다면 **commit & push한 뒤** 실행하세요. 값 몇 개만 바꿔 보는 실험은 push 없이 `--` 뒤 플래그로 하면 됩니다.

### 설정 덮어쓰기 플래그

`src/stv/config.py`의 `Config` 필드가 모두 `--kebab-case` 플래그가 됩니다(우선순위: 플래그 > toml > 기본값). `--` 뒤의 플래그는 train·dev 추론·test 추론 세 단계에 똑같이 전달됩니다. `--output-dir`은 job이 volume 경로로 지정하므로 넣지 마세요.

| 플래그 | 기본 | 설명 |
|---|---|---|
| `--model-id` | `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit` | 8B 이상은 `--batch-size 1 --grad-accum 8` 권장 |
| `--lora-r`, `--lora-alpha` | 16, 16 | |
| `--epochs`, `--lr` | 1.0, 0(자동 1e-4) | `--init-adapter`가 있으면 lr 자동 5e-5 |
| `--batch-size`, `--grad-accum` | 2, 4 | 유효 배치 = 곱 |
| `--train-max-pixels`, `--infer-max-pixels` | 1048576 | 이미지 픽셀 예산 |
| `--prompt-variant` | `a` | `a` 기본 / `b` 글자 읽기 강조 / `c` 한국어 시스템 프롬프트 |
| `--use-dev`, `--dev-min-votes` | false, 3 | dev 다수결 라벨을 학습에 추가 |
| `--no-shuffle-options`, `--no-finetune-vision` | | bool 필드는 `--no-` 접두사로 끕니다 |
| `--eval-steps` | 150 | 이 step마다 검증 → 최고점 어댑터 저장 |
| `--n-tta`, `--infer-batch-size` | 2, 4 | 추론 TTA 횟수·배치 |
| `--max-train-samples`, `--max-infer-samples` | 0(전체) | 빠른 실험용 |
| `--seed` | 42 | `--valid-size`, `--split-seed`는 실험 간 비교를 위해 바꾸지 마세요 |

### pod 관리

```bash
python scripts/runpod_launch.py list             # 내 계정의 pod (ID, 상태, 시간당 요금, 이름)
python scripts/runpod_launch.py stop POD_ID      # pod 삭제
```

| `run` 옵션 | 설명 |
|---|---|
| `--name NAME` | 실험 이름. 같은 이름으로 다시 돌리면 같은 폴더·같은 wandb run에 이어 씁니다 → 새 실험은 새 이름으로 |
| `--infer-only` | 학습 없이 zero-shot 추론만 |
| `--keep` | 끝나도 pod를 삭제하지 않음(SSH로 들어가 디버깅할 때). **직접 `stop` 하지 않으면 계속 과금됩니다** |
| `--dry-run` | API 요청 내용만 출력 |

---

## 5. pod 안에서 일어나는 일

`scripts/runpod_job.sh`가 순서대로 실행합니다.

1. `git clone -b $STV_BRANCH` → `/root/stv`
2. `scripts/setup.sh /workspace/stv/data.zip`: `uv sync --frozen`(첫 실행 5~10분, 이후에는 volume의 uv 캐시로 단축) → 데이터를 컨테이너 로컬 디스크에 압축 해제
3. `scripts/run.sh CONFIG … --output-dir /workspace/stv/outputs/{STV_USER}/{실험 이름}`
   - `stv.train`: 학습. `eval_steps`마다 검증 500문항을 채점해 최고점 어댑터만 저장
   - `stv.inference --split dev`: dev 채점
   - `stv.inference --split test`: `submission_test.csv` 생성
4. 종료 코드를 `exit_code`에 쓰고 **성공·실패와 관계없이 pod 삭제**(`--keep` 제외)

HF 모델 캐시(`/workspace/.cache/huggingface`)와 uv 캐시는 volume에 남으므로 두 번째 실험부터는 모델을 다시 받지 않습니다.

---

## 6. 모니터링

**wandb** — `https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}`, run 이름 `{STV_USER}/{실험 이름}`, 태그 `{STV_USER}`·`runpod`

| 항목 | 내용 |
|---|---|
| `train/loss`, `train/learning_rate`, `train/grad_norm` | 10 step마다 |
| `valid/acc`, `valid/best_acc` | step 0(시작 모델)과 `eval_steps`마다 |
| summary `valid/start_acc`, `valid/best_step`, `valid/best_acc`, `valid/final_acc_tta{N}` | 학습 종료 시 |
| summary `dev/acc_tta{N}`, `dev/n` | dev 채점 후(같은 run에 이어 기록) |
| config | `Config` 전체 값 |

**로그** — pod가 살아 있든 삭제됐든 volume에서 읽습니다.

```bash
bash scripts/s3.sh log r32                 # run.log 마지막 40줄
LINES_N=200 bash scripts/s3.sh log r32     # 200줄
bash scripts/s3.sh log r32 minsu           # 다른 팀원(STV_USER=minsu)의 실험
```

---

## 7. 결과 받기

```bash
bash scripts/s3.sh ls                      # 내 실험 목록 (stv/outputs/{STV_USER}/)
bash scripts/s3.sh ls stv/outputs/         # 팀 전체
bash scripts/s3.sh pull r32                # → outputs/{STV_USER}/r32/
bash scripts/s3.sh pull r32 minsu          # 팀원 결과 → outputs/minsu/r32/
```

| 파일 | 내용 |
|---|---|
| `submission_test.csv` | **제출 파일** |
| `test_probs.npz`, `dev_probs.npz`, `valid_probs.npz` | 문항별 a/b/c/d 확률(`avg` [N,4], `ids`) → 앙상블용 |
| `submission_dev.csv`, `submission_valid.csv` | dev·valid 예측 |
| `adapter_model.safetensors`, `adapter_config.json`, 토크나이저·프로세서 파일 | 최고 검증 점수의 LoRA 어댑터 |
| `run.log` | pod 전체 출력(환경 구성 포함) |
| `log.txt` | step별 loss·valid_acc 요약 |
| `config.toml` | 실행에 쓴 설정 파일 사본(`--` 플래그로 덮어쓴 값은 `run.log` 첫 줄과 wandb config에서 확인) |
| `exit_code` | `0`이면 정상 종료 |
| `wandb_run_id.txt` | train·추론을 한 wandb run으로 잇는 ID |

그 외 S3 작업은 `aws` 명령을 그대로 쓸 수 있습니다(엔드포인트·키는 자동 설정).

```bash
bash scripts/s3.sh aws s3 rm s3://$RUNPOD_VOLUME_ID/stv/outputs/yeseo/smoke/ --recursive   # 실험 삭제
bash scripts/s3.sh aws s3 cp s3://$RUNPOD_VOLUME_ID/stv/outputs/yeseo/r32/submission_test.csv .
```

---

## 8. 문제 해결

| 증상 | 원인·조치 |
|---|---|
| `[실패] … GPU 재고 없음` | 그 데이터센터에 해당 GPU 재고 없음 → 함께 출력된 재고 표를 보고 `RUNPOD_GPU`에 대체 GPU를 추가하거나 잠시 뒤 재시도 ("GPU 재고가 없을 때" 참고) |
| `GitHub repo를 읽을 수 없습니다` (pod 생성 전) | `GITHUB_TOKEN`의 Resource owner가 조직이 아니거나 Contents 권한 없음, `STV_REPO`/`STV_BRANCH` 오타 |
| API는 되는데 volume이 404, S3는 `AccessDenied … not allowed for bucket` | 키와 volume이 서로 다른 계정(개인/팀) 소속. 같은 계정에서 다시 발급 |
| `→ 401` | `RUNPOD_API_KEY`가 틀렸거나 권한이 Read-only |
| pod가 뜨고 1~2분 뒤 사라짐 | pod 안에서 clone 실패. 이유는 `bash scripts/s3.sh log 이름`(clone 오류가 `run.log`에 남음) |
| `run.log`에 `data.zip 없음` | 3번(데이터 업로드)을 안 했거나 다른 volume에 올림 |
| `exit_code`가 0이 아님 | `LINES_N=200 bash scripts/s3.sh log 이름`으로 오류 확인. OOM이면 `-- --batch-size 1 --grad-accum 8` 또는 `--no-finetune-vision` |
| `s3.sh`가 403/SignatureDoesNotMatch | S3 키가 API 키와 다른 것인지 확인(`user_…`/`rps_…`), `RUNPOD_DATACENTER`가 volume 위치와 같은지 확인 |
| `s3.sh ls`에 방금 쓴 파일이 안 보임 | pod가 쓰는 중인 파일은 반영이 늦을 수 있음. 잠시 뒤 다시 |
| wandb에 run이 없음 | `WANDB_API_KEY` 미설정, 또는 `WANDB_ENTITY` 팀에 본인이 속해 있지 않음. wandb 오류는 실험을 멈추지 않고 `run.log`에 `⚠ wandb 초기화 실패`로만 남습니다 |
| `--keep` pod가 실험 후에도 과금 | 정상 동작. `list`로 확인하고 `stop` |

**보안**: `.env`는 절대 커밋하지 마세요. `GITHUB_TOKEN`은 pod 시작 명령에, 나머지 키는 pod 환경변수에 들어가 본인 RunPod 콘솔에서 보입니다. GitHub 토큰은 이 repo 읽기 전용으로만 발급하세요.

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `.env.example` | `.env` 템플릿 |
| `scripts/runpod_launch.py` | 내 PC에서 실행: pod 생성(`run`)·목록(`list`)·삭제(`stop`) |
| `scripts/runpod_job.sh` | pod 안에서 실행: 환경 구성 → 실험 → pod 삭제 |
| `scripts/s3.sh` | 내 PC에서 실행: volume 업로드·목록·로그·다운로드 |
| `scripts/setup.sh`, `scripts/run.sh`, `scripts/env.sh` | 환경 구성, train→dev→test, 환경 감지·캐시 경로 |
| `configs/sample.toml` | 실험 설정 예시(필드 설명 포함) |
| `src/stv/tracking.py` | wandb 기록(`WANDB_API_KEY`가 없으면 아무 일도 하지 않음) |
