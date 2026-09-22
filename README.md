# Scene-Text VQA — RunPod 개발 환경 가이드

실험 코드는 **`notebooks/baseline.ipynb`** 하나입니다(Qwen3-VL + Unsloth QLoRA, 720×960 고정, 정답 글자만 loss + label smoothing). `.env`에 본인 키를 넣고 `up`을 실행하면 RunPod에서 GPU pod가 뜨고 **웹 Jupyter**로 접속해 노트북을 실행합니다. 끝나면 `down`으로 pod를 지워 과금을 멈춥니다. repo 작업본·결과·모델 캐시는 Network Volume(`/workspace`)에 남아 다음 `up` 때 그대로 이어집니다.

```
내 PC                              RunPod pod (GPU)                                   저장소
─────                              ────────────────                                   ──────
runpod.py up   ───────────────▶    /start.sh: Jupyter(8888)
                                   bootstrap: repo clone → uv sync → 커널 등록
브라우저       ◀──────────────▶    notebooks/baseline.ipynb 실행 ─────────────────▶ wandb (loss, valid/acc)
                                   결과 → /workspace/stv/outputs/{STV_USER}/  ─────▶ Network Volume (= S3 bucket)
runpod.py down ───────────────▶    pod 삭제 (volume은 유지)
s3.sh pull     ◀──────────────────────────── S3 API ◀───────────────────────────────
```

내 PC에는 GPU가 필요 없습니다. [`uv`](https://docs.astral.sh/uv/)와 `bash`만 있으면 됩니다(Windows는 WSL).

---

## 1. 준비물 (최초 1회)

### 1-1. 각자 발급하는 키

| `.env` 변수 | 발급 위치 | 비고 |
|---|---|---|
| `RUNPOD_API_KEY` | RunPod Console → Settings → **API Keys** → Create | pod 생성·삭제 권한이 필요하므로 Read/Write(또는 All) |
| `RUNPOD_S3_ACCESS_KEY`, `RUNPOD_S3_SECRET_KEY` | RunPod Console → Settings → **S3 API Keys** → Create | access key는 `user_…`, secret은 `rps_…`. secret은 생성 직후에만 보입니다 |
| `WANDB_API_KEY` | https://wandb.ai/authorize | 비워 두면 노트북이 wandb 기록 없이 실행됩니다 |
| `GITHUB_TOKEN` | GitHub → Settings → Developer settings → Fine-grained token | pod 안에서 repo를 clone하고 **push**하는 데 씁니다. **Resource owner를 조직 `min9-less-min9-team`으로** 선택(개인 계정으로 두면 403) → Repository access에서 이 repo → Permissions **Contents: Read and write** |
| `JUPYTER_PASSWORD` | 직접 정함 | Jupyter 접속 토큰. 비우면 `up` 때마다 무작위로 만들어 출력합니다 |
| `HF_TOKEN` (선택) | https://huggingface.co/settings/tokens | gated 모델을 쓰거나 다운로드 제한에 걸릴 때만 |

> **키와 volume은 같은 RunPod 계정에서 만들어야 합니다.** 콘솔 좌상단에서 개인 계정/팀 계정을 전환할 수 있는데, 개인 계정에서 만든 API 키·S3 키로는 팀 계정의 volume이 보이지 않습니다(API 404, S3 `AccessDenied … not allowed for bucket`). 과금도 키를 만든 계정으로 나갑니다.

토큰이 맞는지는 pod를 띄우기 전에 확인할 수 있습니다(`up`도 pod 생성 전에 같은 검사를 합니다).

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
JUPYTER_PASSWORD=mysecret
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
| `STV_USER` | 결과가 `stv/outputs/{STV_USER}/`에 저장되므로 팀원끼리 겹치지 않게 |
| `RUNPOD_GPU` | RunPod GPU type ID. 쉼표로 여러 개를 쓰면 앞에서부터 재고가 있는 것을 잡습니다. 값에 공백이 있으므로 **따옴표 필수**. 기본은 4090 → 5090. volume이 있는 데이터센터에 있는 GPU만 잡히며, 재고가 없으면 그 데이터센터의 GPU별 재고·가격 표가 출력되므로 거기서 골라 추가하면 됩니다(예: `"NVIDIA RTX PRO 4500 Blackwell"`) |
| `RUNPOD_S3_ENDPOINT`, `RUNPOD_S3_REGION` | (선택) S3 endpoint를 직접 지정. 비우면 `RUNPOD_DATACENTER`로 `https://s3api-{datacenter}.runpod.io`를 만듭니다 |
| `RUNPOD_CLOUD` | Network Volume은 Secure Cloud에서만 붙으므로 `SECURE` 유지 |
| `RUNPOD_DISK_GB` | 컨테이너 디스크. venv(약 15GB)와 압축을 푼 데이터·리사이즈 이미지가 들어갑니다. **pod를 지우면 사라집니다**(volume `/workspace`만 유지) |
| `RUNPOD_IMAGE` | pod 이미지. torch 등은 `uv sync`가 `uv.lock`대로 다시 설치하므로 CUDA 12.8 계열 드라이버가 되는 이미지면 됩니다 |
| `STV_BRANCH` | 처음 `up`할 때 clone할 브랜치. 이후에는 pod 안에서 `git checkout`으로 바꿉니다 |

이미 셸에 export된 환경변수는 `.env`보다 우선합니다. 한 번만 다른 GPU로 돌리고 싶다면:

```bash
RUNPOD_GPU="NVIDIA A100 80GB PCIe" uv run scripts/runpod.py up
```

---

## 3. 데이터 업로드 (volume당 1회)

대회 데이터(`train.csv`, `dev.csv`, `test.csv`, `sample_submission.csv`와 이미지 폴더)를 zip 하나로 묶어 volume에 올립니다.

```bash
bash scripts/s3.sh push-data /path/to/data.zip     # → volume의 stv/data.zip
bash scripts/s3.sh ls stv/                         # 올라갔는지 확인
```

zip 안에 최상위 폴더가 한 겹 더 있어도 됩니다(노트북이 `train.csv` 위치를 찾습니다). `aws` CLI가 없으면 `uvx`가 자동으로 받아 실행합니다.

---

## 4. pod 켜기 → 노트북 실행 → 끄기

```bash
uv run scripts/runpod.py up --dry-run    # 요청 내용만 미리 보기 (pod를 만들지 않음, 키는 ***)
uv run scripts/runpod.py up --open       # pod 생성. Jupyter가 응답할 때까지 기다린 뒤 브라우저로 열기 (1~3분)
```

```
pod abcd1234  RUNNING  NVIDIA GeForce RTX 4090  $0.59/h
  콘솔     https://console.runpod.io/pods?id=abcd1234
  Jupyter  https://abcd1234-8888.proxy.runpod.net/lab?token=mysecret
  repo     /workspace/stv/home/yeseo/scene-text-vision-repo   (준비 로그: bootstrap.log, 끝나면 '[bootstrap] done')
  종료     uv run scripts/runpod.py down   ← 잊으면 계속 과금됩니다
```

| 명령 | 설명 |
|---|---|
| `up` | pod 생성. 사용자당 1개(`stv-{STV_USER}`)만 만들며, 이미 있으면 그 접속 정보를 보여줍니다. `--open`이면 준비되는 대로 Jupyter를 브라우저로 엽니다 |
| `jupyter` | 내 pod의 Jupyter URL 출력 + 브라우저로 열기 |
| `status` | 접속 정보 다시 보기 |
| `down [POD_ID]` | pod 삭제 = 과금 중지. ID를 생략하면 내 pod |
| `list` | 계정의 모든 pod (팀원 것 포함) |

### 접속

출력된 Jupyter URL을 브라우저에서 열면 됩니다. Jupyter는 RunPod HTTP 프록시로 열리고 인증은 URL의 `token=`(`.env`의 `JUPYTER_PASSWORD`)입니다. API 키는 pod 생성·삭제에만 쓰이며, SSH는 필요 없습니다.

환경 준비(bootstrap: `uv sync` + 커널 등록)는 pod가 뜨고 나서 백그라운드로 도니, 먼저 끝났는지 확인하세요. Jupyter 파일 탐색기에서 `/workspace/stv/home/{STV_USER}/bootstrap.log`를 열거나, Launcher → Terminal에서:

```bash
tail -f ../bootstrap.log       # 마지막 줄이 "[bootstrap] done"이면 준비 완료 (첫 실행 5~10분, 이후 volume의 uv 캐시로 1~2분)
```

### 노트북 실행

`/workspace/stv/home/{STV_USER}/scene-text-vision-repo/notebooks/baseline.ipynb`를 열고 커널이 **`stv (uv venv)`** 인지 확인한 뒤(bootstrap이 등록, 노트북 기본값) 위에서부터 실행합니다. 노트북은 RunPod를 자동 감지해 경로를 잡습니다.

| 노트북이 쓰는 경로 | 내용 | pod 삭제 후 |
|---|---|---|
| `/workspace/stv/data.zip` | 3번에서 올린 데이터 | 유지 |
| `/root/data/raw`, `/root/data/resize` | 압축 해제·720×960 리사이즈 (첫 셀, 1~3분) | 삭제 → 다음 `up`에서 첫 셀이 다시 만듦 |
| `/workspace/.cache/huggingface` | 모델 캐시. 처음 한 번만 내려받음 | 유지 |
| `/workspace/stv/outputs/{STV_USER}/` | `OUTPUT_DIR`: 어댑터(`*_best/`), 확률(`probs_*.npy`), 제출(`submission*.csv`), 로그(`*_log.txt`), `experiments.csv` | 유지 |

- **설정**은 "설정" 셀 하나에서 바꿉니다(`LR`, `LORA_R`, `LABEL_SMOOTHING`, …). `RUN_NAME`이 설정 해시로 자동 생성되어 실험별 결과가 따로 저장됩니다.
- **빠른 점검**: 설정 셀의 `QUICK_RUN = True`(또는 환경변수 `VQA_QUICK=1`)로 전체 흐름이 몇 분 안에 도는지 먼저 확인하세요.
- **wandb**: `WANDB_API_KEY`가 있으면 자동으로 켜집니다. run 이름 `{STV_USER}/{RUN_NAME}`, 태그 `{STV_USER}`·`runpod`. `train/loss`·`train/learning_rate`(10 step), `valid/acc`·`valid/best_acc`(step 0과 `EVAL_STEPS`마다), summary에 `valid/best_step`·`valid/final_acc_tta{N}`·`test/pred_dist`. 초기화에 실패해도 실험은 계속됩니다.
- **브라우저를 닫아도** 커널은 계속 돕니다(Jupyter 서버가 pod 안에 있음). 다시 열면 출력이 이어집니다. 단 pod를 `down`하면 커널도 끝납니다.
- 코드를 고쳤다면 Terminal에서 `git add/commit/push` 하세요(`GITHUB_TOKEN`으로 push 가능). commit하지 않은 변경도 repo가 volume에 있어 남지만, volume은 공유·실수 삭제 가능성이 있으니 남길 코드는 push하세요.

### 끝나면 반드시

```bash
uv run scripts/runpod.py down          # pod 삭제. volume의 repo 작업본·결과·모델 캐시는 그대로
```

### GPU 재고가 없을 때

pod를 만들지 못하면 과금 없이 바로 끝나고, 실패 사실이 화면과 `launch.log`(repo root, git 제외)에 남습니다.

```
[실패] yeseo: GPU 재고 없음 (NVIDIA GeForce RTX 4090, NVIDIA GeForce RTX 5090) → pod를 만들지 못했습니다 (과금 없음)
  데이터센터 EU-RO-1 (* = RUNPOD_GPU에 지정한 GPU)
    NVIDIA RTX PRO 4500 Blackwell                   32GB  $0.72/h   재고 Low
  * NVIDIA GeForce RTX 5090                         32GB  $0.99/h   재고 Low
  * NVIDIA GeForce RTX 4090                         24GB  -         재고 없음
```

재고 "Low"는 표시와 달리 생성이 실패할 수 있습니다. 잠시 뒤 다시 시도하거나, 표에 있는 다른 GPU를 한 번만 지정해 실행하세요: `RUNPOD_GPU="NVIDIA RTX PRO 4500 Blackwell" uv run scripts/runpod.py up`

---

## 5. pod 안의 구성

`up`은 이미지 기본 시작 스크립트(`/start.sh`: Jupyter)를 그대로 쓰고, 그 앞에서 `scripts/runpod_bootstrap.sh`를 백그라운드로 실행합니다.

| 경로 | 내용 | pod 삭제 후 |
|---|---|---|
| `/workspace/stv/home/{STV_USER}/scene-text-vision-repo` | repo 작업본. 없을 때만 `STV_BRANCH`를 clone, 있으면 그대로 둠 | 유지 |
| `/workspace/stv/home/{STV_USER}/bootstrap.log` | 환경 준비 로그 | 유지 |
| `/workspace/.cache/uv` | 패키지 캐시 → 두 번째부터 `uv sync`가 빠름 | 유지 |
| `/root/.venv-stv` | venv (`uv sync --frozen --group eda`). network volume에서 import가 느려 컨테이너 디스크에 둠 | 삭제 → 다음 `up`에서 재생성 |

bootstrap은 이 밖에 git credential(`GITHUB_TOKEN`으로 push 가능), Jupyter 커널 `stv (uv venv)`, `.bashrc`(터미널을 열면 repo로 이동 + `scripts/env.sh`)를 설정합니다.

---

## 6. 결과 받기

```bash
bash scripts/s3.sh ls                      # 내 결과 (stv/outputs/{STV_USER}/)
bash scripts/s3.sh ls stv/outputs/         # 팀 전체
bash scripts/s3.sh pull-sub                # 제출 파일(submission*.csv)만 → outputs/{STV_USER}/
bash scripts/s3.sh pull                    # 결과 전체 (어댑터·trainer·wandb 제외) → outputs/{STV_USER}/
bash scripts/s3.sh pull minsu              # 팀원(STV_USER=minsu) 결과 → outputs/minsu/
```

| 파일 | 내용 |
|---|---|
| `submission.csv` | **제출 파일** (가장 최근 전체 실행) / `submission_{RUN_NAME}_{INFER_TAG}.csv` 실험별 사본 |
| `probs_test_*.npy`, `probs_valid_*.npy` | 문항별 a/b/c/d 확률 → 앙상블용 (노트북 마지막 셀) |
| `{RUN_NAME}_best/` | 최고 검증 점수의 LoRA 어댑터 + 프로세서 |
| `{RUN_NAME}_log.txt` | step별 loss·valid_acc |
| `experiments.csv` | 실험 설정·점수 한 줄씩 누적 |
| `label_suspects_{RUN_NAME}.csv` | 모델이 높은 확신으로 라벨과 다르게 답한 검증 문항 |

그 외 S3 작업은 `aws` 명령을 그대로 쓸 수 있습니다(엔드포인트·키는 자동 설정).

```bash
bash scripts/s3.sh aws s3 rm s3://$RUNPOD_VOLUME_ID/stv/outputs/yeseo/old_run_best/ --recursive   # 어댑터 삭제
```

---

## 7. 문제 해결

| 증상 | 원인·조치 |
|---|---|
| `[실패] … GPU 재고 없음` | 그 데이터센터에 해당 GPU 재고 없음 → 함께 출력된 재고 표를 보고 `RUNPOD_GPU`에 대체 GPU를 추가하거나 잠시 뒤 재시도 |
| `GitHub repo를 읽을 수 없습니다` (pod 생성 전) | `GITHUB_TOKEN`의 Resource owner가 조직이 아니거나 Contents 권한 없음, `STV_REPO`/`STV_BRANCH` 오타 |
| API는 되는데 volume이 404, S3는 `AccessDenied … not allowed for bucket` | 키와 volume이 서로 다른 계정(개인/팀) 소속. 같은 계정에서 다시 발급 |
| `→ 401` | `RUNPOD_API_KEY`가 틀렸거나 권한이 Read-only |
| Jupyter URL이 404/502 | pod가 아직 뜨는 중(이미지 pull 1~3분)이면 RunPod 프록시가 404/502를 돌려줍니다. 잠시 뒤 `uv run scripts/runpod.py jupyter`. 오래 계속되면 콘솔에서 pod 로그 확인 |
| Jupyter가 토큰을 물어봄 | URL의 `token=`이 빠짐. `.env`의 `JUPYTER_PASSWORD` 값 입력(비워 뒀다면 `up` 출력에 있던 값) |
| Jupyter에 `stv (uv venv)` 커널이 없음 | bootstrap이 아직 안 끝남 → `bootstrap.log` 확인. 끝났는데도 없으면 브라우저 새로고침 |
| 노트북 첫 셀 `dataset zip not found` | 3번(데이터 업로드)을 안 했거나 다른 volume에 올림 |
| `bootstrap.log`가 `[bootstrap] FAILED`로 끝남 | 바로 위 줄이 원인(주로 clone 실패 = 토큰). 고친 뒤 Terminal에서 `bash scripts/runpod_bootstrap.sh` |
| pod 안에서 `git push`가 403 | `GITHUB_TOKEN`이 Read-only. Contents: Read and write로 재발급 → `down` → `up` |
| OOM | 설정 셀에서 `BATCH_SIZE=1`, `GRAD_ACCUM=8`(유효 배치 유지) 또는 `FINETUNE_VISION=False` |
| `s3.sh`가 403/SignatureDoesNotMatch | S3 키가 API 키와 다른 것인지 확인(`user_…`/`rps_…`), `RUNPOD_DATACENTER`가 volume 위치와 같은지 확인 |
| wandb에 run이 없음 | `WANDB_API_KEY` 미설정, 또는 `WANDB_ENTITY` 팀에 본인이 속해 있지 않음. 설정 셀 출력에 `⚠ wandb 초기화 실패` 이유가 찍힙니다. `QUICK_RUN`은 기록하지 않습니다 |
| 실험이 끝났는데 과금이 계속됨 | pod는 스스로 꺼지지 않습니다. `uv run scripts/runpod.py list`로 확인하고 `down` |

**보안**: `.env`는 절대 커밋하지 마세요. 전달한 키들은 pod 환경변수에 들어가 본인 RunPod 콘솔에서 보입니다. GitHub 토큰은 이 repo에만 권한을 주세요. pod 안의 repo는 원격 URL에 토큰을 넣지 않고 credential helper가 환경변수 `GITHUB_TOKEN`을 읽는 방식이라, volume에 남는 `.git/config`에는 토큰이 들어가지 않습니다.

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `notebooks/baseline.ipynb` | 실험 노트북 (RunPod/로컬 자동 감지, wandb) |
| `.env.example` | `.env` 템플릿 |
| `scripts/runpod.py` | 내 PC에서 실행: pod 생성(`up`)·Jupyter 열기(`jupyter`)·접속 정보(`status`)·삭제(`down`)·목록(`list`) |
| `scripts/runpod_bootstrap.sh` | pod 안에서 자동 실행: repo clone → venv → Jupyter 커널 |
| `scripts/s3.sh` | 내 PC에서 실행: volume 업로드·목록·다운로드 |
| `scripts/env.sh` | 환경 감지·캐시 경로 (pod 터미널에서 자동 적용) |
| `pyproject.toml`, `uv.lock` | 노트북 의존성 (`uv sync --frozen`) |
| `eda/`, `docs/` | EDA 결과·계획 문서 |
