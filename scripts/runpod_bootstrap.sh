#!/usr/bin/env bash
# RunPod pod가 켜질 때 백그라운드로 한 번 실행됩니다 (scripts/runpod.py up이 pod 시작 명령에 넣음).
# Jupyter는 이 스크립트와 별개로 이미지 기본 /start.sh가 띄우므로, 이 스크립트는 개발 환경만 준비합니다.
#   1. git 인증 설정 (GITHUB_TOKEN 환경변수 → clone/push 모두 사용)
#   2. venv (uv sync). venv는 컨테이너 디스크(/root/.venv-stv)에 두어 import가 느린 network volume을 피함
#   3. Jupyter 커널 등록 (stv) — notebooks/baseline.ipynb 가 이 커널을 사용
#   4. .bashrc: 터미널을 열면 repo로 이동 + scripts/env.sh
# 데이터(/workspace/stv/data.zip)는 노트북 첫 셀이 직접 풉니다.
# 진행 상황은 repo 옆의 bootstrap.log 에 남고, 끝나면 마지막 줄이 "[bootstrap] done" 입니다.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
source scripts/env.sh

echo "[bootstrap] start $(date -Is)  repo=$REPO  commit=$(git rev-parse --short HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
[ -e /workspace/stv/data.zip ] || echo "[bootstrap] WARNING: /workspace/stv/data.zip 없음 → 내 PC에서: bash scripts/s3.sh push-data data.zip"

# 1. git: 토큰을 URL이 아니라 credential helper로 전달 (volume에 남는 .git/config에 토큰이 안 들어감)
if [ -n "${GITHUB_TOKEN:-}" ]; then
    git config --global credential.helper '!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f'
fi
git config --global --add safe.directory "$REPO"
[ -n "$(git config --global user.name)" ] || git config --global user.name "${STV_USER:-runpod}"
[ -n "$(git config --global user.email)" ] || git config --global user.email "${STV_USER:-runpod}@runpod"

# 2. venv
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen --group eda
$STV_PY -c "import torch; print('[bootstrap] torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 3. Jupyter 커널
$STV_PY -m ipykernel install --user --name stv --display-name "stv (uv venv)" || echo "[bootstrap] WARNING: 커널 등록 실패"

# 4. 터미널 기본 환경
if ! grep -q "scripts/env.sh" /root/.bashrc 2>/dev/null; then
    printf '\n# stv: repo로 이동 + 환경변수\ncd %q && source scripts/env.sh\n' "$REPO" >> /root/.bashrc
fi

echo "[bootstrap] done $(date -Is)"
