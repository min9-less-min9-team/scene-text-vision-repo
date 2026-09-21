#!/usr/bin/env bash
# 환경 구성 (Colab / RunPod / local 공통): 런타임 준비 → 데이터 연결
#
#   bash scripts/setup.sh                      # 환경만
#   bash scripts/setup.sh /path/to/data.zip    # zip을 data/에 풀기 (Colab Drive는 이 방식 권장)
#   bash scripts/setup.sh /path/to/data_dir    # 디렉터리를 data/로 symlink
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
echo "[setup] env=$STV_ENV HF_HOME=${HF_HOME:-default}"

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv sync --frozen
DEFAULT_DATA=

DATA_SRC=${1:-$DEFAULT_DATA}
if [ -n "$DATA_SRC" ] && [ ! -e data/train.csv ]; then
    if [ -d "$DATA_SRC" ]; then
        ln -sfn "$(realpath "$DATA_SRC")" data
    else
        mkdir -p data
        # 로컬 디스크에 풀어야 이미지 로딩이 빠름 (Drive에서 직접 읽으면 매우 느림)
        python3 -m zipfile -e "$DATA_SRC" data
        # zip 안에 최상위 폴더가 하나 더 있으면 끌어올림
        if [ ! -e data/train.csv ]; then
            inner=$(dirname "$(find data -maxdepth 2 -name train.csv | head -1)")
            [ -n "$inner" ] && [ "$inner" != "." ] && mv "$inner"/* data/
        fi
    fi
fi

[ -e data/train.csv ] && echo "[setup] data ok: $(ls data | tr '\n' ' ')" || echo "[setup] WARNING: data/train.csv 없음"
$STV_PY -c "import torch; print('[setup] torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
