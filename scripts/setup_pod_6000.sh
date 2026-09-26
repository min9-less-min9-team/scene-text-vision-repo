#!/usr/bin/env bash
# pod 시작 시 1회 실행: venv·데이터는 pod 로컬(/root), data.zip·모델 캐시·결과물은 /workspace
#   사용법: bash /workspace/stv/setup_pod.sh qwen   (또는 gemma / hcx)
#   끝나면: source /root/stv_env.sh
set -eo pipefail

MODEL="${1:?사용법: bash setup_pod.sh qwen|gemma|hcx}"
case "$MODEL" in qwen|gemma|hcx) ;; *) echo "qwen|gemma|hcx 중 하나"; exit 1 ;; esac

ZIP=/workspace/stv/data.zip
export UV_PYTHON_INSTALL_DIR=/root/.uv/python
export UV_CACHE_DIR=/root/.uv/cache
export HF_HOME=/workspace/.cache/huggingface      # 모델 가중치는 볼륨에서 읽음 (큰 파일 순차 읽기라 괜찮음)
export STV_DATA_ROOT=/root/data                   # 학습 데이터: 로컬
export STV_EXT_DIR=/root/external                 # 외부 데이터(KRETA/MTVQA) 변환 캐시: 로컬

if [ "$MODEL" = "hcx" ]; then VENV=/root/.venv-hcx; else VENV=/root/.venv-stv; fi
PYX="$VENV/bin/python"

echo "===== [1/4] uv + python 3.11 ====="
command -v uv >/dev/null 2>&1 || pip install -q uv
uv python install 3.11

echo "===== [2/4] venv: $VENV ====="
if [ ! -x "$PYX" ]; then
  uv venv "$VENV" --python 3.11 --python-preference only-managed
fi
if [ "$MODEL" = "hcx" ]; then
  command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }
  uv pip install --python "$PYX" torch torchvision --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python "$PYX" "git+https://github.com/huggingface/transformers" \
      peft bitsandbytes accelerate datasets pandas pillow tqdm wandb packaging
else
  uv pip install --python "$PYX" torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python "$PYX" unsloth==2026.9.7 transformers==5.5.0 peft==0.21.0 bitsandbytes==0.50.2 \
      datasets pandas pillow tqdm wandb packaging
  # unsloth 설치가 torch를 바꿨으면 되돌림
  "$PYX" -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.11.0') else 1)" || \
    uv pip install --python "$PYX" --reinstall torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
fi

echo "===== [3/4] 데이터: $ZIP → $STV_DATA_ROOT/raw ====="
test -f "$ZIP" || { echo "없음: $ZIP"; exit 1; }
if [ -z "$(find "$STV_DATA_ROOT/raw" -name train.csv 2>/dev/null | head -1)" ]; then
  mkdir -p "$STV_DATA_ROOT/raw"
  "$PYX" -c "import zipfile; zipfile.ZipFile('$ZIP').extractall('$STV_DATA_ROOT/raw')"
fi
RAW="$(dirname "$(find "$STV_DATA_ROOT/raw" -name train.csv | head -1)")"
echo "데이터 폴더: $RAW"
ls "$RAW" | grep -E "\.csv$"

echo "===== [4/4] 환경 확인 ====="
"$PYX" -c "import torch, torch.distributed as d; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| cap', torch.cuda.get_device_capability(), '| dist', d.is_available())"
if [ "$MODEL" = "hcx" ]; then
  "$PYX" -c "import transformers as t; ok = hasattr(t, 'HyperCLOVAXVisionV2ForConditionalGeneration'); print('transformers', t.__version__, '| HCX class', ok); assert ok"
else
  "$PYX" -c "import unsloth, transformers, peft, bitsandbytes; print('unsloth', unsloth.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| bnb', bitsandbytes.__version__)"
fi

cat > /root/stv_env.sh << EOF
export UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR
export UV_CACHE_DIR=$UV_CACHE_DIR
export HF_HOME=$HF_HOME
export STV_DATA_ROOT=$STV_DATA_ROOT
export STV_EXT_DIR=$STV_EXT_DIR
export PY=$PYX
EOF
echo
echo "완료. 다음 명령으로 환경 변수 적용:  source /root/stv_env.sh   (PY=$PYX)"
