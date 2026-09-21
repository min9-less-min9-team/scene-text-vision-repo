# source this file. Detects Colab / RunPod / local and points caches at persistent storage.
if [ -e /opt/lab/start-lab.py ]; then
    # Docker Jupyter lab 이미지: torch/unsloth가 시스템 python에 이미 설치돼 있어 uv venv를 쓰지 않음.
    # HF_HOME / HF_ENDPOINT는 컨테이너가 설정한 값을 그대로 사용
    export STV_ENV=lab
    export STV_PY="python"
    # 미러(HF_ENDPOINT)가 닿지 않으면 재시도로 오래 멈추므로 캐시만 사용. 온라인으로 받으려면 HF_HUB_OFFLINE=0
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
elif [ -d /content ] && python3 -c "import google.colab" 2>/dev/null; then
    export STV_ENV=colab
    # Drive가 마운트돼 있으면 모델 캐시를 Drive에 둬서 세션이 바뀌어도 재다운로드하지 않음
    if [ -d /content/drive/MyDrive ]; then
        export HF_HOME=${HF_HOME:-/content/drive/MyDrive/stv/hf_cache}
    fi
elif [ -n "${RUNPOD_POD_ID:-}" ] || [ -d /workspace ]; then
    export STV_ENV=runpod
    # /workspace = network volume (pod 재시작 후에도 유지). 컨테이너 디스크는 작아서 캐시를 옮김
    export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
    export UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/.cache/uv}
else
    export STV_ENV=local
fi
export STV_PY=${STV_PY:-"uv run python"}
export PATH="$HOME/.local/bin:$PATH"
