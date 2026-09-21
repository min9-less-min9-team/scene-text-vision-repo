# source this file. Detects Colab / RunPod / local and points caches at persistent storage.
if [ -d /content ] && python3 -c "import google.colab" 2>/dev/null; then
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
export PATH="$HOME/.local/bin:$PATH"
