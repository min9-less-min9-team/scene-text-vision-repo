#!/usr/bin/env bash
# RunPod pod 안에서 실행됩니다 (scripts/runpod_launch.py가 pod 시작 명령으로 호출).
# 환경 구성 → train → dev 채점 → test 제출 파일. 결과는 Network Volume(/workspace = S3 bucket)에 바로 기록되고,
# 끝나면 성공·실패와 관계없이 pod를 삭제해 과금을 멈춥니다 (STV_KEEP_POD=1이면 유지).
#
#   bash scripts/runpod_job.sh configs/sample.toml [extra flags...]
set -uo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:?usage: runpod_job.sh CONFIG [extra flags...]}
shift
USER_NAME=${STV_USER:-anon}
RUN_NAME=${STV_RUN_NAME:-$(basename "$CONFIG" .toml)}
VOLUME=/workspace/stv
OUT=$VOLUME/outputs/$USER_NAME/$RUN_NAME
mkdir -p "$OUT"

finish() {
    code=$?
    echo "[job] exit code $code $(date -Is)" | tee -a "$OUT/run.log"
    echo "$code" > "$OUT/exit_code"
    if [ "${STV_KEEP_POD:-0}" != "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
        sync
        runpodctl remove pod "$RUNPOD_POD_ID" \
            || curl -s -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY"
    fi
    # 컨테이너가 끝나면 RunPod가 시작 명령을 다시 실행하므로(=실험 재시작) 삭제될 때까지 대기
    sleep infinity
}
trap finish EXIT

{
    set -e
    echo "[job] $USER_NAME/$RUN_NAME  config=$CONFIG  extra=$*  commit=$(git rev-parse --short HEAD)  $(date -Is)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    [ -e "$VOLUME/data.zip" ] || { echo "[job] $VOLUME/data.zip 없음 → bash scripts/s3.sh push-data data.zip"; exit 2; }
    bash scripts/setup.sh "$VOLUME/data.zip"
    cp "$CONFIG" "$OUT/config.toml"
    if [ "${STV_MODE:-full}" = "infer" ]; then
        # 학습 없이 zero-shot 추론만 (파이프라인 점검용). split은 extra flags의 --split, 기본 test
        source scripts/env.sh
        $STV_PY -m stv.inference --config "$CONFIG" "$@" --adapter-dir none --output-dir "$OUT"
    else
        bash scripts/run.sh "$CONFIG" "$@" --output-dir "$OUT"
    fi
} 2>&1 | tee -a "$OUT/run.log"
exit "${PIPESTATUS[0]}"
