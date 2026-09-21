#!/usr/bin/env bash
# train → dev 채점 → test 제출 파일 생성. 추가 인자는 세 단계 모두에 전달됩니다.
#
#   bash scripts/run.sh configs/sample.toml
#   bash scripts/run.sh configs/sample.toml --max-train-samples 0 --output-dir outputs/full
#   nohup bash scripts/run.sh configs/sample.toml > run.log 2>&1 &   # RunPod: 터미널 끊겨도 유지
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

CONFIG=${1:?usage: run.sh CONFIG [extra flags...]}
shift

uv run stv-train --config "$CONFIG" "$@"
uv run stv-infer --config "$CONFIG" "$@" --split dev
uv run stv-infer --config "$CONFIG" "$@" --split test
