#!/usr/bin/env bash
# train(검증 포함) → dev 채점 → test 제출 파일 생성. 추가 인자는 세 단계 모두에 전달됩니다.
#
#   bash scripts/run.sh configs/sample.toml
#   bash scripts/run.sh configs/sample.toml --max-train-samples 1000 --output-dir outputs/quick
#   nohup bash scripts/run.sh configs/sample.toml > run.log 2>&1 &   # RunPod: 터미널 끊겨도 유지
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

CONFIG=${1:?usage: run.sh CONFIG [extra flags...]}
shift

$STV_PY -m stv.train --config "$CONFIG" "$@"
$STV_PY -m stv.inference --config "$CONFIG" "$@" --split dev
$STV_PY -m stv.inference --config "$CONFIG" "$@" --split test
