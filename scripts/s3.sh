#!/usr/bin/env bash
# RunPod Network Volume을 S3 API로 다룹니다 (pod 없이 로컬에서 업로드·다운로드). 설정은 .env.
#
#   bash scripts/s3.sh pull-ckpt <결과폴더> [하위경로]   # LoRA/checkpoint를 받아 zip으로 (기본 하위경로: adapter)
#   bash scripts/s3.sh push-data data.zip     # 데이터 업로드 (최초 1회) → stv/data.zip
#   bash scripts/s3.sh ls [경로]              # 기본: 내 결과 폴더 stv/outputs/{STV_USER}/
#   bash scripts/s3.sh pull [사용자]          # 결과 폴더 전체를 outputs/{사용자}/ 로 내려받기 (어댑터·trainer 제외)
#   bash scripts/s3.sh pull-sub [사용자]      # 제출 파일(submission*.csv)만
#   bash scripts/s3.sh aws ...                # 그대로 aws s3 명령 (예: aws s3 rm s3://$RUNPOD_VOLUME_ID/stv/outputs/x --recursive)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo ".env가 없습니다: cp .env.example .env"; exit 1; }
set -a; source .env; set +a
: "${RUNPOD_S3_ACCESS_KEY:?}" "${RUNPOD_S3_SECRET_KEY:?}" "${RUNPOD_VOLUME_ID:?}" "${STV_USER:?}"
# 엔드포인트: RUNPOD_S3_ENDPOINT(콘솔의 volume S3 정보에 표시된 URL)가 있으면 그대로, 없으면 데이터센터 ID로 조합
[ -n "${RUNPOD_S3_ENDPOINT:-}" ] || : "${RUNPOD_DATACENTER:?RUNPOD_S3_ENDPOINT 또는 RUNPOD_DATACENTER가 필요합니다}"

export AWS_ACCESS_KEY_ID=$RUNPOD_S3_ACCESS_KEY AWS_SECRET_ACCESS_KEY=$RUNPOD_S3_SECRET_KEY
DC=$(echo "${RUNPOD_DATACENTER:-}" | tr '[:upper:]' '[:lower:]')
ENDPOINT=${RUNPOD_S3_ENDPOINT:-https://s3api-$DC.runpod.io}
REGION=${RUNPOD_S3_REGION:-${RUNPOD_DATACENTER:-us-east-1}}
BUCKET=s3://$RUNPOD_VOLUME_ID
aws() {
    local bin=(uvx --from awscli aws)
    type -P aws >/dev/null && bin=(command aws)   # 함수 자신이 아닌 실제 실행 파일이 있을 때만
    "${bin[@]}" --region "$REGION" --endpoint-url "$ENDPOINT" "$@"
}

cmd=${1:-ls}; shift || true
case $cmd in
    push-data) aws s3 cp "${1:?data.zip 경로}" "$BUCKET/stv/data.zip" ;;
    ls)        aws s3 ls "$BUCKET/${1:-stv/outputs/$STV_USER/}" ;;
    pull)      aws s3 sync "$BUCKET/stv/outputs/${1:-$STV_USER}/" "outputs/${1:-$STV_USER}/" --exclude "trainer/*" --exclude "*_best/*" --exclude "wandb/*" ;;
    pull-sub)  aws s3 sync "$BUCKET/stv/outputs/${1:-$STV_USER}/" "outputs/${1:-$STV_USER}/" --exclude "*" --include "submission*.csv" ;;
    aws)       aws "$@" ;;
    *)         sed -n 2,9p "$0"; exit 1 ;;
esac
