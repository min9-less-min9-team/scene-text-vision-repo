#!/usr/bin/env bash
# RunPod Network Volume을 S3 API로 다룹니다 (pod 없이 로컬에서 업로드·다운로드). 설정은 .env.
#
#   bash scripts/s3.sh push-data data.zip     # 데이터 업로드 (최초 1회) → stv/data.zip
#   bash scripts/s3.sh ls [경로]              # 기본: 내 결과 폴더 stv/outputs/{STV_USER}/
#   bash scripts/s3.sh log 실험이름           # run.log 끝부분 보기
#   bash scripts/s3.sh pull 실험이름 [사용자] # 결과를 outputs/{사용자}/{실험이름}/ 으로 내려받기
#   bash scripts/s3.sh aws ...                # 그대로 aws s3 명령 (예: aws s3 rm s3://$RUNPOD_VOLUME_ID/stv/outputs/x --recursive)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo ".env가 없습니다: cp .env.example .env"; exit 1; }
set -a; source .env; set +a
: "${RUNPOD_S3_ACCESS_KEY:?}" "${RUNPOD_S3_SECRET_KEY:?}" "${RUNPOD_VOLUME_ID:?}" "${RUNPOD_DATACENTER:?}" "${STV_USER:?}"

export AWS_ACCESS_KEY_ID=$RUNPOD_S3_ACCESS_KEY AWS_SECRET_ACCESS_KEY=$RUNPOD_S3_SECRET_KEY
DC=$(echo "$RUNPOD_DATACENTER" | tr '[:upper:]' '[:lower:]')
BUCKET=s3://$RUNPOD_VOLUME_ID
aws() {
    local bin=(uvx --from awscli aws)
    command -v aws >/dev/null && bin=(command aws)
    "${bin[@]}" --region "$RUNPOD_DATACENTER" --endpoint-url "https://s3api-$DC.runpod.io" "$@"
}

cmd=${1:-ls}; shift || true
case $cmd in
    push-data) aws s3 cp "${1:?data.zip 경로}" "$BUCKET/stv/data.zip" ;;
    ls)        aws s3 ls "$BUCKET/${1:-stv/outputs/$STV_USER/}" ;;
    log)       aws s3 cp "$BUCKET/stv/outputs/${2:-$STV_USER}/${1:?실험 이름}/run.log" - | tail -n "${LINES_N:-40}" ;;
    pull)      aws s3 sync "$BUCKET/stv/outputs/${2:-$STV_USER}/${1:?실험 이름}/" "outputs/${2:-$STV_USER}/$1/" --exclude "trainer/*" ;;
    aws)       aws "$@" ;;
    *)         sed -n 2,9p "$0"; exit 1 ;;
esac
