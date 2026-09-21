#!/usr/bin/env bash
# (호스트에서 실행) 현재 작업 트리를 Docker Jupyter lab 컨테이너의 /workspace로 복사하고 setup 실행.
# 컨테이너에 repo가 마운트돼 있지 않아서, 코드를 고칠 때마다 다시 실행하면 됩니다.
#
#   bash scripts/lab_sync.sh [CONTAINER]     # 기본 ssafy-ai → http://127.0.0.1:8888/lab
set -euo pipefail
cd "$(dirname "$0")/.."
CONTAINER=${1:-ssafy-ai}
DEST=/workspace/scene-text-vision-repo

docker exec "$CONTAINER" mkdir -p "$DEST"
# tracked + untracked(ignore 제외) 파일만 전송 → data/, .venv/, outputs/는 빠짐
git ls-files -co --exclude-standard -z | tar -c --null -T - | docker exec -i "$CONTAINER" tar -x -C "$DEST"
docker exec "$CONTAINER" bash "$DEST/scripts/setup.sh"
echo "[lab_sync] $CONTAINER:$DEST 동기화 완료 → notebooks/lab.ipynb"
