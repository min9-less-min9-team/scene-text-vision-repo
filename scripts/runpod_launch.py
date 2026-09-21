#!/usr/bin/env python3
"""RunPod GPU pod를 빌려 실험 1개를 돌리고, 끝나면 pod를 스스로 종료시킵니다. (표준 라이브러리만 사용)

    python scripts/runpod_launch.py run configs/sample.toml                       # 실험 이름 = 설정 파일명
    python scripts/runpod_launch.py run configs/sample.toml --name r32 -- --lora-r 32 --lora-alpha 32
    python scripts/runpod_launch.py run configs/sample.toml --keep                # 끝나도 pod 유지 (디버깅, 직접 종료 필요)
    python scripts/runpod_launch.py list                                          # 내 pod 목록
    python scripts/runpod_launch.py stop POD_ID                                   # pod 삭제

설정은 repo root의 .env (.env.example 참고). 이미 export된 환경변수가 .env보다 우선합니다.
결과: Network Volume의 stv/outputs/{STV_USER}/{실험 이름}/  → `bash scripts/s3.sh pull {실험 이름}`으로 내려받기
"""

import argparse
import json
import os
import shlex
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://rest.runpod.io/v1"
# pod 안으로 전달할 변수 (RUNPOD_API_KEY는 pod가 자기 자신을 종료하는 데 사용)
FORWARD = ["STV_USER", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "HF_TOKEN", "RUNPOD_API_KEY"]


def load_env():
    path = ROOT / ".env"
    if not path.exists():
        sys.exit(".env가 없습니다:  cp .env.example .env  후 값을 채우세요")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value[:1] in "\"'":
            value = value[1:value.index(value[0], 1)]
        else:
            value = (" " + value).split(" #")[0].strip()
        os.environ.setdefault(key.strip(), value)


def need(*keys):
    if missing := [k for k in keys if not os.environ.get(k)]:
        sys.exit(f".env에 값이 필요합니다: {', '.join(missing)}")
    return [os.environ[k] for k in keys]


def api(method: str, path: str, body=None):
    req = urllib.request.Request(
        API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as err:
        sys.exit(f"RunPod API {method} {path} → {err.code}\n{err.read().decode(errors='replace')}")


def run(args):
    user, _, volume = need("STV_USER", "RUNPOD_API_KEY", "RUNPOD_VOLUME_ID")
    if not os.environ.get("WANDB_API_KEY"):
        print("⚠ WANDB_API_KEY가 없어 wandb 기록 없이 실행합니다")
    if not (ROOT / args.config).exists():
        sys.exit(f"설정 파일이 없습니다: {args.config} (repo root 기준 경로, push된 브랜치에 있어야 합니다)")
    name = args.name or Path(args.config).stem
    repo, branch = os.environ.get("STV_REPO", ""), os.environ.get("STV_BRANCH", "main")
    token = os.environ.get("GITHUB_TOKEN", "")
    clone_url = f"https://{token + '@' if token else ''}github.com/{repo}.git"

    # pod 시작 명령: repo를 받아 scripts/runpod_job.sh에 넘김. 실패해도 job 스크립트가 pod를 종료
    start = (
        f"rm -rf /root/stv && git clone --depth 1 -b {shlex.quote(branch)} {shlex.quote(clone_url)} /root/stv"
        f" && exec bash /root/stv/scripts/runpod_job.sh {shlex.quote(args.config)} {' '.join(map(shlex.quote, args.extra))}"
        # clone 자체가 실패하면 job 스크립트가 없으므로 여기서 종료
        " || { sleep 60; runpodctl remove pod $RUNPOD_POD_ID; }"
    )
    env = {k: os.environ[k] for k in FORWARD if os.environ.get(k)}
    env |= {"STV_RUN_NAME": name, "WANDB_RUN_NAME": f"{user}/{name}", "STV_KEEP_POD": "1" if args.keep else "0"}
    body = {
        "name": f"stv-{user}-{name}"[:60],
        "imageName": os.environ.get("RUNPOD_IMAGE", "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"),
        "gpuTypeIds": [g.strip() for g in os.environ.get("RUNPOD_GPU", "NVIDIA GeForce RTX 4090").split(",") if g.strip()],
        "gpuCount": int(os.environ.get("RUNPOD_GPU_COUNT", 1)),
        "cloudType": os.environ.get("RUNPOD_CLOUD", "SECURE"),
        "containerDiskInGb": int(os.environ.get("RUNPOD_DISK_GB", 40)),
        "networkVolumeId": volume,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "env": env,
        "dockerStartCmd": ["bash", "-c", start],
    }
    if args.dry_run:
        body["env"] = {k: "***" if "KEY" in k or "TOKEN" in k else v for k, v in env.items()}
        print(json.dumps(body, indent=2, ensure_ascii=False).replace(token, "***") if token else json.dumps(body, indent=2, ensure_ascii=False))
        return
    pod = api("POST", "/pods", body)
    out = f"stv/outputs/{user}/{name}"
    print(f"pod 생성: {pod['id']}  ({pod.get('machine', {}).get('gpuTypeId') or body['gpuTypeIds'][0]}, ${pod.get('costPerHr', '?')}/h)")
    print(f"  콘솔     https://console.runpod.io/pods?id={pod['id']}")
    print(f"  결과     volume {volume}: {out}/   (run.log, log.txt, adapter, *_probs.npz, submission_test.csv)")
    print(f"  로그     bash scripts/s3.sh log {name}")
    print(f"  내려받기 bash scripts/s3.sh pull {name}")
    print(f"  종료     {'직접: python scripts/runpod_launch.py stop ' + pod['id'] if args.keep else '실험이 끝나면(실패 포함) 자동 삭제'}")


def list_pods(_):
    pods = api("GET", "/pods") or []
    for pod in pods:
        print(f"{pod['id']}  {pod.get('desiredStatus', '?'):10s} ${pod.get('costPerHr', '?')}/h  {pod.get('name', '')}")
    if not pods:
        print("실행 중인 pod 없음")


def stop(args):
    api("DELETE", f"/pods/{args.pod_id}")
    print("삭제:", args.pod_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("config")
    p.add_argument("--name", default="", help="실험 이름 (결과 폴더·wandb run 이름). 기본: 설정 파일명")
    p.add_argument("--keep", action="store_true", help="끝나도 pod를 삭제하지 않음")
    p.add_argument("--dry-run", action="store_true", help="요청 내용만 출력")
    p.set_defaults(func=run)
    sub.add_parser("list").set_defaults(func=list_pods)
    p = sub.add_parser("stop")
    p.add_argument("pod_id")
    p.set_defaults(func=stop)

    # "--" 뒤의 인자는 stv.train / stv.inference에 그대로 전달
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:cut])
    args.extra = argv[cut + 1:]
    load_env()
    need("RUNPOD_API_KEY")
    args.func(args)


if __name__ == "__main__":
    main()
