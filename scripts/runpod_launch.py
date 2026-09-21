#!/usr/bin/env python3
"""RunPod GPU pod를 빌려 실험 1개를 돌리고, 끝나면 pod를 스스로 종료시킵니다. (표준 라이브러리만 사용)

    python scripts/runpod_launch.py run configs/sample.toml                       # 실험 이름 = 설정 파일명
    python scripts/runpod_launch.py run configs/sample.toml --name r32 -- --lora-r 32 --lora-alpha 32
    python scripts/runpod_launch.py run configs/sample.toml --keep                # 끝나도 pod 유지 (디버깅, 직접 종료 필요)
    python scripts/runpod_launch.py run configs/sample.toml --name smoke --infer-only -- --split dev --max-infer-samples 5 --n-tta 1
                                                                                  # 학습 없이 zero-shot 추론만 (파이프라인 점검용)
    python scripts/runpod_launch.py list                                          # 내 pod 목록
    python scripts/runpod_launch.py stop POD_ID                                   # pod 삭제

설정은 repo root의 .env (.env.example 참고). 이미 export된 환경변수가 .env보다 우선합니다.
결과: Network Volume의 stv/outputs/{STV_USER}/{실험 이름}/  → `bash scripts/s3.sh pull {실험 이름}`으로 내려받기
"""

import argparse
import json
import os
import shlex
import subprocess
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


def log_line(msg: str):
    """화면 + launch.log(repo root, git 제외)에 기록."""
    print(msg)
    with open(ROOT / "launch.log", "a", encoding="utf-8") as fp:
        fp.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def api(method: str, path: str, body=None, url: str = "", fatal: bool = True):
    req = urllib.request.Request(
        url or API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
        # User-Agent: 기본값(Python-urllib)은 Cloudflare가 403(1010)으로 막음
        headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}", "Content-Type": "application/json",
                 "User-Agent": "stv-runpod-launch/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as err:
        detail = err.read().decode(errors="replace")
        if not fatal:
            return {"error": detail, "status": err.code}
        sys.exit(f"RunPod API {method} {path} → {err.code}\n{detail}")


def gpu_stock(volume: str, gpu_ids: list[str]) -> str:
    """volume이 있는 데이터센터의 GPU별 재고·가격 (pod 생성 실패 시 원인 확인용)."""
    dc = (api("GET", f"/networkvolumes/{volume}", fatal=False) or {}).get("dataCenterId", "")
    query = ('query{gpuTypes{id memoryInGb lowestPrice(input:{gpuCount:1,secureCloud:true,dataCenterId:"%s"})'
             "{stockStatus uninterruptablePrice}}}" % dc)
    data = api("POST", "", {"query": query}, url="https://api.runpod.io/graphql", fatal=False) or {}
    rows = []
    for gpu in (data.get("data") or {}).get("gpuTypes") or []:
        low = gpu.get("lowestPrice") or {}
        if gpu["id"] in gpu_ids or low.get("stockStatus"):
            mark = "*" if gpu["id"] in gpu_ids else " "
            rows.append((low.get("uninterruptablePrice") or 99, f"  {mark} {gpu['id']:45s} {gpu['memoryInGb']:4d}GB  "
                         f"{'$%s/h' % low['uninterruptablePrice'] if low.get('uninterruptablePrice') else '-':9s} 재고 {low.get('stockStatus') or '없음'}"))
    return f"  데이터센터 {dc or '?'} (* = RUNPOD_GPU에 지정한 GPU)\n" + "\n".join(r for _, r in sorted(rows))


def check_repo_access(clone_url: str, branch: str, token: str):
    """pod를 만들기(=과금) 전에 토큰으로 브랜치를 읽을 수 있는지 확인."""
    proc = subprocess.run(["git", "-c", "credential.helper=", "ls-remote", "--heads", clone_url, branch],
                          capture_output=True, text=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if proc.returncode or not proc.stdout.strip():
        msg = (proc.stderr.strip() or f"브랜치 {branch} 없음").replace(token or "\0", "***")
        sys.exit(f"GitHub repo를 읽을 수 없습니다 (GITHUB_TOKEN 권한 / STV_REPO / STV_BRANCH 확인)\n{msg}")


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

    if not args.dry_run:
        check_repo_access(clone_url, branch, token)
    out_dir = f"/workspace/stv/outputs/{user}/{name}"

    # pod 시작 명령: repo를 받아 scripts/runpod_job.sh에 넘김. 실패해도 job 스크립트가 pod를 종료
    start = (
        f"rm -rf /root/stv && git clone --depth 1 -b {shlex.quote(branch)} {shlex.quote(clone_url)} /root/stv 2>/tmp/clone.log"
        f" && exec bash /root/stv/scripts/runpod_job.sh {shlex.quote(args.config)} {' '.join(map(shlex.quote, args.extra))}"
        # clone 자체가 실패하면 job 스크립트가 없으므로 이유를 volume에 남기고 여기서 종료
        f" || {{ mkdir -p {shlex.quote(out_dir)}; cp /tmp/clone.log {shlex.quote(out_dir)}/run.log; sleep 30; runpodctl remove pod $RUNPOD_POD_ID; sleep infinity; }}"
    )
    env = {k: os.environ[k] for k in FORWARD if os.environ.get(k)}
    env |= {"STV_RUN_NAME": name, "WANDB_RUN_NAME": f"{user}/{name}", "STV_KEEP_POD": "1" if args.keep else "0",
            "STV_MODE": "infer" if args.infer_only else "full"}
    body = {
        "name": f"stv-{user}-{name}"[:60],
        "imageName": os.environ.get("RUNPOD_IMAGE", "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"),
        "gpuTypeIds": [g.strip() for g in os.environ.get("RUNPOD_GPU", "NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 5090").split(",") if g.strip()],
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
    pod = api("POST", "/pods", body, fatal=False)
    if "id" not in pod:
        gpus = ", ".join(body["gpuTypeIds"])
        # "no instances currently available" = 재고 소진, "could not find any pods with required specifications" = 그 데이터센터에 없는 GPU
        if any(m in str(pod.get("error", "")).lower() for m in ("no instances", "could not find any pods")):
            log_line(f"[실패] {user}/{name}: GPU 재고 없음 ({gpus}) → pod를 만들지 못했습니다 (과금 없음)")
            print(gpu_stock(volume, body["gpuTypeIds"]))
            sys.exit("잠시 뒤 다시 시도하거나 .env의 RUNPOD_GPU에 재고가 있는 GPU를 쉼표로 추가하세요")
        log_line(f"[실패] {user}/{name}: pod 생성 오류 {pod.get('status')} {pod.get('error')}")
        sys.exit(1)
    log_line(f"[생성] {user}/{name}: pod {pod['id']} {pod.get('machine', {}).get('gpuTypeId') or body['gpuTypeIds'][0]} "
             f"${pod.get('costPerHr', '?')}/h mode={env['STV_MODE']} extra={' '.join(args.extra)}")
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
    p.add_argument("--infer-only", action="store_true", help="학습 없이 zero-shot 추론만 (split은 -- --split dev 처럼 지정, 기본 test)")
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
