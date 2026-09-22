#!/usr/bin/env python3
"""RunPod GPU 개발 환경(웹 Jupyter)을 켜고 끕니다. 실험은 Jupyter 안에서 직접 돌립니다. (표준 라이브러리만 사용)

    uv run scripts/runpod.py up                 # pod 생성 → Jupyter URL 출력
    uv run scripts/runpod.py up --open          # + 준비되면 브라우저로 열기
    uv run scripts/runpod.py jupyter            # 내 pod의 Jupyter를 브라우저로 열기
    uv run scripts/runpod.py status             # 내 pod 접속 정보 다시 보기
    uv run scripts/runpod.py down               # 내 pod 삭제 (과금 중지). volume(/workspace)은 남음
    uv run scripts/runpod.py list               # 계정의 모든 pod
    uv run scripts/runpod.py up --dry-run       # 요청 내용만 출력

설정은 repo root의 .env (.env.example 참고). 이미 export된 환경변수가 .env보다 우선합니다.
pod 안: repo는 volume의 /workspace/stv/home/{STV_USER}/{repo 이름} (pod를 지워도 남음), 부트스트랩 로그는 그 옆의 bootstrap.log
"""

import argparse
import json
import os
import secrets
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://rest.runpod.io/v1"
# pod 환경변수로 전달할 값 (GITHUB_TOKEN은 pod 안의 git clone/push에 사용)
FORWARD = ["STV_USER", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "HF_TOKEN", "GITHUB_TOKEN"]
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
DEFAULT_GPU = "NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 5090"


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
                 "User-Agent": "stv-runpod/2.0"},
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


def check_repo_access(repo: str, branch: str, token: str):
    """pod를 만들기(=과금) 전에 토큰으로 브랜치를 읽을 수 있는지 확인."""
    url = f"https://{'x-access-token:' + token + '@' if token else ''}github.com/{repo}.git"
    proc = subprocess.run(["git", "-c", "credential.helper=", "ls-remote", "--heads", url, branch],
                          capture_output=True, text=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if proc.returncode or not proc.stdout.strip():
        msg = (proc.stderr.strip() or f"브랜치 {branch} 없음").replace(token or "\0", "***")
        sys.exit(f"GitHub repo를 읽을 수 없습니다 (GITHUB_TOKEN 권한 / STV_REPO / STV_BRANCH 확인)\n{msg}")


def pod_name(user: str) -> str:
    return f"stv-{user}"[:60]


def my_pods(user: str) -> list[dict]:
    return [p for p in api("GET", "/pods") or [] if p.get("name") == pod_name(user)]


def jupyter_url(pod: dict) -> str:
    """RunPod HTTP 프록시를 통한 Jupyter 주소. 토큰은 pod env(JUPYTER_PASSWORD)에서."""
    token = (pod.get("env") or {}).get("JUPYTER_PASSWORD") or os.environ.get("JUPYTER_PASSWORD", "")
    return f"https://{pod['id']}-8888.proxy.runpod.net/lab" + (f"?token={token}" if token else "")


def print_info(pod: dict):
    user = os.environ["STV_USER"]
    gpu = (pod.get("machine") or {}).get("gpuTypeId") or ""
    print(f"pod {pod['id']}  {pod.get('desiredStatus', '?')}  {gpu}  ${pod.get('costPerHr', '?')}/h")
    print(f"  콘솔     https://console.runpod.io/pods?id={pod['id']}")
    print(f"  Jupyter  {jupyter_url(pod)}   (열리지 않으면 시작 중 → 잠시 뒤 uv run scripts/runpod.py jupyter)")
    print(f"  repo     /workspace/stv/home/{user}/{repo_dirname()}   (준비 로그: bootstrap.log, 끝나면 '[bootstrap] done')")
    print(f"  종료     uv run scripts/runpod.py down   ← 잊으면 계속 과금됩니다")


def repo_dirname() -> str:
    return os.environ.get("STV_REPO", "stv").split("/")[-1]


def up(args):
    user, _, volume = need("STV_USER", "RUNPOD_API_KEY", "RUNPOD_VOLUME_ID")
    repo, branch = os.environ.get("STV_REPO", ""), os.environ.get("STV_BRANCH", "main")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not repo:
        sys.exit(".env에 STV_REPO가 필요합니다 (예: min9-less-min9-team/scene-text-vision-repo)")
    if not args.dry_run:
        if existing := my_pods(user):
            print(f"이미 실행 중인 pod가 있습니다 ({pod_name(user)}). 새로 만들지 않습니다.")
            print_info(existing[0])
            return
        check_repo_access(repo, branch, token)
    if not os.environ.get("WANDB_API_KEY"):
        print("⚠ WANDB_API_KEY가 없어 pod 안에서 wandb 기록이 꺼집니다")

    home = f"/workspace/stv/home/{user}"
    repo_dir = f"{home}/{repo_dirname()}"
    # 이미지 기본 /start.sh가 Jupyter(8888)를 띄움(JUPYTER_PASSWORD 필요). 그 앞에서 repo clone + 환경 준비를 백그라운드로 시작.
    # repo는 volume에 두므로 두 번째부터는 clone하지 않음 (작업 중이던 파일 유지). 토큰은 환경변수로만 전달.
    clone = (f"git clone -b {shlex.quote(branch)} https://x-access-token:$GITHUB_TOKEN@github.com/{repo}.git {shlex.quote(repo_dir)}"
             f" && git -C {shlex.quote(repo_dir)} remote set-url origin https://github.com/{repo}.git")
    start = (
        f"mkdir -p {shlex.quote(home)}; "
        f"{{ {{ [ -d {shlex.quote(repo_dir)}/.git ] || {clone}; }} && bash {shlex.quote(repo_dir)}/scripts/runpod_bootstrap.sh"
        f" || echo '[bootstrap] FAILED'; }} > {shlex.quote(home)}/bootstrap.log 2>&1 &\n"
        "exec /start.sh"
    )
    env = {k: os.environ[k] for k in FORWARD if os.environ.get(k)}
    env["JUPYTER_PASSWORD"] = os.environ.get("JUPYTER_PASSWORD") or secrets.token_urlsafe(12)
    body = {
        "name": pod_name(user),
        "imageName": os.environ.get("RUNPOD_IMAGE", DEFAULT_IMAGE),
        "gpuTypeIds": [g.strip() for g in os.environ.get("RUNPOD_GPU", DEFAULT_GPU).split(",") if g.strip()],
        "gpuCount": int(os.environ.get("RUNPOD_GPU_COUNT", 1)),
        "cloudType": os.environ.get("RUNPOD_CLOUD", "SECURE"),
        "containerDiskInGb": int(os.environ.get("RUNPOD_DISK_GB", 40)),
        "networkVolumeId": volume,
        "volumeMountPath": "/workspace",
        "ports": ["8888/http"],
        "env": env,
        "dockerStartCmd": ["bash", "-c", start],
    }
    if args.dry_run:
        body["env"] = {k: "***" if ("KEY" in k or "TOKEN" in k or "PASSWORD" in k) else v for k, v in env.items()}
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return
    pod = api("POST", "/pods", body, fatal=False)
    if "id" not in pod:
        gpus = ", ".join(body["gpuTypeIds"])
        # "no instances currently available" = 재고 소진, "could not find any pods with required specifications" = 그 데이터센터에 없는 GPU
        if any(m in str(pod.get("error", "")).lower() for m in ("no instances", "could not find any pods")):
            log_line(f"[실패] {user}: GPU 재고 없음 ({gpus}) → pod를 만들지 못했습니다 (과금 없음)")
            print(gpu_stock(volume, body["gpuTypeIds"]))
            sys.exit("잠시 뒤 다시 시도하거나 .env의 RUNPOD_GPU에 재고가 있는 GPU를 쉼표로 추가하세요")
        log_line(f"[실패] {user}: pod 생성 오류 {pod.get('status')} {pod.get('error')}")
        sys.exit(1)
    log_line(f"[생성] {user}: pod {pod['id']} {(pod.get('machine') or {}).get('gpuTypeId') or body['gpuTypeIds'][0]} ${pod.get('costPerHr', '?')}/h")
    os.environ["JUPYTER_PASSWORD"] = env["JUPYTER_PASSWORD"]

    if not args.no_wait:
        # Jupyter가 응답할 때까지(이미지 pull 포함 보통 1~3분) 대기
        print("pod 시작 대기 중", end="", flush=True)
        for _ in range(60):
            time.sleep(5)
            print(".", end="", flush=True)
            pod = api("GET", f"/pods/{pod['id']}", fatal=False) or pod
            if jupyter_ready(jupyter_url(pod)):
                break
        print()
    print_info(pod)
    if args.open:
        webbrowser.open(jupyter_url(pod))


def jupyter_ready(url: str) -> bool:
    """RunPod HTTP 프록시가 Jupyter에 연결되면 200(로그인 리다이렉트 포함). pod가 뜨는 중이면 프록시가 404/502를 돌려줌."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "stv-runpod/2.0"}), timeout=10) as resp:
            return resp.status < 400
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return False


def jupyter(args):
    """내 pod의 Jupyter URL 출력 + 브라우저로 열기."""
    user = need("STV_USER")[0]
    pods = my_pods(user)
    if not pods:
        sys.exit(f"실행 중인 pod 없음 ({pod_name(user)}) → uv run scripts/runpod.py up")
    url = jupyter_url(api("GET", f"/pods/{pods[0]['id']}", fatal=False) or pods[0])
    print(url)
    if not jupyter_ready(url):
        print("아직 응답이 없습니다(시작 중). 잠시 뒤 다시 시도하세요")
    webbrowser.open(url)


def status(args):
    user = need("STV_USER")[0]
    pods = my_pods(user)
    if not pods:
        print(f"실행 중인 pod 없음 ({pod_name(user)}) → uv run scripts/runpod.py up")
        return
    for pod in pods:
        print_info(api("GET", f"/pods/{pod['id']}", fatal=False) or pod)


def down(args):
    user = need("STV_USER")[0]
    if args.pod_id:
        targets = [{"id": args.pod_id, "name": ""}]
    else:
        targets = my_pods(user)
        if not targets:
            print(f"삭제할 pod 없음 ({pod_name(user)})")
            return
    for pod in targets:
        res = api("DELETE", f"/pods/{pod['id']}", fatal=False)
        if res and res.get("error"):
            log_line(f"[삭제 실패] {pod['id']}: {res['status']} {res['error']}")
        else:
            log_line(f"[삭제] {user}: pod {pod['id']} {pod.get('name', '')}")
    print("volume(/workspace)의 repo·결과·캐시는 그대로 남습니다. 다시 켜기: uv run scripts/runpod.py up")


def list_pods(_):
    pods = api("GET", "/pods") or []
    for pod in pods:
        print(f"{pod['id']}  {pod.get('desiredStatus', '?'):10s} ${pod.get('costPerHr', '?')}/h  {pod.get('name', '')}")
    if not pods:
        print("실행 중인 pod 없음")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("up", help="pod 생성 (웹 Jupyter)")
    p.add_argument("--dry-run", action="store_true", help="요청 내용만 출력")
    p.add_argument("--no-wait", action="store_true", help="Jupyter가 응답할 때까지 기다리지 않음")
    p.add_argument("--open", action="store_true", help="준비되면 Jupyter를 브라우저로 열기")
    p.set_defaults(func=up)
    sub.add_parser("jupyter", help="내 pod의 Jupyter URL 출력 + 브라우저로 열기").set_defaults(func=jupyter)
    sub.add_parser("status", help="내 pod 접속 정보").set_defaults(func=status)
    p = sub.add_parser("down", help="pod 삭제")
    p.add_argument("pod_id", nargs="?", help="생략하면 내 pod(stv-{STV_USER})")
    p.set_defaults(func=down)
    sub.add_parser("list", help="계정의 모든 pod").set_defaults(func=list_pods)

    args = parser.parse_args()
    load_env()
    need("RUNPOD_API_KEY")
    args.func(args)


if __name__ == "__main__":
    main()
