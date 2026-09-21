"""wandb 기록 (선택). WANDB_API_KEY가 없으면 모든 함수가 아무 일도 하지 않습니다.

train → dev 추론 → test 추론이 한 run에 이어 기록되도록 run id를 output_dir/wandb_run_id.txt에 둡니다.
환경변수: WANDB_API_KEY, WANDB_ENTITY(팀), WANDB_PROJECT(기본 stv), WANDB_RUN_NAME(기본 output_dir 폴더명)
"""

import os
from dataclasses import asdict

from .config import Config

_run = None


def enabled() -> bool:
    return bool(os.environ.get("WANDB_API_KEY")) and os.environ.get("WANDB_MODE") != "disabled"


def init(cfg: Config, job: str):
    global _run
    if not enabled() or _run is not None:
        return _run
    import wandb

    os.makedirs(cfg.output_dir, exist_ok=True)
    id_path = os.path.join(cfg.output_dir, "wandb_run_id.txt")
    if os.path.exists(id_path):
        run_id = open(id_path).read().strip()
    else:
        run_id = wandb.util.generate_id()
        with open(id_path, "w") as fp:
            fp.write(run_id)
    _run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "stv"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=os.environ.get("WANDB_RUN_NAME") or os.path.basename(os.path.normpath(cfg.output_dir)),
        id=run_id,
        resume="allow",
        config=asdict(cfg) if job == "train" else None,
        tags=[t for t in (os.environ.get("STV_USER"), os.environ.get("STV_ENV")) if t],
    )
    return _run


def log(metrics: dict, step: int | None = None):
    if _run is not None:
        _run.log(metrics, step=step)


def summary(metrics: dict):
    if _run is not None:
        _run.summary.update(metrics)


def finish():
    global _run
    if _run is not None:
        _run.finish()
        _run = None
