import argparse
import tomllib
from dataclasses import dataclass, fields


@dataclass
class Config:
    model_id: str = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"
    data_dir: str = "data"
    output_dir: str = "outputs/qwen2_5_vl_3b_lora"
    seed: int = 42
    load_in_4bit: bool = True  # false면 bf16 그대로 로드 (zero-shot probe용)

    # data
    max_train_samples: int = 200  # MVP: 일부만 사용, 0이면 전체
    valid_ratio: float = 0.1

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    finetune_vision: bool = False  # vision encoder를 얼려 VRAM 절약

    # train
    epochs: float = 1.0
    batch_size: int = 1
    grad_accum: int = 4
    lr: float = 1e-4

    # inference
    adapter_dir: str = ""  # 비우면 output_dir 사용, "none"이면 zero-shot
    split: str = "test"  # test | valid | dev | train (--valid-ratio 0이면 train.csv 전체)
    max_infer_samples: int = 0
    infer_batch_size: int = 8


def load_config(path: str = "", **overrides) -> Config:
    """For notebooks: toml file + keyword overrides."""
    values = {}
    if path:
        with open(path, "rb") as fp:
            values = tomllib.load(fp)
    return Config(**{**values, **overrides})


def parse_config() -> Config:
    """Every Config field becomes a --kebab-case CLI flag.

    Priority: CLI flag > --config toml > dataclass default.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="", help="toml file with Config field names as keys")
    file_values = {}
    if path := pre.parse_known_args()[0].config:
        with open(path, "rb") as fp:
            file_values = tomllib.load(fp)
        if unknown := set(file_values) - {f.name for f in fields(Config)}:
            raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")

    parser = argparse.ArgumentParser(parents=[pre])
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(flag, type=f.type, default=f.default)
    parser.set_defaults(**file_values)
    args = vars(parser.parse_args())
    args.pop("config")
    return Config(**args)
