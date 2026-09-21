import argparse
import tomllib
from dataclasses import dataclass, fields


@dataclass
class Config:
    model_id: str = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"  # Unsloth가 미리 4bit 양자화한 VL 모델
    data_dir: str = "data"
    output_dir: str = "outputs/qwen3_vl_4b"
    seed: int = 42
    load_in_4bit: bool = True  # false면 bf16 그대로 로드 (zero-shot probe용)

    # 해상도: Qwen3-VL은 32x32 픽셀마다 비전 토큰 1개. 원본 비율을 유지한 채 총 픽셀 수를 [min, max]로 맞춤
    # (데이터 대부분이 720x960 ≈ 0.69MP라 1024² 예산이면 원본 그대로 들어감)
    train_max_pixels: int = 1024 * 1024
    infer_max_pixels: int = 1024 * 1024
    min_pixels: int = 128 * 32 * 32  # 이보다 작은 이미지는 이 크기까지 키움
    max_seq_len: int = 4096  # 이미지 토큰 + 텍스트 토큰 상한 (잘리면 안 됨)

    # data
    max_train_samples: int = 0  # 0이면 전체. 빠른 실험은 1000 등
    valid_size: int = 500  # train.csv에서 떼어 둘 검증 문항 수 (정답이 확실한 train으로 검증). 0이면 검증 없음
    split_seed: int = 42  # 검증 분할 시드. seed와 분리해 모든 실험이 같은 검증 문항으로 비교·앙상블되게 함 (바꾸지 않기)
    use_dev: bool = False  # dev.csv 다수결 라벨을 학습에 추가할지. 라벨 노이즈가 커서 기본 false (docs/EDA.md)
    dev_min_votes: int = 3  # 5명 중 이 수 이상이 같은 답을 고른 dev 문항만 학습에 사용
    shuffle_options: bool = True  # 학습 때 보기 순서를 매번 무작위로 섞기 (정답 글자도 함께 변경)
    prompt_variant: str = "a"  # a: 기본, b: 글자 읽기 강조, c: 한국어 시스템 프롬프트 (학습·추론 공통)

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    finetune_vision: bool = True  # 비전 인코더에도 LoRA (글자 인식 개선). VRAM 부족 시 false
    init_adapter: str = ""  # 시작 LoRA 어댑터 (hub id 또는 폴더). 이어서 학습하거나 그대로 추론. r/alpha/vision은 어댑터에 맞춰짐

    # train
    epochs: float = 1.0
    batch_size: int = 2
    grad_accum: int = 4  # 유효 배치 = batch_size x grad_accum
    lr: float = 0.0  # 0이면 자동: init_adapter 있으면 5e-5, 없으면 1e-4
    eval_steps: int = 150  # optimizer step마다 검증 정확도 측정 → 최고점 어댑터 저장 (1회 약 3분). 0이면 학습 끝에만 저장

    # inference
    adapter_dir: str = ""  # 비우면 output_dir 사용, "none"이면 zero-shot(또는 init_adapter)
    split: str = "test"  # test | valid | dev | train (--valid-size 0이면 train.csv 전체)
    max_infer_samples: int = 0
    infer_batch_size: int = 4
    n_tta: int = 2  # 보기 순서를 바꿔 여러 번 예측해 평균 (1 = 끔). 파인튜닝 모델 +0.8%p, zero-shot +3.7%p 실측

    @property
    def effective_lr(self) -> float:
        return self.lr or (5e-5 if self.init_adapter else 1e-4)


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
