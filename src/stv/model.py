import json
import os

from unsloth import FastVisionModel  # isort: skip

from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from .config import Config


def resolve_local(name: str) -> str:
    """오프라인(HF_HUB_OFFLINE=1)이면 캐시된 snapshot 폴더 경로로 바꿈.
    unsloth는 hub id를 *-bnb-4bit 저장소로 바꿔 요청하므로 경로를 직접 넘겨야 한다."""
    if os.path.isdir(name) or os.environ.get("HF_HUB_OFFLINE") != "1":
        return name
    from huggingface_hub import snapshot_download

    return snapshot_download(name, local_files_only=True)


def load_model(model_name: str, cfg: Config):
    """model_name: base model id, or a saved LoRA adapter dir."""
    return FastVisionModel.from_pretrained(
        resolve_local(model_name),
        load_in_4bit=cfg.load_in_4bit,  # QLoRA
        use_gradient_checkpointing="unsloth",  # 활성값을 CPU로 내려 VRAM 절약
        max_seq_length=cfg.max_seq_len,
    )


def apply_init_adapter_config(cfg: Config) -> str:
    """init_adapter가 있으면 LoRA 구조(r, alpha, 비전 LoRA)를 그 어댑터에 맞추고 폴더 경로를 반환."""
    if not cfg.init_adapter:
        return ""
    adapter_dir = resolve_local(cfg.init_adapter)
    if not os.path.isdir(adapter_dir):
        from huggingface_hub import snapshot_download

        adapter_dir = snapshot_download(cfg.init_adapter)
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as fp:
        acfg = json.load(fp)
    cfg.lora_r, cfg.lora_alpha = acfg["r"], acfg["lora_alpha"]
    cfg.finetune_vision = "visual" in str(acfg["target_modules"])
    if acfg["base_model_name_or_path"].split("/")[-1] != cfg.model_id.split("/")[-1]:
        print(f"⚠ 시작 어댑터의 베이스 모델({acfg['base_model_name_or_path']})이 model_id와 다릅니다")
    # 공개 어댑터는 train.sample(500, random_state=42)를 뺀 나머지로 학습됨 → 같은 검증 문항이어야 점수가 공정
    assert cfg.split_seed == 42 and cfg.valid_size <= 500, "init_adapter를 쓸 때는 split_seed=42, valid_size<=500 유지"
    print(f"시작 어댑터 {cfg.init_adapter}: r={cfg.lora_r} alpha={cfg.lora_alpha} vision={cfg.finetune_vision}")
    return adapter_dir


def add_lora(model, cfg: Config):
    return FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.finetune_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0,  # Unsloth 최적화 경로가 적용되는 값
        bias="none",
        random_state=cfg.seed,
        use_gradient_checkpointing="unsloth",
    )


def load_adapter_weights(model, adapter_dir: str):
    """저장된 LoRA 가중치를 현재 모델에 덮어씀. 구조(r·비전 LoRA 여부)가 다르면 에러"""
    res = set_peft_model_state_dict(model, load_file(os.path.join(adapter_dir, "adapter_model.safetensors")))
    missing_lora = [k for k in res.missing_keys if "lora_" in k]
    assert not res.unexpected_keys and not missing_lora, (
        f"어댑터 구조가 현재 LoRA 설정과 다릅니다 (finetune_vision·lora_r 확인): {(res.unexpected_keys or missing_lora)[:3]}"
    )


def set_image_budget(processor, max_pixels: int, min_pixels: int):
    """processor의 이미지 크기 설정. 원본 비율 유지, 가로·세로 32의 배수, 원본이 예산보다 작으면 그대로."""
    processor.image_processor.size = {"shortest_edge": int(min_pixels), "longest_edge": int(max_pixels)}
