import os

from unsloth import FastVisionModel

from .config import Config


def load_model(model_name: str):
    """model_name: base model id, or a saved LoRA adapter dir."""
    if os.environ.get("HF_HUB_OFFLINE") == "1" and not os.path.isdir(model_name):
        # unsloth는 hub id를 *-bnb-4bit 저장소로 바꿔 요청하므로, 오프라인에서는 캐시된 snapshot 경로를 직접 넘김
        from huggingface_hub import snapshot_download

        model_name = snapshot_download(model_name, local_files_only=True)
    return FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=True,  # QLoRA
        use_gradient_checkpointing="unsloth",
    )


def add_lora(model, cfg: Config):
    return FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.finetune_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=cfg.seed,
        use_gradient_checkpointing="unsloth",
    )
