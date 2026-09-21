from unsloth import FastVisionModel, is_bf16_supported  # isort: skip  (must precede trl)
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer

from .config import Config, parse_config
from .data import load_split, subsample, to_sft_dataset
from .model import add_lora, load_model


def train(cfg: Config):
    train_df = subsample(load_split(cfg, "train"), cfg.max_train_samples, cfg.seed)
    train_dataset = to_sft_dataset(train_df, cfg.data_dir)
    print(f"train samples: {len(train_dataset)}")

    model, processor = load_model(cfg.model_id, cfg.load_in_4bit)
    model = add_lora(model, cfg)
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=train_dataset,
        # VLM에서는 Unsloth 전용 collator 사용
        data_collator=UnslothVisionDataCollator(model, processor),
        args=SFTConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.lr,
            warmup_ratio=0.03,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            optim="adamw_8bit",
            logging_steps=5,
            save_strategy="no",
            seed=cfg.seed,
            # VLM 필수 설정
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            report_to="none",
        ),
    )
    trainer.train()

    model.save_pretrained(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    print("Saved:", cfg.output_dir)


def main():
    train(parse_config())


if __name__ == "__main__":
    main()
