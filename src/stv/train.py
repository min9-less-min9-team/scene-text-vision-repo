from unsloth import FastVisionModel  # isort: skip  (must precede trl)
from unsloth.trainer import UnslothVisionDataCollator

import os
import time

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from .config import Config, parse_config
from .data import CHOICES, VQATrainDataset, load_split, load_train
from .inference import evaluate, predict_probs, save_outputs
from .model import add_lora, apply_init_adapter_config, load_adapter_weights, load_model, set_image_budget


def write_log(cfg: Config, msg: str):
    with open(os.path.join(cfg.output_dir, "log.txt"), "a", encoding="utf-8") as fp:
        fp.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


class BestValidCallback(TrainerCallback):
    """eval_steps마다 검증 정확도 측정 → 최고점이면 LoRA 가중치 저장.
    학습 전 상태(시작 모델)를 step 0 후보로 먼저 저장해 두어, 파인튜닝이 점수를 떨어뜨리면 시작 모델로 돌아감."""

    def __init__(self, model, processor, cfg: Config, valid_df, start_acc: float):
        self.processor, self.cfg, self.valid_df = processor, cfg, valid_df
        self.history = [(0, start_acc)]
        self._save_best(model, start_acc, 0)

    def _save_best(self, model, acc, step):
        self.best_acc, self.best_step = acc, step
        self.best_state = {k: v.detach().to("cpu", copy=True) for k, v in get_peft_model_state_dict(model).items()}
        model.save_pretrained(self.cfg.output_dir)

    def _eval(self, state, model):
        probs = predict_probs(model, self.processor, self.valid_df, self.cfg, n_tta=1, desc=f"valid@{state.global_step}")
        acc = evaluate(self.valid_df, probs, name=f"step {state.global_step}", show_wrong=0)
        self.history.append((state.global_step, acc))
        write_log(self.cfg, f"step {state.global_step}/{state.max_steps} valid_acc {acc:.4f}")
        if acc > self.best_acc:
            self._save_best(model, acc, state.global_step)
            print(f"   ★ best 갱신 → {self.cfg.output_dir}", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            write_log(self.cfg, f"step {state.global_step}/{state.max_steps} loss {logs['loss']:.4f} lr {logs.get('learning_rate', 0):.2e}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.cfg.eval_steps and state.global_step % self.cfg.eval_steps == 0 or state.global_step == state.max_steps:
            self._eval(state, model)


def check_label_masking(collator, dataset, tokenizer):
    """실제로 loss에 들어가는 토큰이 정답 글자 + <|im_end|> + 줄바꿈 뿐인지 확인"""
    batch = collator([dataset[0], dataset[1]])
    for ids, labels in zip(batch["input_ids"], batch["labels"]):
        kept = ids[labels != -100]
        print(f"시퀀스 길이 {len(ids):4d} | loss 대상 토큰 {len(kept)}개 → {tokenizer.decode(kept)!r}")
        assert 1 <= len(kept) <= 3 and tokenizer.decode(kept[:1]) in CHOICES, "정답 응답 외의 토큰이 loss에 포함됨"


def train(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    init_dir = apply_init_adapter_config(cfg)
    train_df, valid_df = load_train(cfg), load_split(cfg, "valid")
    print(f"train {len(train_df)} | valid {len(valid_df)} | 정답 분포 {train_df['answer'].value_counts().sort_index().to_dict()}")

    model, processor = load_model(cfg.model_id, cfg)
    model = add_lora(model, cfg)
    model.print_trainable_parameters()
    if init_dir:
        load_adapter_weights(model, init_dir)
    torch.cuda.set_per_process_memory_fraction(0.95)

    train_ds = VQATrainDataset(train_df, cfg)
    collator = UnslothVisionDataCollator(
        model, processor,
        resize="max",  # Unsloth 자체 리사이즈 끄기 → processor 픽셀 예산(set_image_budget)을 따름
        max_seq_length=cfg.max_seq_len,
        train_on_responses_only=True,  # 정답(assistant) 토큰에만 loss. 질문·이미지·패딩 토큰은 -100
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    set_image_budget(processor, cfg.train_max_pixels, cfg.min_pixels)
    check_label_masking(collator, train_ds, processor.tokenizer)

    # 시작 모델 점수 (step 0 기준선). 검증이 없으면 마지막 상태를 저장
    start_acc = -1.0
    if len(valid_df):
        FastVisionModel.for_inference(model)
        start_acc = evaluate(valid_df, predict_probs(model, processor, valid_df, cfg, desc="step 0 valid"), name="시작 모델 valid")
        write_log(cfg, f"step 0 ({'init_adapter' if init_dir else 'zero-shot'}) valid_acc {start_acc:.4f}")
    set_image_budget(processor, cfg.train_max_pixels, cfg.min_pixels)
    FastVisionModel.for_training(model)
    best_cb = BestValidCallback(model, processor, cfg, valid_df, start_acc) if len(valid_df) else None

    trainer = SFTTrainer(
        model=model,
        processing_class=processor.tokenizer,
        data_collator=collator,
        train_dataset=train_ds,
        callbacks=[best_cb] if best_cb else [],
        args=SFTConfig(
            output_dir=os.path.join(cfg.output_dir, "trainer"),
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.effective_lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            optim="adamw_8bit",
            max_grad_norm=1.0,
            bf16=True,
            logging_steps=10,
            save_strategy="no",  # 저장은 BestValidCallback이 담당
            report_to="none",
            seed=cfg.seed,
            # 비전 데이터용 필수 설정 (Unsloth 권장)
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=cfg.max_seq_len,
            dataloader_num_workers=0,
        ),
    )
    t0 = time.time()
    try:
        result = trainer.train()
        print(f"학습 시간 {(time.time() - t0) / 60:.1f}분 | 평균 train loss {result.training_loss:.4f}")
    except KeyboardInterrupt:
        print("⚠ 학습을 중단했습니다 → 지금까지의 best 어댑터로 복원합니다")

    if best_cb:
        write_log(cfg, f"train end {(time.time() - t0) / 60:.1f} min, best step {best_cb.best_step} valid_acc {best_cb.best_acc:.4f}")
        print("검증 정확도 기록 (step, 정확도):", [(s, round(a, 4)) for s, a in best_cb.history])
        set_peft_model_state_dict(model, best_cb.best_state)
        print(f"best step {best_cb.best_step} (검증 {best_cb.best_acc:.4f}) 어댑터로 복원 → {cfg.output_dir}")
        if best_cb.best_step == 0:
            print("⚠ 어떤 checkpoint도 시작 모델보다 좋지 않아 시작 모델을 그대로 사용합니다. LR·데이터를 점검하세요.")
    else:
        model.save_pretrained(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)

    # 최종 검증: best 어댑터 + TTA. valid_probs.npz는 앙상블 조합을 고를 때 사용
    if len(valid_df):
        FastVisionModel.for_inference(model)
        probs = predict_probs(model, processor, valid_df, cfg, n_tta=cfg.n_tta, desc=f"valid TTA{cfg.n_tta}")
        final_acc = evaluate(valid_df, probs, name=f"valid TTA{cfg.n_tta}")
        write_log(cfg, f"final valid_acc TTA{cfg.n_tta} {final_acc:.4f}")
        print(f"시작 {start_acc:.4f} → best step {best_cb.best_step} {best_cb.best_acc:.4f} → +TTA{cfg.n_tta} {final_acc:.4f}"
              f" (검증 {len(valid_df)}문항 표준오차 약 ±{(final_acc * (1 - final_acc) / len(valid_df)) ** 0.5 * 100:.1f}%p)")
        save_outputs(valid_df, probs, cfg, "valid")


def main():
    train(parse_config())


if __name__ == "__main__":
    main()
