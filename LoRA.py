import torch
from torch.utils.data import DataLoader
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

torch.cuda.empty_cache()

import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

wandb.init(
    project="t5-large-lora-final",
    name="200k-lora-run",
)

def main() -> None:
    # Model: T5-large
    model_name = "t5-large"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Base model (full precision, no quantization for plain LoRA)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    # Disable cache for training to reduce memory and enable checkpointing compatibility
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Prepare for LoRA training
    model = prepare_model_for_kbit_training(model)  # still useful for uniformity
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "k", "v", "o"],  # more modules than QLoRA case
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)

    # Dataset: Lang-8 on HF Hub
    ds = load_dataset("sarayusapa/Grammar_Error_Correction")

    # Choose splits
    train_dataset = ds["train"]
    eval_dataset = ds["validation"]

    # (optional small eval)
    eval_dataset = eval_dataset.select(range(1000))

    src_field, tgt_field, tgt_is_list = "wrong", "correct", False

    def preprocess_function(examples):
        sources = [f"Grammar Correction: {s}" for s in examples["wrong"]]
        targets = examples["correct"]

        model_inputs = tokenizer(
            sources,
            max_length=64,
            truncation=True,
            padding="max_length",
            add_special_tokens=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=64,
                truncation=True,
                padding="max_length",
                add_special_tokens=True
            )["input_ids"]

        labels = [[(t if t != tokenizer.pad_token_id else -100) for t in lab] for lab in labels]
        model_inputs["labels"] = labels
        return model_inputs


    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names, load_from_cache_file=False, desc="Tokenized Train")
    tokenized_eval = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names, load_from_cache_file=False, desc="Tokenized Eval")

    example_labels = tokenized_train["labels"][0]
    print("Example labels:", example_labels)
    print("Number of valid tokens:", sum(t != -100 for t in example_labels))

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    print(tokenized_train["labels"][0][:20])

    # Metrics
    bleu_metric = evaluate.load("bleu")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()

        predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)


        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)


        bleu = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels]
        )
        print(f"BLEU score: {bleu['bleu']}")

        exact_matches = sum(p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels))
        accuracy = exact_matches / len(decoded_preds)

        return {
            "bleu": bleu["bleu"],
            "accuracy": accuracy,
        }


    training_args = Seq2SeqTrainingArguments(
        output_dir="./ModelCheckpoints-LoRA",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        num_train_epochs=2,
        learning_rate=2.4e-4,
        warmup_ratio=0.025,
        save_strategy="steps",
        save_steps=5000,
        eval_strategy="steps",
        eval_steps=3000,
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        optim="adamw_torch",
        dataloader_pin_memory=True,
        predict_with_generate=True,
        bf16=torch.cuda.is_available(),
        lr_scheduler_type="linear",
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    save_dir = "./lora-t5-large-FINAL"
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    trainer.model.push_to_hub("sarayusapa/T5_Large_GEC_LoRA")
    tokenizer.push_to_hub("sarayusapa/T5_Large_GEC_LoRA")


if __name__ == "__main__":
    main()
