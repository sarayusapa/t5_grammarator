import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate
import numpy as np
import wandb
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

torch.cuda.empty_cache()

wandb.init(
    project="t5-large-lora-peft",
    name="lora-peft-run",
)

def main() -> None:
    model_name = "t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True) #dropout rate, attention dropout_rate
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q","k","v", "o"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)

    ds = load_dataset("sarayusapa/Grammar_Error_Correction")

    train_dataset = ds["train"]
    eval_dataset = ds["validation"]

    src_field, tgt_field, tgt_is_list = "wrong", "correct", False

    def preprocess_function(examples):
        sources = [f"Grammar Correction: {s}" for s in examples["wrong"]]
        targets = examples["correct"]

        model_inputs = tokenizer(
            sources,
            max_length=128,
            truncation=True,
            padding="max_length",
            add_special_tokens=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=128,
                truncation=True,
                padding="max_length",
                add_special_tokens=True
            )["input_ids"]

        labels = [[(t if t != tokenizer.pad_token_id else -100) for t in lab] for lab in labels]
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names, load_from_cache_file=False, desc="Tokenized Train")
    tokenized_eval = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names, load_from_cache_file=False, desc="Tokenized Eval")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    print(tokenized_train["labels"][0][:20])


    bleu_metric = evaluate.load("bleu")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)


        bleu = bleu_metric.compute(
            predictions=decoded_preds, references=[[l] for l in decoded_labels]
        )

        exact_matches = sum(p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels))
        accuracy = exact_matches / len(decoded_preds)

        return {
            "bleu": bleu["bleu"],
            "accuracy": accuracy,
        }


    training_args = Seq2SeqTrainingArguments(
        output_dir="./t5_lora_peft_output",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=8,
        save_steps=8,
        max_steps=48,
        learning_rate=2.4e-4,
        warmup_ratio=0.025,
        logging_steps=2,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to=["wandb"],
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        lr_scheduler_type="linear",
        predict_with_generate=True,  # Needed for Seq2Seq generation
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
    metrics = trainer.evaluate()

    print(f"BLEU score: {metrics['eval_bleu']}")
    print(f"Accuracy: {metrics['eval_accuracy']}")

    model.save_pretrained("./t5_lora_peft_output")
    tokenizer.save_pretrained("./t5_lora_peft_output")

    trainer.model.push_to_hub("sarayusapa/T5_Large_QLoRA")
    tokenizer.push_to_hub("sarayusapa/T5_Large_QLoRA")


if __name__ == "__main__":
    main()