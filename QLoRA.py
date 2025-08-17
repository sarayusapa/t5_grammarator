import torch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer
)

torch.cuda.empty_cache()

import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

wandb.init(
    project="t5-large-lang8",  # your project name
    name="first-run",               # run name
)

def main() -> None:
    # Model: T5-large
    model_name = "t5-large"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4-bit quantization config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Base model in 4-bit
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Prepare for k-bit training and apply LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    # Disable cache for training to reduce memory and enable checkpointing compatibility
    if hasattr(model, "config"):
        model.config.use_cache = False

    # Dataset: Lang-8 on HF Hub
    ds = load_dataset("sarayusapa/Grammar_Error_Correction")

    # Choose a train split; create a small eval from it if no dedicated split
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    #small batch for testing, comment out later
    #train_dataset = train_dataset.select(range(500000))  # first 100000 samples
    #eval_dataset = eval_dataset.select(range(5000))    # first 10000 samples

    feature_names = set(train_dataset.features.keys())
    src_field, tgt_field, tgt_is_list = "wrong", "correct", False

    # Preprocess -> prompt + target; mask prompt tokens with -100
    def preprocess_function(examples):
        sources = [f"Grammar Correction: {s}" for s in examples["wrong"]]
        targets = examples["correct"]

        model_inputs = tokenizer(
            sources,
            max_length=128,
            truncation=True,
            padding=False, 
            add_special_tokens=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=128,
                truncation=True,
                padding=False,
                add_special_tokens=True
            )["input_ids"]
            
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Decode generated tokens
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in labels and decode
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute BLEU
        bleu = bleu_metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        # Compute ROUGE
        rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        return {
            "bleu": bleu["bleu"],
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
        }

    # Training
    training_args = TrainingArguments(
        output_dir="./ModelCheckpoints",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        num_train_epochs=2,
        learning_rate=2e-4, 
        warmup_ratio = 0.05,
        logging_steps=100,
        save_strategy="steps",
        save_steps=3000,
        eval_strategy="steps",
        eval_steps = 3000,
        optim="paged_adamw_8bit",
        predict_with_generate=True,
        tf32=True,
        fp16=False,
        bf16=True,
        lr_scheduler_type="cosine", #scales loss function updation based on current value of loss function
        report_to=["wandb"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics = compute_metrics,
    )

    trainer.train()

    # Save LoRA adapters and tokenizer
    save_dir = "./qlora-t5-base"
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":

    main()
















































