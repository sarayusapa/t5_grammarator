import torch
from torch.utils.data import DataLoader
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    predict_with_generate=True,
    default_data_collator,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

torch.cuda.empty_cache()

import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

wandb.init(
    project="t5-large-QL", 
    name="kaggle-run-final",
)

def main() -> None:
    # Model: T5-large
    model_name = "t5-large"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, dropout_rate=0.1, attention_dropout_rate=0.1)
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
    eval_dataset = ds["validation"]


    #small batch for testing, comment out later
    train_dataset = train_dataset.select(range(10000))  # first 100000 samples
    eval_dataset = eval_dataset.select(range(100))    # first 10000 samples

    feature_names = set(train_dataset.features.keys())
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

    bleu_metric = evaluate.load("bleu")
    accuracy = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Decode generated tokens
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        print("Predictions shape/type:", type(predictions), getattr(predictions, 'shape', 'no shape'))
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in labels and decode
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute BLEU
        bleu = bleu_metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        print(f"BLEU score: {bleu['bleu']}") 
        exact_matches = sum(p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels))
        accuracy = exact_matches / len(decoded_preds)
        return {
            "bleu": bleu["bleu"],
            "accuracy": accuracy
        }

    # Training
    training_args = Seq2SeqTrainingArguments(
        output_dir="./ModelCheckpoints",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        num_train_epochs=2,
        learning_rate=1.2e-4, 
        warmup_ratio = 0.05,
        save_strategy="steps",
        save_steps=2000,
        eval_strategy="steps",
        eval_steps = 200,
        logging_strategy = "steps",
        logging_steps = 50,
        optim="paged_adamw_8bit",
        dataloader_pin_memory=True,
        tf32=False,
        fp16=False,
        bf16=True,
        lr_scheduler_type="cosine", 
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

    # Save QLoRA adapters and tokenizer
    save_dir = "./qlora-t5-large"
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    trainer.model.push_to_hub("sarayusapa/T5_Large_QLoRA")
    tokenizer.push_to_hub("sarayusapa/T5_Large_QLoRA")


if __name__ == "__main__":

    main()











