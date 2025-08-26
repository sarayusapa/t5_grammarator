import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
torch.cuda.empty_cache()
import wandb
from peft import LoraConfig, get_peft_model, TaskType

wandb.init(
    project="t5-large-lora-final",
    name="200k-lora-run",
)

def main() -> None:
    model_name = "t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
 
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "k", "v", "o"],  
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)

    ds = load_dataset("sarayusapa/Grammar_Error_Correction")

    train_dataset = ds["train"]
    eval_dataset = ds["validation"]
    eval_dataset = eval_dataset.select(range(1000))

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

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./ModelCheckpoints-LoRA",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        num_train_epochs=2,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        save_strategy="steps",
        save_steps=5000,
        eval_strategy="steps",
        eval_steps=2000,
        logging_strategy="steps",
        logging_steps=50,
        optim="adamw_torch",
        dataloader_pin_memory=True,
        predict_with_generate=True,
        bf16=torch.cuda.is_available(),
        lr_scheduler_type="linear",
        report_to=["wandb"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    save_dir = "./lora-t5-large"
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    trainer.model.push_to_hub("sarayusapa/T5_Large_GEC_LoRA")
    tokenizer.push_to_hub("sarayusapa/T5_Large_GEC_LoRA")

if __name__ == "__main__":
    main()
