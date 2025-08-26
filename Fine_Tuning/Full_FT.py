import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import wandb

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.allow_tf32 = True

wandb.init(
    project="t5-large_full-ft_final",
    name="t5-large_full-ft_200k",
)

def main() -> None:
    model_name = "t5-large"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map="auto")

    if hasattr(model, "config"):
        model.config.use_cache = False

    ds = load_dataset("sarayusapa/Grammar_Error_Correction")
    train_dataset = ds["train"]
    eval_dataset = ds["validation"]
    eval_dataset = eval_dataset.select(range(1000))

    max_source_len = 64
    max_target_len = 64

    def preprocess_function(examples):
        sources = [f"Grammar Correction: {s}" for s in examples["wrong"]]
        targets = examples["correct"]

        model_inputs = tokenizer(
            sources,
            max_length=max_source_len,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_len,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
            )["input_ids"]

        labels = [[(t if t != tokenizer.pad_token_id else -100) for t in lab] for lab in labels]
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        desc="Tokenized Train",
    )
    tokenized_eval = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False,
        desc="Tokenized Eval",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100,pad_to_multiple_of=8)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./ModelCheckpoints_FullFT",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=1e-4,
        num_train_epochs = 3,
        warmup_ratio=0.02,
        lr_scheduler_type="cosine",
        save_strategy="no",
        save_steps=10000,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=2000,
        logging_strategy="steps",
        logging_steps=50,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to=["wandb"],
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        bf16=True,
        tf32=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        seed=42,
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

    save_dir = "./t5-large_fullft"
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    trainer.model.push_to_hub("Hritshhh/T5_Large_Full_FT")
    tokenizer.push_to_hub("Hritshhh/T5_Large_Full_FT")

if __name__ == "__main__":
    main()
