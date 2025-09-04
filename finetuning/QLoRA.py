import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
torch.cuda.empty_cache()
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import precision_score, recall_score, f1_score

wandb.init(
    project="t5-large-QL-final", 
    name="200k-final",
)

def main() -> None:

    model_name = "t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

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
    
    if hasattr(model, "config"):
        model.config.use_cache = False

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
    example_labels = tokenized_train["labels"][0]
    print("Example labels:", example_labels)
    print("Number of valid tokens:", sum(t != -100 for t in example_labels))
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./ModelCheckpoints-FINAL",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        num_train_epochs=3,
        learning_rate=1e-4, 
        warmup_ratio = 0.05,
        save_strategy="steps",
        save_steps=5000,
        eval_strategy="steps",
        eval_steps = 3000,
        logging_strategy = "steps",
        logging_steps = 50,
        optim="paged_adamw_8bit",
        dataloader_pin_memory=True,
        predict_with_generate=True,
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
    )

    trainer.train()

    save_dir = "./qlora-t5-large-FINAL"
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    trainer.model.push_to_hub("sarayusapa/T5_Large_GEC_QLoRA")
    tokenizer.push_to_hub("sarayusapa/T5_Large_GEC_QLoRA")

if __name__ == "__main__":

    main()