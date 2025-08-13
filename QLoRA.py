import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

wandb.init(
    project="t5-large-lang8",
    name="first-run",
)

def main():
    # Model
    model_name = "t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4-bit quantization config
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
    model.config.use_cache = False

    # Dataset
    ds = load_dataset("Hritshhh/T5-Dataset")
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    base_train = ds[split_name]
    split = base_train.train_test_split(test_size=0.01, seed=42)
    train_dataset, eval_dataset = split["train"], split["test"]

    # Small subset for testing
    train_dataset = train_dataset.select(range(1000))
    eval_dataset = eval_dataset.select(range(100))

    # Determine source/target fields
    feature_names = set(train_dataset.features.keys())
    src_field, tgt_field, tgt_is_list = None, None, False
    candidates = [
        ("source", "target", False),
        ("incorrect", "corrected", False),
        ("original", "correction", False),
        ("sentence", "corrections", True),
        ("input", "output", False),
        ("processed_input", "processed_output", False),
        ("input_text", "target_text", False),
    ]
    for s, t, is_list in candidates:
        if s in feature_names and t in feature_names:
            src_field, tgt_field, tgt_is_list = s, t, is_list
            break
    if src_field is None:
        raise ValueError(f"Unsupported dataset schema: {feature_names}")

    # Preprocessing
    def preprocess_function(examples):
        input_ids_list = []
        label_ids_list = []

        for src_text, tgt_val in zip(examples[src_field], examples[tgt_field]):
            tgt_text = tgt_val[0] if (tgt_is_list and isinstance(tgt_val, list) and tgt_val) else (tgt_val or "")
            prompt = f"Correct the grammar of the following sentence.\nInput: {src_text}\nOutput: "
            prompt_ids = tokenizer(prompt, truncation=True, max_length=128)["input_ids"]
            target_ids = tokenizer(tgt_text, truncation=True, max_length=128)["input_ids"] + [tokenizer.eos_token_id]

            input_ids_list.append(prompt_ids + target_ids)
            label_ids_list.append([-100]*len(prompt_ids) + target_ids)

        return {"input_ids": input_ids_list, "labels": label_ids_list}

    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    # Collator handles padding dynamically
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./qlora_t5large_epochs",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        num_train_epochs=2,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        logging_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=200,
        optim="paged_adamw_8bit",
        fp16=True,
        report_to=["wandb"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.model.save_pretrained("./qlora_t5base_model")
    tokenizer.save_pretrained("./qlora_t5base_model")


if __name__ == "__main__":
    main()
