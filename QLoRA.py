import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

wandb.init(
    project="flan-t5-base-lang8",  # your project name
    name="first-run",               # run name
)

def main() -> None:
    # Model: Qwen 0.5B Instruct
    model_name = "google/flan-t5-base"

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
        r=16,
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
    ds = load_dataset("Hritshhh/T5-Dataset")

    # Choose a train split; create a small eval from it if no dedicated split
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    base_train = ds[split_name]
    split = base_train.train_test_split(test_size=0.01, seed=42)
    train_dataset, eval_dataset = split["train"], split["test"]


    #small batch for testing, comment out later
    train_dataset = train_dataset.select(range(50000))  # first 325000 samples
    eval_dataset = eval_dataset.select(range(5000))    # first 32500 samples


    # Infer source/target fields
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
        raise ValueError(f"Unsupported dataset schema: found features {feature_names}")

    # Preprocess -> prompt + target; mask prompt tokens with -100
    def preprocess_function(examples):
        sources = examples[src_field]
        targets = examples[tgt_field]

        input_ids_list = []
        label_ids_list = []

        for src_text, tgt_val in zip(sources, targets):
            tgt_text = (
                (tgt_val[0] if isinstance(tgt_val, list) and len(tgt_val) > 0 else "")
                if tgt_is_list
                else (tgt_val or "")
            )

            prompt = (
                "Correct the grammar of the following sentence.\n"
                f"Input: {src_text}\n"
                "Output: "
            )

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(tgt_text, add_special_tokens=False)["input_ids"] + [
                tokenizer.eos_token_id
            ]

            input_ids = prompt_ids + target_ids
            label_ids = ([-100] * len(prompt_ids)) + target_ids

            input_ids_list.append(input_ids)
            label_ids_list.append(label_ids)

        # pad within batch
        max_len = max((len(x) for x in input_ids_list), default=0)
        padded_inputs, padded_labels, attention_masks = [], [], []
        for inp_ids, lab_ids in zip(input_ids_list, label_ids_list):
            pad_len = max_len - len(inp_ids)
            padded_inputs.append(inp_ids + [tokenizer.pad_token_id] * pad_len)
            padded_labels.append(lab_ids + [-100] * pad_len)
            attention_masks.append([1] * len(inp_ids) + [0] * pad_len)

        return {
            "input_ids": padded_inputs,
            "labels": padded_labels,
            "attention_mask": attention_masks,
        }

    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    # Collator: pad input_ids, attention_mask, and labels to the longest in the batch
    def batch_pad_collator(features):
        if len(features) == 0:
            return {
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "labels": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long),
            }

        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = tokenizer.pad_token_id

        input_ids = [f["input_ids"] + [pad_id] * (max_len - len(f["input_ids"])) for f in features]
        labels = [f["labels"] + [-100] * (max_len - len(f["labels"])) for f in features]
        attention_mask = [
            f["attention_mask"] + [0] * (max_len - len(f["attention_mask"])) for f in features
        ]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    data_collator = batch_pad_collator

    # Training
    training_args = TrainingArguments(
        output_dir="./",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8, #doubled to half the parameter updation frequency
        gradient_checkpointing=True,
        max_steps = 20,
        num_train_epochs=1, #change to 3 later
        learning_rate=1.1e-4, #halved
        warmup_ratio = 0.02,
        logging_steps=2, #change to 10 later
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps = 8,
        optim="paged_adamw_8bit",
        tf32=True,
        bf16=torch.cuda.is_available(),
        lr_scheduler_type="cosine", #scales loss function updation based on current value of loss function
        report_to=["wandb"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_dataset = tokenized_eval,
    )

    trainer.train()

    # Save LoRA adapters and tokenizer
    save_dir = "./qlora-flan-t5-base-lang8"
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":

    main()














