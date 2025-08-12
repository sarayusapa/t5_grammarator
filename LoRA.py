import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate
import wandb
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

wandb.init(
    project="t5-large-lora-peft",
    name="lora-peft-run",
)

def main() -> None:
    model_name = "t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
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

    ds = load_dataset("Hritshhh/T5-Dataset")
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    base_train = ds[split_name]
    split = base_train.train_test_split(test_size=0.01, seed=42)
    train_dataset, eval_dataset = split["train"], split["test"]

    train_dataset = train_dataset.select(range(20000))
    eval_dataset = eval_dataset.select(range(2000))

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

        max_len = max(len(ids) for ids in input_ids_list)
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
        attention_mask = [f["attention_mask"] + [0] * (max_len - len(f["attention_mask"])) for f in features]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    # Load BLEU metric
    bleu = evaluate.load("bleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]  # BLEU expects list of references per prediction
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Convert preds to tensor if not already
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    # If preds are logits (float), convert to token ids
    if preds.dtype != torch.int64:
        preds = preds.argmax(dim=-1)
    # Move to CPU numpy array
    preds = preds.detach().cpu().numpy()
    # Similarly convert labels to numpy array if tensor
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    # Decode preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in labels with pad token id for decoding
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["bleu"]}

    training_args = Seq2SeqTrainingArguments(
        output_dir="./t5_lora_peft_output",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=8, 
        save_steps=8,
        max_steps=24,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_ratio=0.05,
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
        data_collator=batch_pad_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.save_pretrained("./t5_lora_peft_output")
    tokenizer.save_pretrained("./t5_lora_peft_output")

if __name__ == "__main__":
    main()
