from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk
import pandas as pd
from transformers import T5Tokenizer
clean_lines = []
with open("./Clang8/output_data/clang8_source_target_en.spacy_tokenized.tsv", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 2:
            clean_lines.append(parts)
df = pd.DataFrame(clean_lines, columns=["input_text", "target_text"])
df.to_csv("./Clang8/output_data/clang8_cleaned.tsv", sep="\t", index=False)
df = pd.read_csv("./Clang8/output_data/clang8_cleaned.tsv", sep="\t")
tsv_dataset = Dataset.from_pandas(df)
hf_lang8 = load_dataset("MohamedAshraf701/lang-8")
lang8_dataset = hf_lang8["train"].rename_columns({
    "processed_input": "input_text",
    "processed_output": "target_text"
})
combined_data = concatenate_datasets([tsv_dataset, lang8_dataset])
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")

def preprocess(examples):
    inputs = ["correct: " + inp for inp in examples["input_text"]]
    targets = examples["target_text"]
    if not all(isinstance(x, str) for x in targets):
        print("Non-string target found:", targets)
        raise ValueError("Found non-string target in batch.")
    model_inputs = tokenizer(
        inputs,
        max_length=128,     
        truncation=True,
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
def is_valid_example(example):
    return isinstance(example["input_text"], str) and isinstance(example["target_text"], str)
combined_data = combined_data.filter(is_valid_example)
repo_name = "Hritshhh/T5-Dataset"
combined_data.push_to_hub(repo_name,private=False)
tokenized_dataset = combined_data.map(preprocess, batched = True, remove_columns = ["input_text","target_text"])
tokenized_dataset.save_to_disk("./tokenized_grammar_data")