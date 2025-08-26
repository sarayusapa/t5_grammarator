import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score

model_name = "sarayusapa/T5_Large_GEC_FULLFT"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

df = pd.read_csv("../Test_data/eval_dataset.csv")  
wrong_sentences = df["Ungrammatical Statement"].tolist()
correct_sentences = df["Standard English"].tolist()

def preprocess_function(sources, targets):
    model_inputs = tokenizer(
        sources,
        max_length=64,
        truncation=True,
        padding="max_length",
        add_special_tokens=True
    )

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

tokenized_data = preprocess_function(wrong_sentences, correct_sentences)

gleu_metric = evaluate.load("gleu")

predictions = []
references = []

for i in range(len(wrong_sentences)):
    input_ids = torch.tensor(tokenized_data["input_ids"][i]).unsqueeze(0).to("cuda")
    attention_mask = torch.tensor(tokenized_data["attention_mask"][i]).unsqueeze(0).to("cuda")

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=64)

    decoded_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decoded_label = tokenizer.decode(
        [t if t != -100 else tokenizer.pad_token_id for t in tokenized_data["labels"][i]],
        skip_special_tokens=True
    )

    predictions.append(decoded_pred)
    references.append(decoded_label)

gleu_score = gleu_metric.compute(predictions=predictions, references=[[r] for r in references])["gleu"]

y_true = [1] * len(references)
y_pred = [1 if p.strip() == r.strip() else 0 for p, r in zip(predictions, references)]

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"Model: {model_name}")
print(f"GLEU: {gleu_score:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
