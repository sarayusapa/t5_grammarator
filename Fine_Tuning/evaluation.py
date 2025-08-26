import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.gleu_score import corpus_gleu
from sklearn.metrics import precision_score, recall_score, f1_score

model_name = "sarayusapa/T5_Large_GEC_LoRA"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")
model.eval()

df = pd.read_csv("../Test_data/eval_dataset.csv")  
wrong_sentences = df["Ungrammatical Statement"].tolist()
correct_sentences = df["Standard English"].tolist()

batch_size = 64

predictions = []
references = []

for i in range(0, len(wrong_sentences), batch_size):
    batch_sources = wrong_sentences[i:i+batch_size]
    batch_targets = correct_sentences[i:i+batch_size]

    inputs = tokenizer(batch_sources, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64)

    decoded_preds = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    predictions.extend(decoded_preds)
    references.extend(batch_targets)

gleu_score = corpus_gleu([[r.split()] for r in references], [p.split() for p in predictions])

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
