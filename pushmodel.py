from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Path to your saved model folder
model_path = "./T5_large_QLoRA"

# Load model & tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Push to Hugging Face Hub
model.push_to_hub("sarayusapa/T5_large_QLoRA")
tokenizer.push_to_hub("sarayusapa/T5_large_QLoRA")
