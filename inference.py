from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Path to your saved model folder
model_path = "/full/path/to/your/model"

# Load model & tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Push to Hugging Face Hub
model.push_to_hub("sarayusapa/")
tokenizer.push_to_hub("sarayusapa/your-model-name")
