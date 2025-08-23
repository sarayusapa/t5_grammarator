from huggingface_hub import HfApi, HfFolder, Repository
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

save_dir = "./t5-large_fullft"

model = AutoModelForSeq2SeqLM.from_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(save_dir)

model.push_to_hub("sarayusapa/T5_large_GEC_FullFT")
tokenizer.push_to_hub("sarayusapa/T5_large_GEC_FullFT")
