from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model
import torch

base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-large", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, "./qlora-flan-t5-base-lang8")
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained("t5-large")

model.push_to_hub("sarayusapa/T5_large_QLoRA")
tokenizer.push_to_hub("sarayusapa/T5_large_QLoRA")
