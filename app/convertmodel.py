from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from peft import PeftModel

USE_ADAPTER = False  # change to True if using LoRA or QLoRA adapters

MODEL_PATH = "sarayusapa/T5_Large_GEC_FullFT"  # full fine-tuned model by default
BASE_MODEL_PATH = "google/t5-large"
ADAPTER_PATH = "path/to/adapter"   # update this if using LoRA or QLoRA adapters

ONNX_PATH = Path("onnx_t5")

if USE_ADAPTER:
    MODEL_PATH = "merged_model"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_PATH)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.merge_and_unload()
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH,
    export=True,
    provider="CPUExecutionProvider",
)

onnx_model.save_pretrained(ONNX_PATH)
tokenizer.save_pretrained(ONNX_PATH)

print(f"Model exported to {ONNX_PATH.absolute()}")
