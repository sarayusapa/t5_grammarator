from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

model_path = "onnx_t5"
model = ORTModelForSeq2SeqLM.from_pretrained(model_path, provider="CPUExecutionProvider")
tokenizer = AutoTokenizer.from_pretrained(model_path)

app = FastAPI(docs_url=None, redoc_url=None)  


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SingleInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]


@app.post("/predict")
def predict(data: SingleInput):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_length=512,   
        early_stopping=True,
        num_beams=4,       
        length_penalty=1.0, 
        no_repeat_ngram_size=3 
    )
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"output": corrected}

@app.post("/predict_batch")
def predict_batch(data: BatchInput):
    inputs = tokenizer(data.texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_length=512,
        early_stopping=True,
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=3
    )
    corrected = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return {"outputs": corrected}


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def static():
    return FileResponse("static/index.html")