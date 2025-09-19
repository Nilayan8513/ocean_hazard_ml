from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import json
import os

app = FastAPI()

# Allow frontend (React) to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load HF model
MODEL_PATH = "Nilayan87/ocean_hazard"   # your HuggingFace repo
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
clf = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Load dummy hazard dataset
with open("data/social_media_hazard_feed.json", "r", encoding="utf-8") as f:
    all_data = json.load(f)

# Root check
@app.get("/")
def root():
    return {"message": "🌊 Ocean Hazard API is running!"}

# Classify one post
@app.get("/classify")
def classify(text: str = Query(...)):
    result = clf(text)
    return {"input": text, "prediction": result}

# Return only hazard posts
@app.get("/hazards")
def get_hazards():
    hazards = [post for post in all_data if post.get("event_type") != "non-hazard"]
    return hazards

# Return heatmap coordinates only
@app.get("/hazards/heatmap")
def get_heatmap():
    heatmap = [
        {"lat": post["lat"], "lon": post["lon"], "event_type": post["event_type"]}
        for post in all_data if post.get("event_type") != "non-hazard"
    ]
    return heatmap
