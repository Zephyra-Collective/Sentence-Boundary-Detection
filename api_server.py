from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

# Load trained model
model_path = os.path.join(os.path.dirname(_file_), 'crf_model.pkl')
model = joblib.load(model_path)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    # Add your preprocessing logic here using existing utilities
    # prediction = model.predict(processed_features)
    return {"prediction": "sample_output"}