from fastapi import FastAPI, HTTPException
import torch
import numpy as np
from src.model import EnhancedFraudDetector
import os

app = FastAPI()

# Load the trained model
model = EnhancedFraudDetector(input_dim=30)  # Adjust input_dim based on your features
models_dir = r'C:\Users\ELITEBOOK\OneDrive\Desktop\Projects\Fraud-Detection-System\models'
file_path = os.path.join(models_dir, 'fraud_detector_model.pth')
model.load_state_dict(torch.load(file_path))
model.eval()

@app.post("/predict")
async def predict(features: list):
    try:
        # Convert input to tensor
        input_tensor = torch.FloatTensor(features).reshape(1, -1)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            probability = prediction.item()
            
        return {
            "probability": probability,
            "prediction": "Fraud" if probability > 0.5 else "Normal",
            "confidence": probability if probability > 0.5 else 1 - probability
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))