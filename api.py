from fastapi import FastAPI
import torch
from model import MedicalNet

app = FastAPI()

model = MedicalNet()
model.load_state_dict(torch.load("global_model.pth"))
model.eval()

@app.post("/predict")
def predict(features: list):

    X = torch.tensor([features], dtype=torch.float32)
    outputs = model(X)
    prediction = torch.argmax(outputs, dim=1).item()

    return {"prediction": prediction}