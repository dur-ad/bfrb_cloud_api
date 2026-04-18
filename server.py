from fastapi import FastAPI
import torch
import numpy as np
import torch.nn.functional as F
from model_class import LightweightMV2

app = FastAPI()

CLASS_NAMES = [
    "Cuticle Picking", "Eyeglasses", "Face Touching", "Hair Pulling",
    "Hand Waving", "Knuckle Cracking", "Leg Scratching", "Leg Shaking",
    "Nail Biting", "Phone Call", "Raising Hand", "Reading",
    "Scratching Arm", "Sitting Still", "Sit-to-Stand", "Standing",
    "Stand-to-Sit", "Stretching", "Thumb Sucking", "Walking",
]

print("Loading model...")

ckpt = torch.load(
    "model_state_dict.pt",
    map_location="cpu",
    weights_only=False
)

input_size = ckpt["input_size"]

model = LightweightMV2(
    input_size=input_size,
    num_classes=20
)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

mu = ckpt["mu"].astype(np.float32)
sigma = ckpt["sigma"].astype(np.float32)

print("Model loaded successfully!")

@app.get("/")
def home():
    return {"message": "BFRB Model API Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        arr = np.array(data["features"], dtype=np.float32)

        if arr.ndim != 2:
            return {"error": "Expected shape (T, F)"}

        arr = (arr - mu) / sigma

        tensor = torch.from_numpy(arr).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=-1)[0].numpy()

        top5 = sorted(
            enumerate(probs),
            key=lambda x: -x[1]
        )[:5]

        result = [
            {
                "class": CLASS_NAMES[i],
                "probability": float(p)
            }
            for i, p in top5
        ]

        return {"predictions": result}

    except Exception as e:
        return {"error": str(e)}