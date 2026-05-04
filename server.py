from fastapi import FastAPI
import torch
import numpy as np
import torch.nn.functional as F
from model_class import LightweightMV2
from labels import CLASS_NAMES

app = FastAPI()

print("Loading model...")

ckpt = torch.load(
    "model_state_dict.pt",
    map_location="cpu",
    weights_only=False
)

input_size = ckpt["input_size"]

model = LightweightMV2(
    input_size=input_size,
    num_classes=len(CLASS_NAMES)
)

model.load_state_dict(ckpt["model_state_dict"])
model.eval() #disables dropout, batchnorm

mu = ckpt["mu"].astype(np.float32)
sigma = ckpt["sigma"].astype(np.float32)

print("Model loaded successfully!")

#a simple health check
@app.get("/")
def home():
    return {"message": "BFRB Model API Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        print("\n--- NEW REQUEST ---", flush=True)
        print("Raw input keys:", data.keys(), flush=True)

        arr = np.array(data["features"], dtype=np.float32) #shape = (T,F)
        print("Input shape:", arr.shape, flush=True)

        if arr.ndim != 2:
            return {"error": "Expected shape (T, F)"}

        arr = (arr - mu) / sigma

        tensor = torch.from_numpy(arr).unsqueeze(0) #shape = (1,T,F)

        with torch.no_grad(): #no grad computation
            logits = model(tensor) #shape = (1, 24)
            probs = F.softmax(logits, dim=-1)[0].numpy() #1d array of probablities

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

        print("Top predictions:", result, flush=True)
        print("--- END REQUEST ---\n", flush=True)

        return {"predictions": result}

    except Exception as e:
        #any error is returned as a JSON error message
        print("ERROR:", str(e), flush=True)
        return {"error": str(e)}
