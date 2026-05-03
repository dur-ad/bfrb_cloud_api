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
        # ---------- LOG INCOMING REQUEST ----------
        print("\n--- NEW REQUEST ---", flush=True)
        print("Raw input keys:", data.keys(), flush=True)
        # DO NOT print(data) – it floods the logs with huge arrays
        # ---------- END LOG ----------

        arr = np.array(data["features"], dtype=np.float32)

        # ---------- LOG INPUT SHAPE ----------
        print("Input shape:", arr.shape, flush=True)
        # ---------- END LOG ----------

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

        # ---------- LOG TOP PREDICTIONS ----------
        print("Top predictions:", result, flush=True)
        print("--- END REQUEST ---\n", flush=True)
        # ---------- END LOG ----------

        return {"predictions": result}

    except Exception as e:
        # Log the error as well
        print("ERROR:", str(e), flush=True)
        return {"error": str(e)}
