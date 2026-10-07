"""
FastAPI service for Chest X-Ray Pneumonia Detection.
Loads the saved best model once at startup and exposes POST /predict.
"""
import io
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

MODEL_PATH = "model_artifacts/best_model.pth"
CONFIDENCE_THRESHOLD = 0.60  # below this, return "uncertain" (stretch goal)

app = FastAPI(title="Chest X-Ray Pneumonia Detector", version="1.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_baseline_cnn():
    class BaselineCNN(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Dropout(0.4), nn.Linear(256, 64), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(64, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return BaselineCNN()


def build_resnet():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, 2))
    return model


# ---- Load model once at startup ----
checkpoint = torch.load(MODEL_PATH, map_location=device)
model_type = checkpoint["model_type"]
model = build_resnet() if model_type == "resnet18" else build_baseline_cnn()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

IDX_TO_CLASS = {v: k for k, v in checkpoint["class_to_idx"].items()}
IMG_SIZE = checkpoint["img_size"]
MEAN, STD = checkpoint["mean"], checkpoint["std"]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    model_type: str


@app.get("/")
def root():
    return {"status": "ok", "model_type": model_type}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # --- input validation ---
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG or PNG image.",
        )

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. Is it a valid JPEG/PNG?")

    # --- inference ---
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pneumonia_prob = probs[1].item()
        pred_idx = int(probs.argmax().item())

    confidence = probs[pred_idx].item()

    # --- stretch goal: confidence threshold -> "uncertain" ---
    if confidence < CONFIDENCE_THRESHOLD:
        label = "UNCERTAIN"
    else:
        label = IDX_TO_CLASS[pred_idx]

    return PredictionResponse(label=label, confidence=round(confidence, 4), model_type=model_type)
