import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.model import build_model
from src.dataset import get_transforms
from monitoring.database import SessionLocal, PredictionLog, init_db

app = FastAPI(title="Cat vs Dog Classifier API")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/model.pt"
CLASS_NAMES_PATH = "models/class_names.json"

# Plus de chargement ici au niveau module
_model = None
_class_names = None
_transform = get_transforms(train=False)

init_db()


def get_model():
    global _model, _class_names
    if _model is None:
        if Path(CLASS_NAMES_PATH).exists():
            with open(CLASS_NAMES_PATH) as f:
                _class_names = json.load(f)["class_names"]
        else:
            _class_names = ["cat", "dog"]

        m = build_model(num_classes=len(_class_names), freeze_backbone=True)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        m.to(DEVICE)
        m.eval()
        _model = m

    return _model, _class_names


@app.get("/")
def root():
    _, class_names = get_model()
    return {"message": "Cat vs Dog Classifier API", "classes": class_names}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400, detail="Le fichier doit être une image (jpg/png)."
        )

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire l'image.")

    input_tensor = _transform(image).unsqueeze(0).to(DEVICE)

    model, class_names = get_model()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    predicted_class = class_names[pred_idx]
    confidence = probs[pred_idx].item()
    prob_dict = {
        class_names[i]: round(probs[i].item(), 4) for i in range(len(class_names))
    }

    db = SessionLocal()
    log_entry = PredictionLog(
        predicted_class=predicted_class,
        confidence=confidence,
        prob_cat=prob_dict.get("cat", 0.0),
        prob_dog=prob_dict.get("dog", 0.0),
    )
    db.add(log_entry)
    db.commit()
    prediction_id = log_entry.id
    db.close()

    return JSONResponse(
        {
            "prediction_id": prediction_id,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
        }
    )


class FeedbackRequest(BaseModel):
    prediction_id: int
    correct: bool
    true_class: str | None = None


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    db = SessionLocal()
    log_entry = (
        db.query(PredictionLog).filter(PredictionLog.id == req.prediction_id).first()
    )

    if log_entry is None:
        db.close()
        raise HTTPException(status_code=404, detail="Prediction ID introuvable.")

    log_entry.feedback = req.correct
    log_entry.true_class = req.true_class
    db.commit()
    db.close()

    return {"message": "Feedback enregistré, merci !"}
