import os
import re
from functools import lru_cache

from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from transformers import pipeline

MODEL_ID = "imfarzanansari/skintelligent-acne"
SEVERITY_LABELS = {
    -1: "Clear Skin",
    0: "Occasional Spots",
    1: "Mild Acne",
    2: "Moderate Acne",
    3: "Severe Acne",
    4: "Very Severe Acne",
}


@lru_cache(maxsize=1)
def get_classifier():
    model_kwargs = {}
    token = os.getenv("HF_TOKEN")
    if token:
        model_kwargs["token"] = token

    return pipeline(
        task="image-classification",
        model=MODEL_ID,
        **model_kwargs,
    )


def parse_severity(label: str) -> int:
    """Extract a severity score (-1 to 4) from a model class label."""
    match = re.search(r"-?\d+", label)
    if not match:
        raise ValueError(f"Unable to parse severity from label: {label}")

    value = int(match.group(0))
    if value < -1 or value > 4:
        raise ValueError(f"Model returned out-of-range severity: {value}")
    return value


app = Flask(__name__)


@app.get("/health")
def health_check():
    return jsonify({"status": "ok", "model": MODEL_ID})


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing image file. Use multipart/form-data with field name 'image'."}), 400

    image_file = request.files["image"]
    if not image_file.filename:
        return jsonify({"error": "No file selected."}), 400

    try:
        image = Image.open(image_file.stream).convert("RGB")
    except UnidentifiedImageError:
        return jsonify({"error": "Uploaded file is not a valid image."}), 400

    try:
        classifier = get_classifier()
        result = classifier(image, top_k=1)
        top_prediction = result[0]
        raw_label = str(top_prediction["label"])
        severity_score = parse_severity(raw_label)
    except Exception as exc:
        return jsonify({"error": "Model inference failed.", "details": str(exc)}), 500

    return jsonify(
        {
            "severity_score": severity_score,
            "severity_label": SEVERITY_LABELS[severity_score],
            "raw_model_label": raw_label,
            "confidence": top_prediction.get("score"),
            "score_range": {"min": -1, "max": 4},
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
