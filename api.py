import json
import html
import re
from pathlib import Path

import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS


PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE_PATH = PROJECT_ROOT / "sentiment_pipeline.pkl"
METADATA_PATH = PROJECT_ROOT / "model_metadata.json"
LEGACY_MODEL_PATH = PROJECT_ROOT / "sentiment_model2.pkl"
LEGACY_VECTORIZER_PATH = PROJECT_ROOT / "tfidf_vectorizer2.pkl"

app = Flask(__name__)
CORS(app)


def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\bbr\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_model_bundle():
    if PIPELINE_PATH.exists():
        print("Loading sentiment pipeline...")
        pipeline = joblib.load(PIPELINE_PATH)
        metadata = {}
        if METADATA_PATH.exists():
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

        return {
            "mode": "pipeline",
            "pipeline": pipeline,
            "metadata": metadata,
        }

    print("Loading legacy model and vectorizer...")
    vectorizer = joblib.load(LEGACY_VECTORIZER_PATH)
    model = joblib.load(LEGACY_MODEL_PATH)
    return {
        "mode": "legacy",
        "vectorizer": vectorizer,
        "model": model,
        "metadata": {
            "task": "binary",
            "label_names": {"0": "Negatif", "1": "Positif"},
            "binary_decision_threshold": 0.5,
        },
    }


MODEL_BUNDLE = load_model_bundle()
print("Model's ready!")


def get_label_name(label, metadata):
    label_names = metadata.get("label_names") or {"0": "Negatif", "1": "Positif"}
    return label_names.get(str(label), str(label))


def predict_with_pipeline(text, bundle):
    pipeline = bundle["pipeline"]
    metadata = bundle.get("metadata", {})
    threshold = metadata.get("binary_decision_threshold")
    task = metadata.get("task", "binary")

    if task == "binary" and threshold is not None and hasattr(pipeline, "predict_proba"):
        classes = list(pipeline.classes_)
        positive_index = classes.index(1)
        probability = pipeline.predict_proba([text])[0]
        positive_score = float(probability[positive_index])
        prediction = 1 if positive_score >= float(threshold) else 0
        confidence = positive_score if prediction == 1 else 1 - positive_score
        return prediction, confidence

    prediction = pipeline.predict([text])[0]
    confidence = None
    if hasattr(pipeline, "predict_proba"):
        confidence = float(max(pipeline.predict_proba([text])[0]))

    return prediction, confidence


def predict_with_legacy_model(text, bundle):
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]
    metadata = bundle.get("metadata", {})
    threshold = metadata.get("binary_decision_threshold", 0.5)

    text_vector = vectorizer.transform([text])
    probability = model.predict_proba(text_vector)[0]
    classes = list(model.classes_)
    positive_index = classes.index(1)
    positive_score = float(probability[positive_index])
    prediction = 1 if positive_score >= float(threshold) else 0
    confidence = positive_score if prediction == 1 else 1 - positive_score
    return prediction, confidence


def predict_sentiment_label(text):
    if MODEL_BUNDLE["mode"] == "pipeline":
        prediction, confidence = predict_with_pipeline(text, MODEL_BUNDLE)
    else:
        prediction, confidence = predict_with_legacy_model(text, MODEL_BUNDLE)

    metadata = MODEL_BUNDLE.get("metadata", {})
    return {
        "prediction": int(prediction),
        "sentimen": get_label_name(prediction, metadata),
        "confidence": confidence,
    }


@app.route("/health", methods=["GET"])
def health():
    metadata = MODEL_BUNDLE.get("metadata", {})
    return jsonify(
        {
            "status": "ok",
            "model_mode": MODEL_BUNDLE["mode"],
            "task": metadata.get("task", "binary"),
            "selected_model": metadata.get("selected_model"),
        }
    )


@app.route("/metadata", methods=["GET"])
def metadata():
    model_metadata = MODEL_BUNDLE.get("metadata", {})
    if not model_metadata:
        return jsonify({"error": "Metadata model tidak ditemukan."}), 404

    return jsonify(
        {
            **model_metadata,
            "runtime": {
                "model_mode": MODEL_BUNDLE["mode"],
                "metadata_path": str(METADATA_PATH),
                "pipeline_path": str(PIPELINE_PATH),
            },
        }
    )


@app.route("/predict", methods=["POST"])
def predict_sentiment():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body harus berupa JSON object."}), 400

    if "text" not in data:
        return jsonify({"error": "Field 'text' wajib diisi."}), 400

    review_text = data["text"]
    if not isinstance(review_text, str):
        return jsonify({"error": "Field 'text' harus berupa string."}), 400

    review_text = review_text.strip()
    if not review_text:
        return jsonify({"error": "Teks ulasan tidak boleh kosong!"}), 400

    try:
        result = predict_sentiment_label(clean_text(review_text))
    except Exception as exc:
        return jsonify({"error": "Prediksi gagal.", "detail": str(exc)}), 500

    response = {
        "ulasan_asli": review_text,
        "label": result["prediction"],
        "sentimen": result["sentimen"],
    }
    if result["confidence"] is not None:
        response["confidence_score"] = f"{round(result['confidence'] * 100, 2)}%"

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
