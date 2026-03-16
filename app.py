from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import logging

# --------------------------------------------------
# Initialize app & logging
# --------------------------------------------------
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------
# Load models
# --------------------------------------------------
MODEL_DIR = "models"

try:
    model_vgg = tf.keras.models.load_model(os.path.join(MODEL_DIR, "vgg16.h5"))
    model_resnet = tf.keras.models.load_model(os.path.join(MODEL_DIR, "resnet50.h5"))
    model_inception = tf.keras.models.load_model(os.path.join(MODEL_DIR, "inceptionv3.h5"))

    models = [model_vgg, model_resnet, model_inception]
    logging.info("✅ All models loaded successfully")

except Exception as e:
    logging.error(f"❌ Error loading models: {e}")
    models = []

# --------------------------------------------------
# Class labels
# --------------------------------------------------
CLASS_NAMES = ["Normal", "Polyps", "Ulcerative colitis", "Esophagitis"]

# --------------------------------------------------
# Health route (Render checks this)
# --------------------------------------------------
@app.route("/")
def health():
    return "Backend is running 🚀"

# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return tf.convert_to_tensor(image, dtype=tf.float32)

# --------------------------------------------------
# Prediction route
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image key found in the request"}), 400

    file = request.files["image"]

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_tensor = preprocess_image(image)

        predictions = []

        for model in models:
            pred = model(img_tensor, training=False).numpy()[0]
            predictions.append(pred)

        avg_prediction = np.mean(predictions, axis=0)

        predicted_class = CLASS_NAMES[int(np.argmax(avg_prediction))]
        confidence = float(np.max(avg_prediction) * 100)

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Invalid image file"}), 500


# --------------------------------------------------
# Run server (Render compatible)
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)