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

# Use try-except in case models are missing when starting the server
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
CLASS_NAMES = ["Normal", "Polyps", "Ulcerative colotis", "Esophagitis"]

# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
def preprocess_image(image):
    # Note: If your InceptionV3 was trained on 299x299 (its default), 
    # you may need to write a separate preprocessing function for it.
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    # Convert to tensor for faster model() execution
    return tf.convert_to_tensor(image, dtype=tf.float32)

# --------------------------------------------------
# Prediction route
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # 1. Check if the image part is in the request
    if "image" not in request.files:
        return jsonify({"error": "No image key found in the request"}), 400

    file = request.files["image"]
    
    # 2. Check if a file was actually selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # 3. Safely open and process the image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_tensor = preprocess_image(image)

        # --------------------------------------------------
        # Ensemble prediction (AVERAGE)
        # --------------------------------------------------
        predictions = []
        for i, model in enumerate(models):
            # Use model(img, training=False) instead of model.predict(img) 
            # for much faster single-image inference in TF2
            pred = model(img_tensor, training=False).numpy()[0]
            predictions.append(pred)

            # Debugging Observation
            idx = int(np.argmax(pred))
            conf = float(np.max(pred)) * 100
            print(f"Model {i+1} -> {CLASS_NAMES[idx]} ({conf:.2f}%)")

        # --------------------------------------------------
        # Aggregate and Return
        # --------------------------------------------------
        avg_prediction = np.mean(predictions, axis=0)

        predicted_class = CLASS_NAMES[int(np.argmax(avg_prediction))]
        confidence = float(np.max(avg_prediction) * 100)

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        # Catch image processing errors (e.g., corrupt files)
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Failed to process the image. Ensure it is a valid image file."}), 500

# --------------------------------------------------
# Run server
# --------------------------------------------------
if __name__ == "__main__":
    # debug=True causes Flask to restart and load models twice. 
    # use_reloader=False prevents the double-loading of heavy TF models.
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)