from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
from PIL import Image
import io
import os
import tensorflow as tf

app = Flask(__name__)

# Load the trained model and class labels
model = load_model("crop_disease_model.h5", custom_objects={'InputLayer': tf.keras.layers.InputLayer})

with open("class_labels.pkl", "rb") as f:
    class_labels = pickle.load(f)

@app.route("/")
def home():
    return "🌿 Crop Disease Detection API is up and running!"

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    
    try:
        img = Image.open(file).convert("RGB").resize((224, 224))
    except:
        return jsonify({"error": "Invalid image format"}), 400

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    label = class_labels[pred_index]

    return jsonify({
        "prediction": label,
        "confidence": f"{confidence:.2f}"
    })

# 🔥 Bind to the correct port for Render.com
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
