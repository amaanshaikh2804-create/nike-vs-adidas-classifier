import io

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow.keras.models import load_model

IMG_SIZE = 120  # train.py jaisa hi

app = Flask(__name__)
model = load_model("nike_adidas_model.keras")


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify(error="No image received. Choose a JPG or PNG file."), 400
    try:
        img = Image.open(io.BytesIO(file.read())).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    except Exception:
        return jsonify(error="This file isn’t an image. Try a JPG or PNG."), 400

    arr = np.array(img, dtype="float32") / 255.0
    probs = model.predict(arr.reshape(1, IMG_SIZE, IMG_SIZE, 1), verbose=0)[0]
    return jsonify(nike=float(probs[0]), adidas=float(probs[1]))


if __name__ == "__main__":
    app.run(debug=False)
