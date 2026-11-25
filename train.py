"""Training script: notebook ka poora code ek jagah. Run: python train.py"""
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras import layers, models

# ---- Apne folder ke hisaab se badlein ----
BASE = Path(r"E:\Deep_Learning_Project\NIKE_vs_ADIDAS-master")
TRAIN_DIR = BASE / "TRAIN"
TEST_DIR = BASE / "TEST"
IMG_SIZE = 120
EPOCHS = 15
MODEL_FILE = "nike_adidas_model.keras"


def label_img(name):
    prefix = name.split("_")[0].upper()
    if prefix == "NIKE":
        return [1, 0]
    if prefix == "ADIDAS":
        return [0, 1]
    return None


def load_folder(folder):
    if not folder.exists():
        raise SystemExit(f"Folder nahi mila: {folder}\nMilne wale folders: {os.listdir(BASE)}")
    data = []
    for name in os.listdir(folder):
        label = label_img(name)
        if label is None:
            continue
        img = Image.open(folder / name).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        data.append((np.array(img, dtype="float32") / 255.0, label))
    random.shuffle(data)
    x = np.array([d[0] for d in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array([d[1] for d in data])
    return x, y


x_train, y_train = load_folder(TRAIN_DIR)
x_test, y_test = load_folder(TEST_DIR)
print(f"Train: {len(x_train)} images, Test: {len(x_test)} images")

model = models.Sequential([
    layers.Conv2D(16, 3, activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(2, activation="softmax"),
])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32)

loss, acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {acc:.2%}")

model.save(MODEL_FILE)
print(f"Model save ho gaya: {MODEL_FILE}")
