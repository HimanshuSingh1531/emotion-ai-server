import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io

# ------------------
# MODEL DEFINITION
# ------------------
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# ------------------
# LOAD MODEL
# ------------------
checkpoint = torch.load("emotion_model.pth", map_location="cpu")
classes = checkpoint["classes"]

model = EmotionCNN(len(classes))
model.load_state_dict(checkpoint["model_state"])
model.eval()

print("✅ Model loaded with classes:", classes)

# ------------------
# FACE DETECTOR (ADDED)
# ------------------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ------------------
# FLASK APP
# ------------------
app = Flask(__name__)


@app.route("/")
def home():
    return "Emotion AI Server is running 🚀"

# ------------------
# PREDICT ROUTE
# ------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    # 🔥 ANDROID IMAGE FIX (ADDED)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 🔥 FACE DETECTION (ADDED)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return jsonify({"emotion": "neutral"})

    x, y, w, h = faces[0]
    img = gray[y:y+h, x:x+w]

    # SAME PREPROCESSING AS TRAINING
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    if confidence.item() < 0.6:
        emotion = "neutral"
    else:
        emotion = classes[predicted.item()]

    return jsonify({
        "emotion": emotion,
        "confidence": float(confidence.item())
    })


# ------------------
# RUN SERVER (RENDER COMPATIBLE)
# ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
