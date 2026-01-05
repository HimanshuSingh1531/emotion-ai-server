import torch
import torch.nn as nn
import cv2
import numpy as np
from flask import Flask, request, jsonify

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
# FLASK APP
# ------------------
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    # Decode image
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # Preprocess
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        emotion = classes[predicted.item()]

    return jsonify({
        "emotion": emotion
    })


# ------------------
# RUN SERVER
# ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
