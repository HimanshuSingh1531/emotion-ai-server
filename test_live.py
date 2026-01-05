import cv2
import torch
import torch.nn as nn
import numpy as np

# ------------------
# MODEL DEFINITION (SAME AS TRAINING)
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
# LOAD TRAINED MODEL
# ------------------
checkpoint = torch.load("emotion_model.pth", map_location="cpu")
classes = checkpoint["classes"]

model = EmotionCNN(len(classes))
model.load_state_dict(checkpoint["model_state"])
model.eval()

print("✅ Model loaded with classes:", classes)

# ------------------
# FACE DETECTOR
# ------------------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ------------------
# CAMERA
# ------------------
cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        # ---- FACE CROP ----
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))

        # ---- PREPROCESS (EXACTLY SAME AS TRAINING) ----
        face = face.astype("float32") / 255.0
        face = (face - 0.5) / 0.5   # NORMALIZATION MATCH

        face_tensor = torch.tensor(face).unsqueeze(0).unsqueeze(0)

        # ---- PREDICT ----
        with torch.no_grad():
            outputs = model(face_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

            emotion = classes[predicted.item()]
            conf = confidence.item()

        # ---- CONFIDENCE THRESHOLD ----
        if conf < 0.60:
            emotion_text = "uncertain"
        else:
            emotion_text = f"{emotion} ({conf:.2f})"

        # ---- DRAW RESULTS ----
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            emotion_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Live Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
