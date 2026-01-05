import cv2
import os

# 🔁 YAHAN EMOTION CHANGE KARNA HAI
emotion = "happy"   # happy / sad / angry / neutral

# 📁 SAVE DIRECTORY
save_dir = f"dataset/train/{emotion}"
os.makedirs(save_dir, exist_ok=True)

# 🔢 ALREADY SAVED IMAGES COUNT (IMPORTANT FIX)
existing_images = len([
    f for f in os.listdir(save_dir)
    if f.endswith(".jpg")
])
count = existing_images

print(f"Starting from image number: {count}")

# 😀 FACE DETECTOR LOAD
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# 📷 CAMERA OPEN
cap = cv2.VideoCapture(0)

print("\nINSTRUCTIONS:")
print("C = Capture face image")
print("Q = Quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    face = None

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))

        # 🟩 GREEN BOX
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Face Crop (48x48)", face)

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    # 📸 SAVE IMAGE
    if key == ord('c') and face is not None:
        img_path = f"{save_dir}/{count}.jpg"
        cv2.imwrite(img_path, face)
        print(f"✅ Saved: {img_path}")
        count += 1

    # ❌ EXIT
    elif key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
