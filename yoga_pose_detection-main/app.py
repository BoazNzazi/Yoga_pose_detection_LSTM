import os
import cv2
import numpy as np
import pickle
import torch
import tensorflow as tf
import mediapipe as mp
from flask import Flask, request, render_template

# Init Flask
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load LSTM model and preprocessors
model = tf.keras.models.load_model("yoga_pose_model_lstm.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load MediaPipe and YOLO
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

def detect_persons(image):
    results = yolo(image)
    persons = []

    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == 0:  # 'person' class in COCO
            x1, y1, x2, y2 = map(int, box)
            persons.append((x1, y1, x2, y2))

    return persons

def extract_keypoints(crop):
    image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    return keypoints if len(keypoints) == 99 else None, results

def predict_pose_from_crop(crop):
    keypoints, results = extract_keypoints(crop)
    if keypoints:
        x = np.array(keypoints).reshape(1, -1)
        x = scaler.transform(x)
        x = x.reshape(1, 33, 3)
        pred = model.predict(x)
        predicted_class = encoder.inverse_transform([np.argmax(pred)])[0]
        confidence = np.max(pred) * 100
        return predicted_class, confidence, results
    return None, None, None

def draw_results(image, persons):
    for i, (x1, y1, x2, y2) in enumerate(persons):
        crop = image[y1:y2, x1:x2]
        predicted_class, confidence, results = predict_pose_from_crop(crop)

        if results and results.pose_landmarks:
            offset_landmarks(results, x1, y1, crop.shape, image.shape)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        label = f"{predicted_class} ({confidence:.1f}%)" if predicted_class else "Pose inconnue"
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    annotated_path = os.path.join(app.config["UPLOAD_FOLDER"], "annotated.jpg")
    cv2.imwrite(annotated_path, image)
    return annotated_path

def offset_landmarks(results, x_offset, y_offset, crop_size, image_size):
    """Décale les landmarks détectés pour qu’ils soient cohérents avec l’image originale."""
    for i in range(len(results.pose_landmarks.landmark)):
        lm = results.pose_landmarks.landmark[i]
        lm.x = (lm.x * crop_size[1] + x_offset) / image_size[1]
        lm.y = (lm.y * crop_size[0] + y_offset) / image_size[0]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename != "":
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded.jpg")
            file.save(image_path)

            image = cv2.imread(image_path)
            persons = detect_persons(image)
            if not persons:
                return render_template("index.html", error="Aucune personne détectée.")

            result_img_path = draw_results(image, persons)
            return render_template("index.html", image_path=result_img_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
