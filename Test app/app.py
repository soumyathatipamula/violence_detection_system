# app.py
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('../Models/violence_detection.h5')

# Constants for video preprocessing
IMG_SIZE = 128
FRAME_COUNT = 15

def load_video(path, nframes=FRAME_COUNT, size=(IMG_SIZE, IMG_SIZE)):
    frames = []
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(1, (total_frames // nframes) - 1)

    for _ in range(nframes):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frame = frame[:, :, [2, 1, 0]] / 255.0
        frames.append(frame)
        for _ in range(skip_frames):
            cap.grab()
    cap.release()
    return frames if len(frames) == nframes else None

def predict_video_class(video_path):
    frames = load_video(video_path)
    if frames is None:
        return "Insufficient frames"
    input_data = np.expand_dims(np.array(frames), axis=0)
    prediction = model.predict(input_data)
    return "Violence" if prediction >= 0.5 else "Normal"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    filepath = os.path.join('./uploads', file.filename)
    file.save(filepath)
    result = predict_video_class(filepath)
    os.remove(filepath)  # Clean up uploaded file
    return jsonify({'result': result})

if __name__ == '__main__':
    os.makedirs('./uploads', exist_ok=True)
    app.run(debug=True,port=5050)
