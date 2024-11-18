from flask import Flask, request, jsonify, render_template, Response
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from collections import deque
import time
import telepot

app = Flask(__name__)

# Constants
IMG_SIZE = 128
FRAME_COUNT = 15
ALERT_THRESHOLD = 15
TELEGRAM_BOT_TOKEN = '7897089869:AAEeEmH0vbW4xiTi_D6Pf3C8H3qss9bTYEU'
TELEGRAM_GROUP_ID = '-1002490050732'
LOCATION = "Hyderabad"
SAVED_IMAGE_PATH = './static/images/savedImage.jpg'

# Load the pre-trained model
model = load_model('./static/Model/violence_detection.h5')

# Generate live stream frames
def generate_live_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_queue = deque(maxlen=FRAME_COUNT)
    true_count = 0
    image_saved = False
    alert_sent = False

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame
            frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame_normalized = frame_resized[:, :, [2, 1, 0]] / 255.0
            frame_queue.append(frame_normalized)

            output_frame = frame.copy()
            label = "Normal"
            text_color = (0, 255, 0)

            if len(frame_queue) == FRAME_COUNT:
                # Prepare input sequence
                input_sequence = np.expand_dims(np.array(frame_queue), axis=0)
                prediction = model.predict(input_sequence)
                if prediction[0] > 0.5:
                    label = "Violence Detected"
                    text_color = (0, 0, 255)
                    true_count += 1
                FONT = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, label, (35, 50), FONT, 1.25, text_color, 3)
                # Save an image if the threshold is met
                if true_count >= ALERT_THRESHOLD and not image_saved:
                    cv2.imwrite(SAVED_IMAGE_PATH, frame)
                    image_saved = True

                # Send alert via Telegram
                if true_count >= ALERT_THRESHOLD and not alert_sent:
                    bot = telepot.Bot(TELEGRAM_BOT_TOKEN)
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    bot.sendMessage(
                        TELEGRAM_GROUP_ID,
                        f"VIOLENCE ALERT!!\nLOCATION: {LOCATION}\nTIME: {timestamp}"
                    )
                    with open(SAVED_IMAGE_PATH, 'rb') as photo:
                        bot.sendPhoto(TELEGRAM_GROUP_ID, photo)
                    alert_sent = True

            # Display the prediction on the output frame
            FONT = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(output_frame, label, (35, 50), FONT, 1.25, text_color, 3)

            # Encode the frame for streaming
            _, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        # Release the video capture and delete the file after analysis
        cap.release()
        if os.path.exists(video_path):
            os.remove(video_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Stream from the uploaded video
    video_path = './uploads/latest_video.mp4'  # Replace with the dynamically uploaded file path
    return Response(generate_live_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    file_path = './uploads/latest_video.mp4'
    os.makedirs('./uploads', exist_ok=True)
    file.save(file_path)

    return jsonify({'message': 'Video uploaded successfully', 'video_path': file_path})

if __name__ == '__main__':
    os.makedirs('./uploads', exist_ok=True)
    app.run(debug=True, port=5050)
