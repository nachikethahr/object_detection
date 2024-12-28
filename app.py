from flask import Flask, Response, render_template, jsonify
import cv2
import numpy as np
from gtts import gTTS
import os
import time

app = Flask(__name__)

# Load the SSD MobileNet model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

# Class labels for COCO dataset
classNames = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 
              5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 
              10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 
              14: 'motorbike', 15: 'person', 16: 'pottedplant', 
              17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

detected_object = None  # Variable to store the last detected object
audio_feedback_enabled = True  # Toggle for audio feedback

def detect_objects(frame):
    global detected_object
    # Prepare the image for the model
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    max_confidence = 0
    detected_label = None

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            if confidence > max_confidence:  # Check for the highest confidence
                max_confidence = confidence
                class_id = int(detections[0, 0, i, 1])
                detected_label = classNames[class_id]

    if detected_label and detected_label != detected_object:  # If a new object was detected
        detected_object = detected_label
        if audio_feedback_enabled:
            speak(detected_object)  # Speak the detected object
        time.sleep(2)  # Delay for 2 seconds

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    
    # Play the audio file using VLC
    os.system("vlc --play-and-exit output.mp3")  # Use --play-and-exit to close VLC after playing

def generate_frames():
    camera = cv2.VideoCapture(0)  # Open the default camera
    while True:
        success, frame = camera.read()  # Read a frame from the camera
        if not success:
            break
        
        detect_objects(frame)  # Detect objects in the frame

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Return the frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected_object')
def get_detected_object():
    return jsonify(detected_object=detected_object)

@app.route('/toggle_audio')
def toggle_audio():
    global audio_feedback_enabled
    audio_feedback_enabled = not audio_feedback_enabled
    return jsonify(audio_feedback_enabled=audio_feedback_enabled)

if __name__ == '__main__':
    app.run(debug=True)