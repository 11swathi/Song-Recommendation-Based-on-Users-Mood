from flask import Flask, render_template, Response, redirect, url_for, request
import cv2
import numpy as np
import tensorflow as tf
import pickle

app = Flask(__name__)

# Load the pre-trained emotion detection model
emotion_model = tf.keras.models.load_model('emotion_model.h5')

# Load the emotion to songs mapping
with open('emotion_to_songs.pkl', 'rb') as f:
    emotion_to_songs = pickle.load(f)

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the camera
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.resize(frame, (1000, 600))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    success, frame = camera.read()
    if success:
        # Convert the captured frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load the Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)
                
                # Predict emotion
                prediction = emotion_model.predict(roi_gray)
                emotion_label = emotion_labels[np.argmax(prediction)]
                
                # Get recommended songs, limit to 25
                recommended_songs = emotion_to_songs.get(emotion_label, [])[:25]
                
                return render_template('result.html', emotion=emotion_label, songs=recommended_songs)
        else:
            return render_template('index.html', error="No face detected. Please try again.")
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
