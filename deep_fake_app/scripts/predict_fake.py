import tensorflow as tf
import numpy as np
import cv2
import os

# Load Keras model
MODEL_PATH = os.path.join("deep_fake_app", "models", "deepfake_detector.h5")
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_frame(frame):
    frame = cv2.resize(frame, (256, 256))
    frame = frame.astype("float32") / 255.0 
    frame = np.expand_dims(frame, axis=0) 
    return frame

def extract_frames(video_path):
    """Extract frames from the uploaded video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    return frames[:10] 

def predict_fake(video_path):
    """Predict whether the video is Real or Fake."""
    frames = extract_frames(video_path)
    processed_frames = np.vstack([preprocess_frame(frame) for frame in frames])

    predictions = model.predict(processed_frames)
    avg_prediction = np.mean(predictions) 

    result = "Fake" if avg_prediction > 0.5 else "Real"

    return result