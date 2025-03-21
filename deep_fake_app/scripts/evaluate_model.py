import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ✅ Define Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "dataset_small", "processed_frames")
MODEL_PATH = os.path.join(BASE_DIR, "models", "deepfake_detector.h5")

# ✅ Load Preprocessed Validation Dataset
print("📌 Loading validation dataset...")
X_val = np.load(os.path.join(DATASET_PATH, "val", "X_val.npy"))
y_val = np.load(os.path.join(DATASET_PATH, "val", "y_val.npy"))

print(f"✅ Loaded validation data: {X_val.shape} images")

# ✅ Load the Trained Model
print("📌 Loading trained model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ✅ Make Predictions
y_pred_prob = model.predict(X_val) 
y_pred = (y_pred_prob > 0.5).astype("int32") 

# ✅ Calculate Accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"🎯 Model Accuracy on Validation Set: {accuracy * 100:.2f}%")

# ✅ Generate Detailed Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_val, y_pred, target_names=["Real", "Fake"]))

# ✅ Generate Confusion Matrix
print("\n🔎 Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
