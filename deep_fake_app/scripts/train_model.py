import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# ✅ Define Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "dataset_small", "processed_frames", "train")
VALSET_PATH = os.path.join(BASE_DIR, "dataset_small", "processed_frames", "val")

# ✅ Load Preprocessed Data
print("📌 Loading preprocessed dataset...")
X_train = np.load(os.path.join(DATASET_PATH, "X_train.npy"))
y_train = np.load(os.path.join(DATASET_PATH, "y_train.npy"))
X_val = np.load(os.path.join(VALSET_PATH, "X_val.npy"))
y_val = np.load(os.path.join(VALSET_PATH, "y_val.npy"))

# ✅ Define Image Shape (Ensure Consistency)
IMG_SIZE = (256, 256, 3)

# ✅ Data Augmentation for Training Data
train_datagen = ImageDataGenerator(
    rotation_range=20,        
    width_shift_range=0.2,    
    height_shift_range=0.2,   
    zoom_range=0.2,           
    horizontal_flip=True,     
    brightness_range=[0.8, 1.2]
)

# ✅ No Augmentation for Validation Data
val_datagen = ImageDataGenerator()

# ✅ Apply Augmentation to Training Data
X_train = np.squeeze(X_train)  
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

# ✅ Apply Validation Data
X_val = np.squeeze(X_val) 
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

# ✅ Load Pretrained EfficientNetB0 Model
base_model = EfficientNetB0(input_shape=(256, 256, 3), include_top=False, weights="imagenet")
base_model.trainable = False 

# ✅ Build Model
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation="relu", kernel_regularizer=l2(0.01)), 
    Dropout(0.5), 
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy", 
    metrics=["accuracy"]
)

# ✅ Early Stopping: Stop training when validation loss stops improving
early_stopping = EarlyStopping(
    monitor="val_loss",  
    patience=5,          
    restore_best_weights=True
)

# ✅ Train Model
print("🚀 Training the model...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50, 
    batch_size=32,
    callbacks=[early_stopping]
)

print(f"Real Videos in Training: {np.sum(y_train == 0)}")
print(f"Fake Videos in Training: {np.sum(y_train == 1)}")
print(f"Real Videos in Validation: {np.sum(y_val == 0)}")
print(f"Fake Videos in Validation: {np.sum(y_val == 1)}")

# ✅ Save Model
model.save(os.path.join(BASE_DIR, "models", "deepfake_detector.h5"))
print("✅ Model Training Complete! Model saved as deepfake_detector.h5")
