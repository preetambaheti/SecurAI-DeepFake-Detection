import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Define Augmentation Strategy
datagen = ImageDataGenerator(
    rotation_range=20,        
    width_shift_range=0.2,    
    height_shift_range=0.2,   
    zoom_range=0.2,           
    horizontal_flip=True,     
    brightness_range=[0.8, 1.2]
)

# ✅ Define Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "dataset_small", "frames")
OUTPUT_PATH = os.path.join(BASE_DIR, "dataset_small", "processed_frames")

# ✅ Ensure Output Directory Exists
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "val"), exist_ok=True)

# ✅ Image Preprocessing Settings
IMG_SIZE = (256, 256) 

# ✅ Load and Augment Images
def load_images(label, folder, apply_augmentation=False):
    images = []
    labels = []
    folder_path = os.path.join(DATASET_PATH, folder)

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        image = cv2.imread(file_path)

        if image is None:
            print(f"❌ Skipping corrupted file: {file_name}")
            continue

        # ✅ Resize and Normalize Image
        image = cv2.resize(image, IMG_SIZE)
        image = image / 255.0 

        # ✅ Apply Augmentation only for Training Data
        if apply_augmentation:
            image = np.expand_dims(image, axis=0) 
            image = datagen.flow(image, batch_size=1)[0]  

        images.append(image)
        labels.append(label)  

    return np.array(images), np.array(labels)

# ✅ Load Data (With Augmentation for Training, Without for Validation)
print("📌 Loading and augmenting real images for training...")
real_images, real_labels = load_images(label=0, folder="real", apply_augmentation=True)

print("📌 Loading and augmenting fake images for training...")
fake_images, fake_labels = load_images(label=1, folder="fake", apply_augmentation=True)

# ✅ Combine Datasets
X = np.concatenate((real_images, fake_images), axis=0)
y = np.concatenate((real_labels, fake_labels), axis=0)

# ✅ Split Data (80% Train, 20% Validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Save Preprocessed Data as Numpy Arrays (Faster Training)
np.save(os.path.join(OUTPUT_PATH, "train", "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_PATH, "train", "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_PATH, "val", "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_PATH, "val", "y_val.npy"), y_val)

print("✅ Preprocessing & Augmentation Complete! Processed data saved in dataset_small/processed_frames")
