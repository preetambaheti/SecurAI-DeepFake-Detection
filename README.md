# AI/ML-Based Detection of Face-Swap Deep Fake Videos

## 📌 Project Overview
Deepfake videos, particularly face-swapped deepfakes, pose a significant threat in spreading misinformation and identity fraud. Our AI/ML-based solution aims to detect these manipulated videos by leveraging deep learning models to classify videos as real or fake.

## 🏆 Features
- 🔍 **Deepfake Detection**: Detects face-swapped deepfake videos with high accuracy.
- 📊 **AI-Powered Analysis**: Utilizes deep learning models (EfficientNet/XceptionNet) for precise classification.
- 🎥 **Frame-by-Frame Processing**: Extracts keyframes for accurate analysis.
- 🌐 **Web-Based Interface**: Simple UI for uploading and analyzing videos.
- 📈 **Confidence Score**: Provides probability scores for fake vs. real classification.

---
## ⚙️ Tech Stack
- **Python** (TensorFlow, OpenCV, NumPy, Pandas)
- **Machine Learning** (CNN, XceptionNet, EfficientNet)
- **Flask** (For Backend API Development)
- **HTML, CSS, JavaScript** (For Frontend Development)
- **Bootstrap** (For Styling)
- **Jinja2** (For Dynamic Templating in Flask)

---
## 📂 Folder Structure
```
📁 AI_DeepFake_Detection
│-- 📁 dataset/         # Training dataset
│-- 📁 dataset_small/   # Sample dataset
│-- 📁 deep_fake_app/   # Main application folder
│   │-- 📁 models/      # Pretrained models
│   │-- 📁 templates/   # HTML templates (Frontend UI)
│   │-- 📁 static/      # Static files (CSS, JS, Images)
│   │-- app.py         # Flask Backend API
│-- 📁 scripts/         # Supporting scripts
│   │-- evaluate_model.py  # Model evaluation
│   │-- extract_frames.py  # Frame extraction from videos
│   │-- predict_fake.py    # Prediction script
│   │-- preprocess_images.py  # Preprocessing input images
│   │-- train_model.py    # Model training script
│-- 📁 uploads/         # Uploaded videos for detection
│-- README.md          # Project Documentation
│-- requirements.txt   # Dependencies
```

---
## 🚀 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/SecurAI-DeepFake-Detection/AI-DeepFake-Detection.git
cd AI-DeepFake-Detection
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---
## 🔧 How to Run the Code
### 🖥️ Backend (Flask API)
1. Navigate to the `deep_fake_app/` folder:
```bash
cd deep_fake_app
```
2. Run the Flask server:
```bash
python app.py
```
3. The API will be available at:
```
http://127.0.0.1:5000/
```

### 🌐 Frontend (Web UI)
1. Ensure the Flask backend is running.
2. Open the browser and visit:
```
http://127.0.0.1:5000/
```

---
## 🎥 How to Use the Application
1. **Upload a Video** 📤
   - Click on the **Upload** button and select a video.
2. **Processing** ⏳
   - The AI model extracts frames and runs deepfake detection.
3. **View Results** 📊
   - The system displays whether the video is real or fake along with a confidence score.

---
## 🛠️ Training the Model (Optional)
1. Ensure you have the required dataset inside the `dataset/` folder.
2. Run the training script:
```bash
python scripts/train_model.py
```
3. The trained model will be saved in `deep_fake_app/models/` folder.

---
## 🔗 Deployment
To deploy on a cloud platform like **Heroku, AWS, or GCP**, use the following steps:
1. Create a **requirements.txt** & **Procfile**
2. Push the code to a GitHub repository
3. Deploy using Heroku CLI:
```bash
heroku create deepfake-detector
heroku git:remote -a deepfake-detector
heroku buildpacks:add heroku/python
heroku push origin main
```
4. Open the deployed app URL.

---