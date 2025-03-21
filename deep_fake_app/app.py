from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
import os
import time

from scripts.predict_fake import predict_fake  # Import AI function

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Store the uploaded filename in session to access in processing route
            session['filename'] = file.filename

            # Redirect to processing page before prediction
            return redirect(url_for('processing'))
    
    return render_template('upload.html')

@app.route('/processing')
def processing():
    filename = session.get('filename', None)
    if not filename:
        return redirect(url_for('upload')) 
    
    return render_template('processing.html', filename=filename)

@app.route('/process_video')
def process_video():
    """Simulate video processing and then redirect to results."""
    filename = session.get('filename', None)
    if not filename:
        return redirect(url_for('upload')) 
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Run AI model
    prediction = predict_fake(video_path)

    return redirect(url_for('result', filename=filename, prediction=prediction))

@app.route('/result')
def result():
    filename = request.args.get('filename')
    prediction = request.args.get('prediction')
    return render_template('result.html', filename=filename, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
