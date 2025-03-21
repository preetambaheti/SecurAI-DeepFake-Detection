import cv2
import os

# Set paths
video_path_real = "dataset_small/real_videos"  # Path to folder containing real videos
video_path_fake = "dataset_small/fake_videos"  # Path to folder containing fake videos
output_path_real = "dataset_small/frames/real"  # Output folder for real video frames
output_path_fake = "dataset_small/frames/fake"  # Output folder for fake video frames

# Create output directories if not exist
os.makedirs(output_path_real, exist_ok=True)
os.makedirs(output_path_fake, exist_ok=True)

def extract_frames(video_path, save_path, label, interval=50):
    """Extract frames from videos at a given interval."""
    for video_file in os.listdir(video_path):
        video_full_path = os.path.join(video_path, video_file)
        cap = cv2.VideoCapture(video_full_path)
        frame_count = 0
        print(f"Processing video: {video_file}") 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0:
                frame_name = f"{label}_{video_file.split('.')[0]}_frame_{frame_count}.jpg"
                cv2.imwrite(os.path.join(save_path, frame_name), frame)
                print(f"Saved frame: {frame_name}") 

            frame_count += 1

        cap.release()

    

# Extract frames from real and fake videos
extract_frames(video_path_real, output_path_real, "real")
extract_frames(video_path_fake, output_path_fake, "fake")

print("✅ Frames extracted successfully!")
