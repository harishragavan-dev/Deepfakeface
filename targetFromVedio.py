import cv2
import os
import argparse
from mtcnn import MTCNN
from tqdm import tqdm

# Initialize MTCNN detector
detector = MTCNN()

def save_face_image(face_image, save_path, count):
    """
    Save the face image to a file.
    """
    file_name = os.path.join(save_path, f"face_{count}.jpg")
    cv2.imwrite(file_name, face_image)
    return file_name

def is_face_angle_acceptable(face_box, image_width, image_height):
    """
    Check if the face is facing a direction that is acceptable (e.g., not too profile).
    This function is a placeholder and should be adapted based on your criteria.
    """
    x, y, w, h = face_box
    aspect_ratio = w / h
    return aspect_ratio > 0.5 and aspect_ratio < 2.0

def process_video(video_path, output_folder, text_file_path):
    """
    Process the video and save faces seen from different directions.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    frame_count = 0
    face_count = 0

    # Open a text file to save the paths of saved images
    with open(text_file_path, "w") as path_file:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces in the frame
            detections = detector.detect_faces(frame)

            for detection in detections:
                x, y, w, h = detection['box']
                face_image = frame[y:y+h, x:x+w]
                
                if is_face_angle_acceptable(detection['box'], frame.shape[1], frame.shape[0]):
                    face_count += 1
                    face_image_path = save_face_image(face_image, output_folder, face_count)
                    
                    # Save the path of the saved face image
                    path_file.write(f"{face_image_path}\n")
            
            frame_count += 1

    cap.release()
    print(f"Processed {frame_count} frames. Saved {face_count} faces.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and save faces from a video.')
    parser.add_argument('-v', '--video', type=str, required=True, help='Path to the input video file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output folder for saving faces')
    parser.add_argument('-t', '--textfile', type=str, required=True, help='Path to the text file for saving face image paths')
    
    args = parser.parse_args()
    
    process_video(args.video, args.output, args.textfile)
