import cv2
import face_recognition
import insightface
import os
import argparse
import logging
from tqdm import tqdm

# Suppress ONNX Runtime warnings
os.environ['ORT_LOG_LEVEL'] = 'ERROR'

# Configure logging to suppress warnings and informational messages
logging.basicConfig(level=logging.ERROR)

# Function to get face encodings from an image
def get_face_encodings(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return face_recognition.face_encodings(image_rgb)

# Function to check if any face in target_frame matches any face in target_encodings
def identify(target_frame, target_encodings):
    frame_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
    frame_encodings = face_recognition.face_encodings(frame_rgb)
    
    for frame_encoding in frame_encodings:
        results = face_recognition.compare_faces(target_encodings, frame_encoding)
        if True in results:
            return True
    
    return False

# Function to swap faces using insightface
def swap_faces(target_frame, source_image):
    target_faces = face_analyser.get(target_frame)
    source_faces = face_analyser.get(source_image)

    if not target_faces or not source_faces:
        print("No faces found in one of the images.")
        return target_frame

    # Swap the matched face with the source face
    for target_face in target_faces:
        img_fake = model_swap_insightface.get(img=target_frame, target_face=target_face, source_face=source_faces[0], paste_back=True)
        target_frame = img_fake

    return target_frame

# Function to process video frames, identify faces, and swap faces if a match is found
def process_video_and_swap_faces(video_path, target_image_paths, source_image_path, output_video_path):
    # Load the target and source images
    target_encodings = []
    for target_image_path in target_image_paths:
        target_image = cv2.imread(target_image_path)
        if target_image is None:
            print(f"Error: Cannot open target image file {target_image_path}")
            continue

        encodings = get_face_encodings(target_image)
        if not encodings:
            print(f"No faces found in target image {target_image_path}")
            continue
        
        target_encodings.extend(encodings)
    
    if not target_encodings:
        print("No valid target faces found in any target image.")
        return

    source_image = cv2.imread(source_image_path)
    if source_image is None:
        print(f"Error: Cannot open source image file {source_image_path}")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a VideoWriter object to save the output video
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (frame_width, frame_height))

    frame_count = 0
    with tqdm(total=total_frames, desc="Processing video frames") as pbar:
        while True:
            ret, target_frame = cap.read()
            
            if not ret:
                break
            
            # Identify and swap the face if matched
            is_match = identify(target_frame, target_encodings)
            print(f"Frame {frame_count} match: {is_match}")

            if is_match:
                swapped_frame = swap_faces(target_frame, source_image)
                video_writer.write(swapped_frame)
            else:
                video_writer.write(target_frame)  # If no match, write the original frame
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    video_writer.release()
    print(f"Output video saved to {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video frames and swap faces.')
    parser.add_argument('-tv', '--videopath', type=str, required=True, help='Path to the input video file')
    parser.add_argument('-s', '--source', type=str, required=True, help='Path to the source image file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output video file')
    parser.add_argument('-t', '--target', type=str, help='Path to the target image file')
    parser.add_argument('-tf', '--targetfile', type=str, help='Path to a text file containing paths to target image files')
    
    args = parser.parse_args()
    
    # Initialize global models and analyzers
    model_path = './models/inswapper_128.onnx'
    providers = ['CPUExecutionProvider']  # Adjust based on available providers

    model_swap_insightface = insightface.model_zoo.get_model(model_path, providers=providers)
    face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))

    # Collect target image paths
    target_image_paths = []
    if args.target:
        target_image_paths.append(args.target)
    if args.targetfile:
        with open(args.targetfile, 'r') as file:
            target_image_paths.extend(file.read().splitlines())

    if not target_image_paths:
        print("Error: No target images specified. Use -t for single target file or -tf for list of target files.")
        exit(1)

    process_video_and_swap_faces(args.videopath, target_image_paths, args.source, args.output)

