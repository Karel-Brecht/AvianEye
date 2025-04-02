import os
import subprocess
import tempfile

import cv2

def calculate_iou(bbox1, bbox2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0


def add_audio_to_video(source_video_path: str, target_video_path: str):
    """
    Adds audio from source_video_path to the target_video_path.
    Overwrites the target video with the new version containing audio.
    """
    
    if not os.path.exists(source_video_path):
        print(f"Warning: Source video file {source_video_path} not found. Adding audio failed.")
        return
        
    # Create a temporary file for the output with audio using mkstemp
    temp_output_fd, temp_output = tempfile.mkstemp(suffix='.mp4')
    os.close(temp_output_fd)  # Close the file descriptor, we only need the file name
    
    cmd = [
        'ffmpeg',
        '-i', target_video_path,        # Input video (no audio)
        '-i', source_video_path,        # Input original video (with audio)
        '-map', '0:v',                  # Use video from first input
        '-map', '1:a',                  # Use audio from second input
        '-c:v', 'copy',                 # Copy the video stream
        '-c:a', 'copy',                 # Copy the audio stream
        '-y',                           # Overwrite output file if it exists
        temp_output                     # Temporary output file
    ]
    
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        # Replace the original output with the version that has audio
        os.replace(temp_output, target_video_path)
        print(f"Added audio to {target_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error while adding audio: {e}")
        print("Adding audio failed.")
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg to add audio.")
        print("Adding audio failed.")

    finally:
        # Clean up the temporary file in case of error
        if os.path.exists(temp_output):
            os.remove(temp_output)

def resize_to_fit(image, target_width, target_height):
    """
    Resize the image to fit within the target dimensions while maintaining aspect ratio.
    Return image and new dimensions.

    Args:
        - image: Input image
        - target_width: Desired width
        - target_height: Desired height

    Returns:
        - Resized image
        - New width
        - New height
    """

    h, w = image.shape[:2]

    if h <= target_height and w <= target_width:
        return image, w, h

    aspect = w / h
    target_aspect = target_width / target_height
    
    if aspect > target_aspect:  # Image is wider than target
        new_width = target_width
        new_height = int(target_width / aspect)
    else:  # Image is taller than target
        new_height = target_height
        new_width = int(target_height * aspect)
        
    return cv2.resize(image, (new_width, new_height)), new_width, new_height
