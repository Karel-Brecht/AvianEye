import cv2
import sys
import os

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.operations.detection import BirdDetector


# Set dimensions of video frames
frame_width = 1280
frame_height = 720

detector = BirdDetector()

# Video source is MP4 file stored locally
cap = cv2.VideoCapture("downloads/videos/download.mp4")

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("No video frame avalable")
        break

    # Resize the frame
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Detect birds in the frame
    detections = detector.detect_birds(frame)
    # Visualise detections
    for detection in detections:
        bb = detection.bbox
        conf = detection.confidence
        class_name = detection.class_name
        # Draw a rectangle around the object
        cv2.rectangle(
            frame,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            (0, 255, 0),
            3,
        )

        # Add some text labelling to the rectalngle
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            class_name + " " +str(round(conf, 3)) + "%",
            (int(bb[0]), int(bb[1]) - 10),
            font,
            1,
            (255, 255, 255),
            2,
        )

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # End program when q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
