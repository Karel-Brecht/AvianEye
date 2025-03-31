from dataclasses import dataclass
import cv2
from PIL import Image

from .download import Downloader
from .detection import BirdDetector, Detection
from .classification import BirdClassifier

        
@dataclass
class Classification:
    """Represents a single detection with its bounding box, confidence, and class name."""
    detection: Detection
    predicted_cls: str  # Class name of the detected object
    predicted_prob: float  # Confidence score of the detection
    all_probs: dict[str, float]  # Probabilities for each class


class Frame:
    """Represents a single frame of the video, including detections and classifications."""

    def __init__(self, frame_id: int, image):
        self.frame_id = frame_id
        self.image = image
        self.detections: list[Detection] = []
        self.classifications: list[Classification] = []  # List of classifications (e.g., bird species)

    def add_detection(self, detection: Detection):
        """Adds a detection to the frame."""
        self.detections.append(detection)
        return detection

    def add_classification(self, detection: Detection, predicted_cls: str, predicted_prob: float, all_probs: dict[str, float]):
        """Adds a classification to the identification."""
        classification = Classification(detection, predicted_cls, predicted_prob, all_probs)
        self.classifications.append(classification)
        return classification
    
    def remove_detections(self):
        """Removes all detections from the frame."""
        self.detections = []

    def remove_classifications(self):
        """Removes all classifications from the frame."""
        self.classifications = []


class VideoProcessor:
    """Handles the full pipeline: downloading, extracting frames, detecting birds, and classifying species."""

    def __init__(self, video_link: str, downloader: Downloader, detector_model: BirdDetector, classifier_model: BirdClassifier):
        self.size = (1280, 720)

        self.video_link = video_link
        self.downloader = downloader
        self.detector_model = detector_model
        self.classifier_model = classifier_model

        self.video_path = None
        self.frames: list[Frame] = []
        self.frame_rate = None

    def remove_frames(self):
        """Removes all frames from the video."""
        self.frames = []

    def remove_detections(self):
        """Removes all detections from the video."""
        for frame in self.frames:
            frame.remove_detections()

    def remove_classifications(self):
        """Removes all classifications from the video."""
        for frame in self.frames:
            frame.remove_classifications()

    # LOAD VIDEO

    def download_video(self):
        """Downloads the video from the given link."""
        self.video_path = self.downloader.download_video(self.video_link)
        if not self.video_path:
            raise Exception("Failed to download video.")
        return self.video_path

    def extract_frames(self):
        """Extracts frames from the video file."""
        cap = cv2.VideoCapture(self.video_path)
        self.frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to a standard size
            frame = cv2.resize(frame, self.size)

            self.frames.append(Frame(frame_id, frame))
            frame_id += 1
        cap.release()

    # PROCESSING

    def detect_birds(self, annotate: bool = False):
        """Runs object detection on each frame to detect birds."""
        for frame in self.frames:
            detections = self.detector_model.detect_birds(frame.image)  # Returns [(bbox, confidence), ...]
            for detection in detections:
                frame.add_detection(detection)

        if annotate:
            self.annotate_detections()

    def classify_birds(self, annotate: bool = False):
        """Classifies the detected birds in each frame."""  
        for frame in self.frames:
            for detection in frame.detections:
                # Crop the detected bird from the frame
                x1, y1, x2, y2 = detection.bbox
                bird_image = frame.image[y1:y2, x1:x2]

                # Convert to PIL Image for classification
                bird_image_pil = cv2.cvtColor(bird_image, cv2.COLOR_BGR2RGB)
                bird_image_pil = Image.fromarray(bird_image_pil)

                predicted_cls, predicted_prob, all_probs = self.classifier_model.classify(bird_image_pil)
                frame.add_classification(detection, predicted_cls, predicted_prob, all_probs)

        if annotate:
            self.annotate_classifications()

    # ANNOTATIONS

    def annotate_detections(self):
        """Runs object detection on each frame to detect birds."""
        for frame in self.frames:
            for detection in frame.detections:
                cv2.rectangle(
                    frame.image,
                    (detection.bbox[0],detection.bbox[1]),
                    (detection.bbox[2], detection.bbox[3]),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame.image,
                    f"{detection.class_name} {detection.confidence:.2f}",
                    (detection.bbox[0], detection.bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, 
                    (255, 255, 255),
                    2
                )

    def annotate_classifications(self):
        """Annotates the classifications on the frames."""
        for frame in self.frames:
            for classification in frame.classifications:
                cv2.rectangle(
                    frame.image,
                    (classification.detection.bbox[0], classification.detection.bbox[1]),
                    (classification.detection.bbox[2], classification.detection.bbox[3]),
                    (255, 0, 0),
                    2
                )
                cv2.putText(
                    frame.image,
                    f"{classification.predicted_cls} {classification.predicted_prob:.2f}",
                    (classification.detection.bbox[0], classification.detection.bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, 
                    (255, 255, 255),
                    2
                )

    def annotate_frame_ids(self):
        """Annotates the frame IDs on the frames."""
        for frame in self.frames:
            cv2.putText(
                frame.image,
                f"Frame ID: {frame.frame_id}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, 
                (255, 255, 255),
                2
            )
                
    # EXPORTS

    def export_frames(self, output_dir: str):
        """Exports the frames with detections and classifications to the specified directory."""
        for frame in self.frames:
            output_path = f"{output_dir}/frame_{frame.frame_id}.jpg"
            cv2.imwrite(output_path, frame.image)
            print(f"Saved frame {frame.frame_id} to {output_path}")

    def export_frame(self, frame_id: int, output_dir: str):
        """Exports a single frame with detections and classifications."""
        if frame_id < len(self.frames):
            frame = self.frames[frame_id]
            output_path = f"{output_dir}/frame_{frame.frame_id}.jpg"
            cv2.imwrite(output_path, frame.image)
            print(f"Saved frame {frame_id} to {output_path}")
        else:
            print(f"Frame {frame_id} does not exist.")

    def export_video(self, output_path: str):
        """Exports the processed video with detections and classifications."""
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, self.frame_rate, self.size)

        for frame in self.frames:
            out.write(frame.image)

        out.release()
        print(f"Saved processed video to {output_path}")
