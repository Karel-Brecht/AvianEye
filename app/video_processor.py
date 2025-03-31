from typing import Optional
from collections import defaultdict
from dataclasses import dataclass
import cv2
from PIL import Image
import numpy as np

from .download import Downloader
from .detection import BirdDetector, Detection
from .classification import BirdClassifier
from .utils import calculate_iou

        
@dataclass
class Classification: # TODO: rename as Observation
    """Represents a single detection with its bounding box, confidence, and class name."""
    detection: Detection
    predicted_cls: str  # Class name of the detected object
    predicted_prob: float  # Confidence score of the detection
    all_probs: dict[str, float]  # Probabilities for each class
    taken: bool = False  # If detection is taken as a true detection
    taken_cls: str = None  # The actual class that it has been taken for
    tracking_id: Optional[int] = None  # ID to track object across frames


class Frame:
    """Represents a single frame of the video, including detections and classifications."""

    def __init__(self, frame_id: int, image):
        self.frame_id = frame_id
        self.image = image
        self.detections: list[Detection] = []
        self.classifications: list[Classification] = []  # List of classifications (e.g., bird species) # TODO: rename as observations
        self.detected_species: dict[str, int] = {}  # Dictionary to store species counts

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

    def remove_detected_species(self):
        """Removes all detected species from the frame."""
        self.detected_species = {}
        for classification in self.classifications:
            classification.taken = False
            classification.taken_cls = None
            classification.tracking_id = None


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
        
        # Parameters for tracking and temporal smoothing
        self.iou_threshold = 0.5  # IoU threshold for tracking # TODO: fine-tune this a bit
        self.temporal_window_size = 30  # Number of frames to consider for temporal smoothing # TODO: fine-tune this a bit
        self.min_detection_confidence = 0.10  # Minimum detection confidence # TODO: is already set in detector model
        self.min_classification_confidence = 0.0  # Minimum classification confidence # TODO: fine-tune this a bit
        
        # Track information
        self.next_track_id = 0
        self.tracks = {}  # track_id -> list of classifications across frames

        self.summary_statistics = None

    def remove_frames(self):
        """Removes all frames from the video."""
        self.frames = []
        self.next_track_id = 0
        self.tracks = {}
        self.summary_statistics = None

    def renew_frames(self):
        """Renews the frames from the video."""
        old_frames = self.frames
        self.frames = []
        self.extract_frames()
        for i, frame in enumerate(self.frames):
            if i < len(old_frames):
                frame.detections = old_frames[i].detections
                frame.classifications = old_frames[i].classifications
                frame.detected_species = old_frames[i].detected_species
        self.next_track_id = 0
        self.tracks = {}
        self.summary_statistics = None

    def remove_detections(self):
        """Removes all detections from the video."""
        for frame in self.frames:
            frame.remove_detections()

    def remove_classifications(self):
        """Removes all classifications from the video."""
        for frame in self.frames:
            frame.remove_classifications()

    def remove_detected_species(self):
        """Removes all detected species from the video."""
        for frame in self.frames:
            frame.remove_detected_species()
            self.next_track_id = 0
            self.tracks = {}
            self.summary_statistics = None

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

    def classify_birds(self, padding: float = -1, annotate: bool = False):
        """
        Classifies the detected birds in each frame.
        
        Args:
            padding (float): Padding to add around the detected bird for classification. 
                             If negative, no padding is added. Otherwise, it adds a percentage of the bounding box size.
            annotate (bool): Whether to annotate the classifications on the frames.
        """  
        for frame in self.frames:
            for detection in frame.detections:
                # Crop the detected bird from the frame
                x1, y1, x2, y2 = detection.bbox

                if padding > 0:
                    # Add percentual padding where possible
                    h_padding = int((y2 - y1) * padding)
                    w_padding = int((x2 - x1) * padding)
                    y1 = max(0, y1 - h_padding)
                    y2 = min(frame.image.shape[0], y2 + h_padding)
                    x1 = max(0, x1 - w_padding)
                    x2 = min(frame.image.shape[1], x2 + w_padding)

                bird_image = frame.image[y1:y2, x1:x2]

                # Convert to PIL Image for classification
                bird_image_pil = cv2.cvtColor(bird_image, cv2.COLOR_BGR2RGB)
                bird_image_pil = Image.fromarray(bird_image_pil)

                predicted_cls, predicted_prob, all_probs = self.classifier_model.classify(bird_image_pil)
                frame.add_classification(detection, predicted_cls, predicted_prob, all_probs)

        if annotate:
            self.annotate_classifications()

###########################################################################################################################

    def assign_tracking_ids(self):
        """Assign tracking IDs to detections across frames based on IoU, ensuring only one classification per track ID per frame."""
        if not self.frames:
            return
            
        # Process first frame
        for classification in self.frames[0].classifications:
            if classification.detection.confidence >= self.min_detection_confidence:
                classification.tracking_id = self.next_track_id
                self.tracks[self.next_track_id] = [classification]
                self.next_track_id += 1
        
        # Process subsequent frames
        for i in range(1, len(self.frames)):
            current_frame = self.frames[i]
            prev_frame = self.frames[i-1]
            
            # Step 1: Calculate all possible matches between current and previous frames
            possible_matches = []
            
            for curr_idx, current_classification in enumerate(current_frame.classifications):
                if current_classification.detection.confidence < self.min_detection_confidence:
                    continue
                
                for prev_classification in prev_frame.classifications:
                    if prev_classification.tracking_id is None:
                        continue
                    
                    iou = calculate_iou(
                        current_classification.detection.bbox,
                        prev_classification.detection.bbox
                    )
                    
                    if iou > self.iou_threshold:
                        possible_matches.append((
                            curr_idx, 
                            prev_classification.tracking_id, 
                            iou
                        ))
            
            # Step 2: Sort matches by IoU (highest first)
            possible_matches.sort(key=lambda x: x[2], reverse=True)
            
            # Step 3: Assign track IDs, ensuring one classification per track ID
            assigned_curr_idxs = set()
            assigned_track_ids = set()
            
            for curr_idx, track_id, iou in possible_matches:
                # Skip if this current classification or track ID has already been assigned
                if curr_idx in assigned_curr_idxs or track_id in assigned_track_ids:
                    continue
                
                # Assign track ID to current classification
                current_classification = current_frame.classifications[curr_idx]
                current_classification.tracking_id = track_id
                
                # Update tracking information
                self.tracks[track_id].append(current_classification)
                
                # Mark as assigned
                assigned_curr_idxs.add(curr_idx)
                assigned_track_ids.add(track_id)
            
            # Step 4: Create new tracks for unassigned detections
            for curr_idx, current_classification in enumerate(current_frame.classifications):
                if (curr_idx not in assigned_curr_idxs and 
                    current_classification.detection.confidence >= self.min_detection_confidence):
                    # Create new track
                    current_classification.tracking_id = self.next_track_id
                    self.tracks[self.next_track_id] = [current_classification]
                    self.next_track_id += 1
    
    def smooth_classifications(self):
        """Apply temporal smoothing to classifications based on tracking data."""
        tracks_to_remove = []
        for track_id, classifications in self.tracks.items():

            # If tracking is too short, remove it
            if len(classifications) < 3:
                # print(f"Track {track_id} is too short, removing it.")
                for classification in classifications:
                    classification.tracking_id = None
                tracks_to_remove.append(track_id)
                continue

            # If track has only one detection, no smoothing needed
            if len(classifications) <= 1:
                continue
                
            # For each classification in the track
            for i, classification in enumerate(classifications):
                # Determine window size (adjust for beginning and end of track)
                window_start = max(0, i - self.temporal_window_size // 2)
                window_end = min(len(classifications), i + self.temporal_window_size // 2 + 1)
                window = classifications[window_start:window_end]
                
                # Count class votes weighted by probability
                class_votes = defaultdict(float)
                for cls in window:
                    for class_name, prob in cls.all_probs.items():
                        class_votes[class_name] += prob
                
                # Get class with highest votes
                best_class = max(class_votes.items(), key=lambda x: x[1])
                
                # Only update if the weighted vote is confident enough
                if best_class[1] / len(window) >= self.min_classification_confidence:
                    # TODO: should this update the predicted_cls and predicted_prob or the taken_cls?
                    classification.predicted_cls = best_class[0]
                    classification.predicted_prob = best_class[1] / len(window)

                # TODO: remove tracks that too ambiguous
        
        # Remove the short tracks from the tracks dictionary
        for track_id in tracks_to_remove:
            del self.tracks[track_id]


    
    def process_observations(self):
        """Processes the observations to count species with temporal consistency."""
        # Step 1: Assign tracking IDs to detections across frames
        self.assign_tracking_ids()
        
        # Step 2: Smooth classifications based on temporal consistency
        self.smooth_classifications()
        
        # Step 3: Count unique individuals and species per frame
        for frame in self.frames:
            # Reset detected species count
            frame.detected_species = {}
            
            # Process only classifications with tracking IDs and sufficient confidence
            processed_track_ids = set()
            for classification in frame.classifications:
                # Skip if no tracking ID or already processed this track in this frame
                if (classification.tracking_id is None or 
                    classification.tracking_id in processed_track_ids or
                    classification.detection.confidence < self.min_detection_confidence or
                    classification.predicted_prob < self.min_classification_confidence):
                    continue
                    
                # Mark this track as processed for this frame
                processed_track_ids.add(classification.tracking_id)

                # TODO: what is it doing with duplicate track_ids in the same frame
                
                # Update species count
                species = classification.predicted_cls
                if species not in frame.detected_species:
                    frame.detected_species[species] = 0
                frame.detected_species[species] += 1
                
                # Mark as taken
                classification.taken = True
                classification.taken_cls = species
            
            # Sort dictionary by key
            frame.detected_species = dict(sorted(frame.detected_species.items()))
        
        # Optional: Aggregate statistics across all frames
        self._generate_summary_statistics()





    
    def _generate_summary_statistics(self):
        """Generate summary statistics for the entire video."""
        species_counts = defaultdict(int)
        species_confidence = defaultdict(list)
        unique_tracks = set()
        
        for track_id, classifications in self.tracks.items():
            # Consider only classifications with sufficient confidence
            valid_classifications = [c for c in classifications if c.taken]
            
            if len(valid_classifications) == 0:
                print(f"Track {track_id} has no valid classifications.")
                continue
                
            # Count votes for most frequent species in this track
            species_votes = defaultdict(int)
            for c in valid_classifications:
                species_votes[c.predicted_cls] += 1
            
            # Get most frequently predicted species for this track
            if species_votes:
                most_common_species = max(species_votes.items(), key=lambda x: x[1])[0]
                species_counts[most_common_species] += 1
                unique_tracks.add(track_id)
                
                # Collect confidence values
                for c in valid_classifications:
                    if c.predicted_cls == most_common_species:
                        species_confidence[most_common_species].append(c.predicted_prob)
        
        # Store summary statistics
        self.summary_statistics = {
            "total_unique_birds": len(unique_tracks),
            "total_unique_species": len(species_counts),
            "species_counts": dict(sorted(species_counts.items())),
            "average_confidence": {
                species: sum(confs)/len(confs) if confs else 0
                for species, confs in species_confidence.items()
            }
        }

    def get_summary_statistics(self):
        """Returns the summary statistics for the entire video."""
        if not hasattr(self, 'summary_statistics'):
            self._generate_summary_statistics()
        return self.summary_statistics
    


###########################################################################################################################
    

    def process_observations_simple(self):
        """Processes the observations to count species."""
        for frame in self.frames:
            for classification in frame.classifications:
                species = classification.predicted_cls
                if species not in frame.detected_species:
                    frame.detected_species[species] = 0
                frame.detected_species[species] += 1
            # order dictionary by key
            frame.detected_species = dict(sorted(frame.detected_species.items()))

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

    def annotate_species(self):
        """Annotates the species on the right side of the frames."""
        for frame in self.frames:
            for classification in frame.classifications:
                if classification.taken:
                    cv2.rectangle(
                        frame.image,
                        (classification.detection.bbox[0], classification.detection.bbox[1]),
                        (classification.detection.bbox[2], classification.detection.bbox[3]),
                        (0, 0, 255),
                        2
                    )
                    cv2.putText(
                        frame.image,
                        f"Species: {classification.taken_cls}",
                        (classification.detection.bbox[0], classification.detection.bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, 
                        (255, 255, 255),
                        2
                    )

    def annotate_species_counts(self):
        """Annotates the species counts on the right side of the frames with a grayish background box."""
        for frame in self.frames:
            # Draw a grayish background box
            box_width = 300
            box_height = 200
            top_left = (10, 10)
            bottom_right = (top_left[0] + box_width, top_left[1] + box_height)

            # Draw the box by merging part of the image with a grayish rectangle
            # This creates a semi-transparent effect
            sub_img = frame.image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            box_rect = np.ones((box_height, box_width, 3), dtype=np.uint8) * 50
            res = cv2.addWeighted(sub_img, 0.25, box_rect, 0.75, 1.0)
            frame.image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = res

            # Add species counts text
            y_offset = 40
            for i, (species, count) in enumerate(frame.detected_species.items()):
                if i >= 5:  # Limit to 5 species
                    break
                cv2.putText(
                    frame.image,
                    f"{species}: {count}",
                    (top_left[0] + 10, top_left[1] + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,  # Smaller font size
                    (255, 255, 255),  # White text
                    1
                )
                y_offset += 25

    def annotate_frame_ids(self):
        """Annotates the frame IDs on the frames."""
        for frame in self.frames:
            cv2.putText(
                frame.image,
                f"Frame ID: {frame.frame_id}",
                (self.size[0] - 300, 30),
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

    # RUN

    def run(self, output_path: str, number_frames: bool = False):
        """Main method to run the entire pipeline."""
        self.download_video()
        self.extract_frames()

        self.detect_birds(annotate=False)
        self.classify_birds(annotate=False)
        self.process_observations()

        self.annotate_classifications()
        self.annotate_species_counts()

        if number_frames:
            self.annotate_frame_ids()

        self.export_video(output_path)
