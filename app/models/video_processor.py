from collections import defaultdict
import json
import cv2
from PIL import Image

from ..operations.download import Downloader
from ..operations.detection import BirdDetector
from ..operations.classification import BirdClassifier
from ..operations.tracking import Tracker
from ..visualization.annotation import Annotator
from ..models.data_classes import Frame


class VideoProcessor:
    """Handles the full pipeline: downloading, extracting frames, detecting birds, and classifying species."""

    def __init__(
        self,
        video_link: str,
        downloader: Downloader,
        detector: BirdDetector,
        classifier: BirdClassifier,
    ):
        self.size = (1280, 720)

        self.video_link = video_link
        self.downloader = downloader
        self.detector = detector
        self.classifier = classifier

        self.annotator = Annotator()

        self.video_path = None
        self.frames: list[Frame] = []
        self.frame_rate = None

        # Parameters for tracking and temporal smoothing
        self.iou_threshold = (
            0.5  # IoU threshold for tracking # TODO: fine-tune this a bit
        )
        self.temporal_window_size = 30  # Number of frames to consider for temporal smoothing # TODO: fine-tune this a bit
        self.min_track_seconds = (
            0.30  # Minimum length of a track in seconds # TODO: fine-tune this a bit
        )
        self.max_gap_to_bridge = 5  # Maximum gap size to bridge in frames # TODO: fine-tune this a bit # TODO set in seconds
        self.min_detection_confidence = 0.10  # Minimum detection confidence # TODO: is already set in detector model
        self.min_classification_confidence = (
            0.0  # Minimum classification confidence # TODO: fine-tune this a bit
        )
        self.tracker = None # Gets set in extract_frames()

        self.summary_statistics = None

    def remove_frames(self):
        """Removes all frames from the video."""
        self.frames = []

    def renew_frames(self):
        """Renews the frames from the video."""
        old_frames = self.frames
        self.frames = []
        self.extract_frames()
        for i, frame in enumerate(self.frames):
            if i < len(old_frames):
                frame.detections = old_frames[i].detections
                frame.observations = old_frames[i].observations
                frame.detected_species = old_frames[i].detected_species

    def remove_detections(self):
        """Removes all detections from the video."""
        for frame in self.frames:
            frame.remove_detections()

    def remove_observations(self):
        """Removes all observations from the video."""
        for frame in self.frames:
            frame.remove_observations()

    def remove_detected_species(self):
        """Removes all detected species from the video."""
        for frame in self.frames:
            frame.remove_detected_species()
            self.tracker = Tracker(
                self.iou_threshold, self.temporal_window_size,
                int(self.frame_rate * self.min_track_seconds),
                self.max_gap_to_bridge,
                self.min_detection_confidence,
                self.min_classification_confidence,
            )

    # RUN

    def run(self, output_path: str, frame_ids: bool = False):
        """Main method to run the entire pipeline."""
        self.download_video()
        self.extract_frames()

        self.detect_birds(annotate=False)
        self.classify_birds(annotate=False)
        self.process_observations()

        # Export summary statistics to JSON file
        with open("summary_statistics.json", "w") as f:
            json.dump(self.summary_statistics, f, indent=4)

        self.annotator.annotate_species(self.frames)
        self.annotator.annotate_species_counts(self.frames)

        if frame_ids:
            self.annotator.annotate_frame_ids(self.frames)

        self.export_video(output_path) # TODO: return success or failure

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
        self.tracker = Tracker(
            self.iou_threshold, self.temporal_window_size,
            int(self.frame_rate * self.min_track_seconds),
            self.max_gap_to_bridge,
            self.min_detection_confidence,
            self.min_classification_confidence,
        )
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
            detections = self.detector.detect_birds(
                frame.image
            )  # Returns [(bbox, confidence), ...]
            for detection in detections:
                frame.add_detection(detection)

        if annotate:
            self.annotator.annotate_detections(self.frames)

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

                predicted_cls, predicted_prob, all_probs = (
                    self.classifier.classify(bird_image_pil)
                )
                frame.add_observation(
                    detection, predicted_cls, predicted_prob, all_probs
                )

        if annotate:
            self.annotator.annotate_classifications(self.frames)

    def process_observations(self):
        """Processes the observations to count species with temporal consistency."""
        # Step 1: Assign tracking IDs to detections across frames
        self.tracker.assign_tracking_ids(self.frames)

        # Step 2: Bridge gaps in tracks
        self.tracker.bridge_track_gaps(self.frames)

        # Step 3: Smooth observations based on temporal consistency and filter short tracks
        self.tracker.smooth_observations()

        # Step 4: Count unique individuals and species per frame
        for frame in self.frames:
            # Reset detected species count
            frame.detected_species = {}

            # Process only observations with tracking IDs and sufficient confidence
            for observation in frame.observations:
                # Skip if no tracking ID or already processed this track in this frame
                if observation.tracking_id is None:
                    continue

                # Update species count
                species = observation.predicted_cls
                if species not in frame.detected_species:
                    frame.detected_species[species] = 0
                frame.detected_species[species] += 1

                # Mark as taken
                observation.taken = True
                observation.taken_cls = species

            # Sort dictionary by key
            frame.detected_species = dict(sorted(frame.detected_species.items()))

        # Optional: Aggregate statistics across all frames
        self._generate_summary_statistics()

    def _generate_summary_statistics(self):
        """Generate summary statistics for the entire video."""
        species_counts = defaultdict(int)
        species_confidence = defaultdict(list)
        unique_tracks = set()

        for track_id, observations in self.tracker.tracks.items():
            # Consider only observations with sufficient confidence
            valid_observations = [c for c in observations if c.taken]

            if len(valid_observations) == 0:
                continue

            # Count votes for most frequent species in this track
            species_votes = defaultdict(int)
            for c in valid_observations:
                # Give less weight to interpolated observations
                weight = 1 if not c.is_interpolated else 0.7  # TODO: parameterize this
                species_votes[c.predicted_cls] += weight

            # Get most frequently predicted species for this track
            if species_votes:
                most_common_species = max(species_votes.items(), key=lambda x: x[1])[0]
                species_counts[most_common_species] += 1
                unique_tracks.add(track_id)

                # Collect confidence values
                for c in valid_observations:
                    if c.predicted_cls == most_common_species:
                        species_confidence[most_common_species].append(c.predicted_prob)

        # Store summary statistics
        self.summary_statistics = {
            "total_unique_birds": len(unique_tracks),
            "total_unique_species": len(species_counts),
            "species_counts": dict(sorted(species_counts.items())),
            "average_confidence": {
                species: sum(confs) / len(confs) if confs else 0
                for species, confs in species_confidence.items()
            },
        }

    
    def process_observations_simple(self): # TODO: merge with process_observations or leave out?
        """Processes the observations to count species."""
        for frame in self.frames:
            for observation in frame.observations:
                species = observation.predicted_cls
                if species not in frame.detected_species:
                    frame.detected_species[species] = 0
                frame.detected_species[species] += 1
            # order dictionary by key
            frame.detected_species = dict(sorted(frame.detected_species.items()))

    # EXPORTS

    def export_frames(self, output_dir: str):
        """Exports the frames with detections and observations to the specified directory."""
        for frame in self.frames:
            output_path = f"{output_dir}/frame_{frame.frame_id}.jpg"
            cv2.imwrite(output_path, frame.image)
            print(f"Saved frame {frame.frame_id} to {output_path}")

    def export_frame(self, frame_id: int, output_dir: str):
        """Exports a single frame with detections and observations."""
        if frame_id < len(self.frames):
            frame = self.frames[frame_id]
            output_path = f"{output_dir}/frame_{frame.frame_id}.jpg"
            cv2.imwrite(output_path, frame.image)
            print(f"Saved frame {frame_id} to {output_path}")
        else:
            print(f"Frame {frame_id} does not exist.")

    def export_video(self, output_path: str):
        """Exports the processed video with detections and observations."""
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            output_path, fourcc, self.frame_rate, self.size
        )  # TODO: what if output path doesn't end on .mp4?

        for frame in self.frames:
            out.write(frame.image)

        out.release()
        print(f"Saved processed video to {output_path}")
