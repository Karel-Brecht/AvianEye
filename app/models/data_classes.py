from dataclasses import dataclass
from typing import Optional
from ..operations.detection import Detection


@dataclass
class Observation:  # TODO: rename as Observation
    """Represents a single detection with its bounding box, confidence, and class name."""

    detection: Detection
    predicted_cls: str  # Class name of the detected object
    predicted_prob: float  # Confidence score of the detection
    all_probs: dict[str, float]  # Probabilities for each class
    taken: bool = False  # If detection is taken as a true detection
    taken_cls: str = None  # The actual class that it has been taken for
    tracking_id: Optional[int] = None  # ID to track object across frames
    is_interpolated: bool = (
        False  # If the observation is interpolated (i.e., not directly from a detection)
    )


class Frame:
    """Represents a single frame of the video, including detections and observations."""

    def __init__(self, frame_id: int, image):
        self.frame_id = frame_id
        self.image = image
        self.detections: list[Detection] = []
        self.observations: list[Observation] = (
            []
        )  # List of observations (e.g., bird species) # TODO: rename as observations
        self.detected_species: dict[str, int] = {}  # Dictionary to store species counts

    def add_detection(self, detection: Detection):
        """Adds a detection to the frame."""
        self.detections.append(detection)
        return detection

    def add_observation(
        self,
        detection: Detection,
        predicted_cls: str,
        predicted_prob: float,
        all_probs: dict[str, float],
    ):
        """Adds an observation to the identification."""
        observation = Observation(detection, predicted_cls, predicted_prob, all_probs)
        self.observations.append(observation)
        return observation

    def remove_detections(self):
        """Removes all detections from the frame."""
        self.detections = []

    def remove_observations(self):
        """Removes all observations from the frame."""
        self.observations = []

    def remove_detected_species(self):
        """Removes all detected species from the frame."""
        self.detected_species = {}
        for observation in self.observations:
            if observation.is_interpolated:
                # Remove observation from the frame
                self.observations.remove(observation)
                del observation
            else:
                observation.taken = False
                observation.taken_cls = None
                observation.tracking_id = None

    def get_summary_statistics(self):
        """Returns the summary statistics for the entire video."""
        if not hasattr(self, "summary_statistics"):
            self._generate_summary_statistics()
        return self.summary_statistics