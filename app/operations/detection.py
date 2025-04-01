from dataclasses import dataclass
from ultralytics import YOLO

@dataclass
class Detection:
    """Represents a single detection with its bounding box, confidence, and class name."""
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str

class BirdDetector:
    def __init__(self, model_path='models/detection_model/yolov10n.pt', confidence_threshold=0.10):
        """Initialize the BirdDetector with a YOLO model."""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect_birds(self, frame) -> list[Detection]:
        """Detect birds in a given frame using YOLO."""
        detect_params = self.model.predict(source=[frame], conf=self.confidence_threshold, save=False)
        detections = []

        if len(detect_params[0]) != 0:
            boxes = detect_params[0].boxes
            for i in range(len(detect_params[0])):
                box = boxes[i]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]
                clsID = box.cls.numpy()[0]
                # Name of object detected (e.g. 'bird')
                class_name = self.model.names[int(clsID)]

                if 'bird' in class_name.lower():
                    detections.append(Detection(
                        bbox=(int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])),
                        confidence= conf,
                        class_name= class_name
                    ))

        return detections
