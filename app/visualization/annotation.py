import cv2
import numpy as np

from ..models.data_classes import Frame


class Annotator:
    def __init__(self):
        pass

    def annotate_detections(self, frames: list[Frame]):
        """Runs object detection on each frame to detect birds."""
        for frame in frames:
            for detection in frame.detections:
                cv2.rectangle(
                    frame.image,
                    (detection.bbox[0], detection.bbox[1]),
                    (detection.bbox[2], detection.bbox[3]),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame.image,
                    f"{detection.class_name} {detection.confidence:.2f}",
                    (detection.bbox[0], detection.bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

    def annotate_classifications(self, frames: list[Frame]):
        """Annotates the classifications on the frames."""
        for frame in frames:
            for observation in frame.observations:
                cv2.rectangle(
                    frame.image,
                    (
                        observation.detection.bbox[0],
                        observation.detection.bbox[1],
                    ),
                    (
                        observation.detection.bbox[2],
                        observation.detection.bbox[3],
                    ),
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    frame.image,
                    f"{observation.predicted_cls} {observation.predicted_prob:.2f}",
                    (
                        observation.detection.bbox[0],
                        observation.detection.bbox[1] - 10,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

    def annotate_species(self, frames: list[Frame]):
        """Annotates the species on the right side of the frames."""
        for frame in frames:
            for observation in frame.observations:
                if observation.taken:
                    cv2.rectangle(
                        frame.image,
                        (
                            observation.detection.bbox[0],
                            observation.detection.bbox[1],
                        ),
                        (
                            observation.detection.bbox[2],
                            observation.detection.bbox[3],
                        ),
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        frame.image,
                        f"Species: {observation.taken_cls}",
                        (
                            observation.detection.bbox[0],
                            observation.detection.bbox[1] - 10,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

    def annotate_species_counts(self, frames: list[Frame]): # TODO: make this prettier
        """Annotates the species counts on the right side of the frames with a grayish background box."""
        for frame in frames:
            # Draw a grayish background box
            box_width = 300
            box_height = 200
            top_left = (10, 10)
            bottom_right = (top_left[0] + box_width, top_left[1] + box_height)

            # Draw the box by merging part of the image with a grayish rectangle
            # This creates a semi-transparent effect
            sub_img = frame.image[
                top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]
            ]
            box_rect = np.ones((box_height, box_width, 3), dtype=np.uint8) * 50
            res = cv2.addWeighted(sub_img, 0.25, box_rect, 0.75, 1.0)
            frame.image[
                top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]
            ] = res

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
                    1,
                )
                y_offset += 25

    def annotate_frame_ids(self, frames: list[Frame]):
        """Annotates the frame IDs on the frames."""
        for frame in frames:
            # get frame size
            frame_width = frame.image.shape[1]
            cv2.putText(
                frame.image,
                f"Frame ID: {frame.frame_id}",
                (frame_width - 300, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )