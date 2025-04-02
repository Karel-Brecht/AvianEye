from collections import defaultdict

from ..utils.utils import calculate_iou
from ..models.data_classes import Observation, Frame
from .detection import Detection

class Tracker:
    def __init__(self, iou_threshold, temporal_window_size, 
                 min_track_frames, max_gap_to_bridge,
                 min_detection_confidence,
                 min_classification_confidence):
        self.iou_threshold = iou_threshold
        self.temporal_window_size = temporal_window_size
        self.min_track_frames = min_track_frames
        self.max_gap_to_bridge = max_gap_to_bridge
        self.min_detection_confidence = min_detection_confidence
        self.min_classification_confidence = min_classification_confidence
        
        # Track information
        self.next_track_id = 0
        self.tracks: dict[int, list[Observation]] = {}  # track_id -> list of observations
        self.track_frame_indices = {}  # track_id -> list of frame indices


    def assign_tracking_ids(self, frames: list[Frame]):
        """
        Assign tracking IDs to detections across frames based on IoU, ensuring only one observation per track ID per frame.
        
        A track is a sequence of observations that are linked together based on their spatial and temporal proximity.
        """
        if not frames:
            return

        # Process first frame
        for observation in frames[0].observations:
            if observation.detection.confidence >= self.min_detection_confidence:
                observation.tracking_id = self.next_track_id
                self.tracks[self.next_track_id] = [observation]
                self.track_frame_indices[self.next_track_id] = [0]  # Store frame index
                self.next_track_id += 1

        # Process subsequent frames
        for i in range(1, len(frames)):
            current_frame = frames[i]
            prev_frame = frames[i - 1]

            # Step 1: Calculate all possible matches between current and previous frames
            possible_matches = []

            for curr_idx, current_observation in enumerate(
                current_frame.observations
            ):
                if (
                    current_observation.detection.confidence
                    < self.min_detection_confidence
                ):
                    continue

                for prev_observation in prev_frame.observations:
                    if prev_observation.tracking_id is None:
                        continue

                    iou = calculate_iou(
                        current_observation.detection.bbox,
                        prev_observation.detection.bbox,
                    )

                    if iou > self.iou_threshold:
                        possible_matches.append(
                            (curr_idx, prev_observation.tracking_id, iou)
                        )

            # Step 2: Sort matches by IoU (highest first)
            possible_matches.sort(key=lambda x: x[2], reverse=True)

            # Step 3: Assign track IDs, ensuring one observation per track ID
            assigned_curr_idxs = set()
            assigned_track_ids = set()

            for curr_idx, track_id, iou in possible_matches:
                # Skip if this current observation or track ID has already been assigned
                if curr_idx in assigned_curr_idxs or track_id in assigned_track_ids:
                    continue

                # Assign track ID to current observation
                current_observation = current_frame.observations[curr_idx]
                current_observation.tracking_id = track_id

                # Update tracking information
                self.tracks[track_id].append(current_observation)
                self.track_frame_indices[track_id].append(i)  # Store frame index

                # Mark as assigned
                assigned_curr_idxs.add(curr_idx)
                assigned_track_ids.add(track_id)

            # Step 4: Create new tracks for unassigned detections that are not duplicate
            observations_to_assign = {}
            for curr_idx, current_observation in enumerate(
                current_frame.observations
            ):
                if (
                    curr_idx not in assigned_curr_idxs
                    and current_observation.detection.confidence
                    >= self.min_detection_confidence
                ):

                    # print(f"frame: {i}, curridx: {curr_idx}")

                    abort = False
                    # Calculate all ious with assigned
                    for ref_idx, ref_observation in enumerate(
                        current_frame.observations
                    ):
                        if ref_idx not in assigned_curr_idxs or ref_idx == curr_idx:
                            continue
                        # print(f"frame: {i}, ref_idx: {ref_idx}, curr_idx: {curr_idx}")

                        iou = calculate_iou(
                            current_observation.detection.bbox,
                            ref_observation.detection.bbox,
                        )
                        if iou > 0.5:  # TODO: parametrize
                            abort = True
                            break
                    if abort:
                        continue

                    # Calculate all ious with to be assigned observations
                    for (
                        ref_idx,
                        ref_observation,
                    ) in observations_to_assign.items():
                        if (
                            ref_idx == curr_idx
                        ):  # TODO: I think this normally can't happen
                            continue
                        iou = calculate_iou(
                            current_observation.detection.bbox,
                            ref_observation.detection.bbox,
                        )
                        if iou > 0.5:  # TODO: parametrize
                            # calculate similarity between classe probabilities
                            class_similarity = 0.0
                            for (
                                class_name,
                                prob,
                            ) in current_observation.all_probs.items():
                                if class_name in ref_observation.all_probs:
                                    class_similarity += min(
                                        prob, ref_observation.all_probs[class_name]
                                    )

                            # Combined score for IoU and class similarity
                            duplicate_score = (
                                iou * 0.7 + class_similarity * 0.3
                            )  # TODO: parametrize all weights
                            if (
                                duplicate_score > 0.4
                            ):  # TODO: Different threshold for considering it a duplicate
                                if (
                                    current_observation.predicted_prob
                                    > ref_observation.predicted_prob
                                ):
                                    observations_to_assign[curr_idx] = (
                                        current_observation
                                    )
                                    # remove ref_observation from the list of observations to be assigned
                                    del observations_to_assign[ref_idx]
                                abort = True
                                break
                    if abort:
                        continue
                    # Add to observations to assign
                    observations_to_assign[curr_idx] = current_observation

            for curr_idx, current_observation in observations_to_assign.items():
                if curr_idx in assigned_curr_idxs:  # TODO: I think this is redundant
                    continue
                # Create new track
                current_observation.tracking_id = self.next_track_id
                self.tracks[self.next_track_id] = [current_observation]
                self.track_frame_indices[self.next_track_id] = [i]  # Store frame index
                self.next_track_id += 1

    # TODO: maybe remove extra duplicate detections if bbox corners line up too well


    def bridge_track_gaps(self, frames: list[Frame]):
        """Bridge short gaps in tracks by interpolating detections and class probabilities."""
        # Create dictionary to map frame index to frame object for faster lookup
        frame_dict = {i: frame for i, frame in enumerate(frames)}

        # Track IDs that need updating after gap bridging
        # Ordered dictionary! so that we can merge in progressive order
        tracks_to_merge = {}  # old_track_id -> new_track_id 

        # Look for potential track continuations (tracks that end then restart)
        for track_id, frame_indices in self.track_frame_indices.items():
            # Skip if track is empty
            if not frame_indices:
                continue

            track_end_frame = frame_indices[-1]
            track_end_observation = self.tracks[track_id][-1]

            # Look for potential continuations within max_gap_to_bridge frames
            potential_continuations = []

            # Search for tracks that start after this one ends
            for other_track_id, other_frame_indices in self.track_frame_indices.items():
                if (
                    not other_frame_indices
                    or other_track_id == track_id
                    or other_track_id in tracks_to_merge
                    or other_track_id in tracks_to_merge.values()
                ):
                    continue

                other_start_frame = other_frame_indices[0]

                # Check if the other track starts within the bridging window
                gap_size = other_start_frame - track_end_frame
                if 1 <= gap_size <= self.max_gap_to_bridge:
                    other_start_observation = self.tracks[other_track_id][0]

                    # Calculate IoU between end of track and start of potential continuation
                    iou = calculate_iou(
                        track_end_observation.detection.bbox,
                        other_start_observation.detection.bbox,
                    )

                    # Check class similarity
                    class_similarity = 0.0
                    for class_name, prob in track_end_observation.all_probs.items():
                        if class_name in other_start_observation.all_probs:
                            class_similarity += min(
                                prob, other_start_observation.all_probs[class_name]
                            )

                    # Combined score for IoU and class similarity # TODO: add temporal proximity to score
                    continuity_score = (
                        iou * 0.7 + class_similarity * 0.3
                    )  # TODO: parametrize all weights

                    if (
                        continuity_score > 0.3
                    ):  # Threshold for considering it a continuation
                        potential_continuations.append(
                            (other_track_id, gap_size, continuity_score)
                        )

            # If we found potential continuations, bridge the gap to the best one
            if potential_continuations:

                # Take best contiuation for smalles gap size elements
                # first sort by continuity score (ascending)
                potential_continuations.sort(key=lambda x: x[2])
                # then iterate over the list and take the last one that has the same gap size
                best_gap_size = self.max_gap_to_bridge
                for continuation_track_id, gap_size, continuity_score in potential_continuations:
                    if gap_size <= best_gap_size:
                        best_continuation_id = continuation_track_id
                        best_gap_size = gap_size

                # Create interpolated observations for the gap
                self._interpolate_gap(
                    track_id,
                    best_continuation_id,
                    track_end_frame,
                    self.track_frame_indices[best_continuation_id][0],
                    frame_dict,
                )

                # Mark continuation track for merging
                tracks_to_merge[best_continuation_id] = track_id

                # TODO: wouldn't it be better to merge these two tracks immediately here?

                # TODO: Or maybe better, allow multiple merges with the same new_track_id (tracks_to_merge.values())
                #       but in a second pass allow only the ones with the shortest gap_size

        # Merge tracks that were connected by bridging
        self._merge_tracks(tracks_to_merge)


    def _interpolate_gap(
        self, track_id1, track_id2, end_frame_idx, start_frame_idx, frame_dict
    ):
        """Create interpolated observations for the gap between two tracks."""
        # Get the last observation of track1 and first observation of track2
        last_observation = self.tracks[track_id1][-1]
        first_observation = self.tracks[track_id2][0]

        # Get bounding boxes
        start_bbox = last_observation.detection.bbox
        end_bbox = first_observation.detection.bbox

        # Get class probabilities
        start_probs = last_observation.all_probs
        end_probs = first_observation.all_probs

        # Number of frames to interpolate
        gap_size = start_frame_idx - end_frame_idx - 1

        # Create interpolated observations for each frame in the gap
        for i in range(1, gap_size + 1):
            # Calculate interpolation factor (0.0 to 1.0)
            t = i / (gap_size + 1)

            # Interpolate bounding box
            interpolated_bbox = (
                int(start_bbox[0] + t * (end_bbox[0] - start_bbox[0])),
                int(start_bbox[1] + t * (end_bbox[1] - start_bbox[1])),
                int(start_bbox[2] + t * (end_bbox[2] - start_bbox[2])),
                int(start_bbox[3] + t * (end_bbox[3] - start_bbox[3])),
            )

            # Interpolate confidence
            interpolated_confidence = (
                (1 - t) * last_observation.detection.confidence
                + t * first_observation.detection.confidence
            )

            # Interpolate class probabilities
            interpolated_probs = {}
            all_classes = set(start_probs.keys()) | set(end_probs.keys())

            for class_name in all_classes:
                start_prob = start_probs.get(class_name, 0.0)
                end_prob = end_probs.get(class_name, 0.0)
                interpolated_probs[class_name] = (1 - t) * start_prob + t * end_prob

            # Determine predicted class (highest probability)
            predicted_cls = max(interpolated_probs.items(), key=lambda x: x[1])[0]
            predicted_prob = interpolated_probs[predicted_cls]

            # Create interpolated detection
            interpolated_detection = Detection(
                bbox=interpolated_bbox,
                confidence=interpolated_confidence,
                class_name=predicted_cls,
            )

            # Create interpolated observation
            interpolated_observation = Observation(
                detection=interpolated_detection,
                predicted_cls=predicted_cls,
                predicted_prob=predicted_prob,
                all_probs=interpolated_probs,
                tracking_id=track_id1,
                is_interpolated=True,
            )

            # Get the frame for this interpolation
            frame_idx = end_frame_idx + i
            frame: Frame = frame_dict[frame_idx]

            # Add interpolated observation to frame
            frame.observations.append(interpolated_observation)

            # Add to track
            insert_idx = len(self.tracks[track_id1])
            self.tracks[track_id1].insert(insert_idx, interpolated_observation)
            self.track_frame_indices[track_id1].insert(insert_idx, frame_idx)


    def _merge_tracks(self, tracks_to_merge):
        """Merge tracks after gap bridging."""
        for old_track_id, new_track_id in tracks_to_merge.items():
            # Skip if either track no longer exists
            if old_track_id not in self.tracks or new_track_id not in self.tracks:
                continue

            # Update tracking IDs in all observations from old track
            for observation in self.tracks[old_track_id]:
                observation.tracking_id = new_track_id

            # Append old track's observations to new track
            self.tracks[new_track_id].extend(self.tracks[old_track_id])
            self.track_frame_indices[new_track_id].extend(
                self.track_frame_indices[old_track_id]
            )

            # Sort by frame index
            combined = list(
                zip(self.track_frame_indices[new_track_id], self.tracks[new_track_id])
            )
            combined.sort(key=lambda x: x[0])

            # Unzip sorted lists
            self.track_frame_indices[new_track_id], self.tracks[new_track_id] = zip(
                *combined
            )
            self.track_frame_indices[new_track_id] = list(
                self.track_frame_indices[new_track_id]
            )
            self.tracks[new_track_id] = list(self.tracks[new_track_id])

            # Remove old track
            del self.tracks[old_track_id]
            del self.track_frame_indices[old_track_id]


    def smooth_observations(self):
        """Apply temporal smoothing to observations based on tracking data and filter out short tracks."""
        tracks_to_remove = []

        for track_id, observations in self.tracks.items():
            # If tracking is too short, remove it
            if len(observations) < self.min_track_frames:
                # print(f"Track {track_id} is too short, removing it.")
                for observation in observations:
                    observation.tracking_id = None
                tracks_to_remove.append(track_id)
                continue

            # If track has only one detection, no smoothing needed
            if len(observations) <= 1:
                continue

            # For each observation in the track
            for i, observation in enumerate(observations):
                # Skip smoothing for interpolated observations
                if observation.is_interpolated:
                    continue

                # Determine window size (adjust for beginning and end of track)
                window_start = max(0, i - self.temporal_window_size // 2)
                window_end = min(
                    len(observations), i + self.temporal_window_size // 2 + 1
                )
                window = observations[window_start:window_end]

                # Count class votes weighted by probability
                class_votes = defaultdict(float)
                for cls in window:
                    weight = 1.0
                    # Reduce weight for interpolated observations
                    if cls.is_interpolated:
                        weight = 0.7  # TODO: parameterize this

                    for class_name, prob in cls.all_probs.items():
                        class_votes[class_name] += prob * weight

                # Get class with highest votes
                best_class = max(class_votes.items(), key=lambda x: x[1])

                observation.predicted_cls = best_class[0]
                observation.predicted_prob = best_class[1] / len(window)

        # Remove the short tracks from the tracks dictionary
        for track_id in tracks_to_remove:
            if track_id in self.tracks:
                del self.tracks[track_id]
            if track_id in self.track_frame_indices:
                del self.track_frame_indices[track_id]

        # TODO: remove tracks that are too ambiguous

        # TODO: second stage for removing duplicate tracks
