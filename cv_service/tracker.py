# cv_service/tracker.py
# Wraps YOLOv8 BoT-SORT tracking and adds appearance-based Re-ID.
# Equipment filtering is applied HERE because model.track() runs its own
# detection internally — detector.py cannot intercept those results.

import cv2
import numpy as np
import time
from ultralytics import YOLO

LOST_TRACK_BUFFER_SECS    = 8.0
REID_SIMILARITY_THRESHOLD = 0.60

# Classes to reject — people, regular vehicles, animals
REJECTED_CLASS_IDS = {
    0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
}

# Classes to accept
ACCEPTED_CLASS_IDS = {7, 8}   # truck, boat (excavators match as boat in COCO)

# Keywords in class name to accept
CONSTRUCTION_KEYWORDS = [
    "truck", "excavator", "bulldozer", "crane","machinery"
]


class Track:
    """Represents one tracked machine with its visual fingerprint."""

    def __init__(self, track_id: int, bbox: list, class_name: str, frame: np.ndarray):
        self.track_id   = track_id
        self.bbox       = bbox
        self.class_name = class_name
        self.histogram  = self._compute_histogram(frame, bbox)
        self.last_seen  = time.time()
        self.is_active  = True

    def update(self, bbox: list, frame: np.ndarray):
        self.bbox      = bbox
        self.last_seen = time.time()
        new_hist       = self._compute_histogram(frame, bbox)
        # EMA update: 80% old fingerprint, 20% new — stable but adapts slowly
        self.histogram = 0.8 * self.histogram + 0.2 * new_hist

    def _compute_histogram(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 5 or y2 - y1 < 5:
            return np.zeros(96)

        crop = frame[y1:y2, x1:x2]
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])

        hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        norm = np.linalg.norm(hist)
        return hist / (norm + 1e-6)


class Tracker:
    def __init__(self, model: YOLO):
        self.model          = model
        self.active_tracks: dict[int, Track] = {}
        self.lost_tracks:   dict[int, Track] = {}
        self.id_remapping:  dict[int, int]   = {}

    def update(self, frame: np.ndarray) -> list[dict]:
        """
        Run BoT-SORT tracking on the frame and return filtered equipment detections.

        FIX: removed the unused `detections` parameter. Previously, main.py called
        detector.detect() and passed the results here, but model.track() runs its
        own internal detection — the external detections were never used, wasting
        one full YOLO inference per frame.

        Returns list of dicts with keys:
            track_id, bbox, class_name, confidence, is_reid
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            conf=0.35,
            verbose=False
        )

        tracked     = []
        current_ids = set()

        for result in results:
            if result.boxes.id is None:
                continue

            for box in result.boxes:
                raw_id     = int(box.id[0])
                class_id   = int(box.cls[0])
                class_name = self.model.names[class_id]
                conf       = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                bbox       = [x1, y1, x2, y2]

                # ── Filter non-equipment ─────────────────────────────
                if class_id in REJECTED_CLASS_IDS:
                    continue

                is_equipment = (
                    class_id in ACCEPTED_CLASS_IDS or
                    any(kw in class_name.lower() for kw in CONSTRUCTION_KEYWORDS)
                )
                if not is_equipment:
                    continue
                # ── Remap COCO 'boat' → 'excavator' in construction context ─────────
                if class_name == "boat":
                    class_name = "excavator"

                # ── Re-ID ────────────────────────────────────────────
                final_id, is_reid = self._apply_reid(raw_id, bbox, class_name, frame)
                current_ids.add(final_id)

                if final_id in self.active_tracks:
                    self.active_tracks[final_id].update(bbox, frame)
                else:
                    self.active_tracks[final_id] = Track(
                        final_id, bbox, class_name, frame
                    )

                tracked.append({
                    "track_id":   final_id,
                    "bbox":       bbox,
                    "class_name": class_name,
                    "confidence": round(conf, 3),
                    "is_reid":    is_reid
                })

        self._update_lost_buffer(current_ids)
        return tracked

    def _apply_reid(
        self,
        raw_id:     int,
        bbox:       list,
        class_name: str,
        frame:      np.ndarray
    ) -> tuple[int, bool]:
        # Step 1: Already active → no Re-ID needed
        if raw_id in self.active_tracks:
            return raw_id, False

        # Step 2: Already remapped from a previous frame
        if raw_id in self.id_remapping:
            return self.id_remapping[raw_id], True

        # Step 3: Brand new detection → compare to lost tracks only
        if not self.lost_tracks:
            return raw_id, False

        new_hist = Track(raw_id, bbox, class_name, frame).histogram
        best_id  = None
        best_sim = 0.0

        for lost_id, lost_track in self.lost_tracks.items():
            if lost_track.class_name != class_name:
                continue

            sim = float(np.dot(new_hist, lost_track.histogram))
            if sim > best_sim:
                best_sim = sim
                best_id  = lost_id

        if best_sim >= REID_SIMILARITY_THRESHOLD and best_id is not None:
            self.id_remapping[raw_id] = best_id
            del self.lost_tracks[best_id]
            print(
                f"[Re-ID] Track {raw_id} → restored as {best_id} "
                f"(similarity={best_sim:.2f})"
            )
            return best_id, True

        return raw_id, False

    def _update_lost_buffer(self, current_ids: set):
        now = time.time()

        for tid, track in list(self.active_tracks.items()):
            if tid not in current_ids:
                track.is_active = False
                self.lost_tracks[tid] = track
                del self.active_tracks[tid]

        for tid in list(self.lost_tracks.keys()):
            if now - self.lost_tracks[tid].last_seen > LOST_TRACK_BUFFER_SECS:
                del self.lost_tracks[tid]
                self.id_remapping = {
                    k: v for k, v in self.id_remapping.items()
                    if v != tid
                }