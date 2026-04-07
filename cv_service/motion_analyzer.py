# cv_service/motion_analyzer.py
# Uses Farneback Optical Flow to detect motion inside equipment bounding boxes.
# Splits each bbox into 3 vertical regions to handle articulated motion
# (e.g. excavator arm moving while tracks are stationary).
#
# motion_source values:
#   "arm_only"   → top region active, middle & bottom still  (excavator digging)
#   "cab_only"   → middle region active only                 (cab rotation / swing)
#   "tracks"     → bottom region active, top & middle still  (machine translating)
#   "full_body"  → multiple regions active simultaneously
#   "none"       → no region is moving

import cv2
import numpy as np

# How much motion (in pixels/frame) counts as ACTIVE
MOTION_THRESHOLD = 1.2

# Minimum fraction of pixels that must be moving to trigger ACTIVE
MOVING_PIXEL_RATIO = 0.08   # 8% of region pixels


class MotionAnalyzer:
    def __init__(self):
        self.prev_gray = None   # grayscale version of previous frame

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Call this every frame BEFORE analyze_bbox().
        Converts frame to grayscale and stores it for flow computation.
        Returns the grayscale frame (we reuse it in analyze_bbox).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
        return gray

    def analyze_bbox(
        self,
        current_gray: np.ndarray,
        bbox: list
    ) -> dict:
        """
        Analyze motion inside a bounding box using optical flow.

        Input:
            current_gray — grayscale current frame
            bbox         — [x1, y1, x2, y2]

        Output: dict with keys:
            "is_active"      : bool   — True if any region is moving
            "motion_source"  : str    — "arm_only"/"cab_only"/"tracks"/"full_body"/"none"
            "region_scores"  : dict   — motion score per region
            "global_score"   : float  — average motion across whole bbox
        """
        x1, y1, x2, y2 = bbox

        # Safety: clamp to frame dimensions
        h, w = current_gray.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return self._empty_result()

        # Crop both frames to the bounding box region
        prev_crop    = self.prev_gray[y1:y2, x1:x2]
        current_crop = current_gray[y1:y2, x1:x2]

        # Compute dense optical flow (Farneback algorithm)
        # flow shape: (H, W, 2) — x and y displacement per pixel
        flow = cv2.calcOpticalFlowFarneback(
            prev_crop, current_crop,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Compute motion magnitude per pixel
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # ── Split bbox into 3 vertical regions ─────────────────────────
        bbox_h     = y2 - y1
        top_end    = bbox_h // 3
        middle_end = (2 * bbox_h) // 3

        top_score    = self._region_score(magnitude[0:top_end, :])
        middle_score = self._region_score(magnitude[top_end:middle_end, :])
        bottom_score = self._region_score(magnitude[middle_end:, :])
        global_score = self._region_score(magnitude)

        # ── Determine motion source ─────────────────────────────────────
        top_active    = top_score    > MOTION_THRESHOLD
        middle_active = middle_score > MOTION_THRESHOLD
        bottom_active = bottom_score > MOTION_THRESHOLD

        active_regions = (top_active, middle_active, bottom_active)

        # FIX: each combination of active regions is now handled explicitly.
        # Previously middle_active was checked in any() but never in the
        # specific cases — a cab-only rotation landed in full_body incorrectly.
        if not any(active_regions):
            motion_source = "none"
            is_active     = False

        elif top_active and not middle_active and not bottom_active:
            # Only arm is moving → excavator digging / reaching
            motion_source = "arm_only"
            is_active     = True

        elif middle_active and not top_active and not bottom_active:
            # Only cab is moving → cab rotating during swing phase
            motion_source = "cab_only"
            is_active     = True

        elif bottom_active and not top_active and not middle_active:
            # Only tracks are moving → machine translating
            motion_source = "tracks"
            is_active     = True

        else:
            # Multiple regions active simultaneously → full-body motion
            motion_source = "full_body"
            is_active     = True

        # Update previous frame
        self.prev_gray = current_gray.copy()

        return {
            "is_active":     is_active,
            "motion_source": motion_source,
            "region_scores": {
                "top":    round(top_score, 3),
                "middle": round(middle_score, 3),
                "bottom": round(bottom_score, 3)
            },
            "global_score":  round(global_score, 3)
        }

    def _region_score(self, region_magnitude: np.ndarray) -> float:
        """
        Score = mean magnitude of pixels that are actually moving.
        Ignores still pixels to avoid noise bringing average down.
        """
        moving_pixels = region_magnitude[region_magnitude > 0.5]
        if len(moving_pixels) == 0:
            return 0.0
        ratio = len(moving_pixels) / max(region_magnitude.size, 1)
        if ratio < MOVING_PIXEL_RATIO:
            return 0.0
        return float(np.mean(moving_pixels))

    def _empty_result(self) -> dict:
        return {
            "is_active":     False,
            "motion_source": "none",
            "region_scores": {"top": 0.0, "middle": 0.0, "bottom": 0.0},
            "global_score":  0.0
        }