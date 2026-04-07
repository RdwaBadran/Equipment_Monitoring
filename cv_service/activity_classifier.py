# cv_service/activity_classifier.py
# Rule-based activity classification using optical flow region scores.
# No ML training needed — pure logic based on motion patterns.
#
# Activities (matching the assessment spec exactly):
#   DIGGING          → arm moving downward into ground (arm_only, top active)
#   SWINGING/LOADING → arm rotating laterally while cab turns (top + middle OR cab_only)
#   DUMPING          → very fast downward motion in top region (releasing load)
#   WAITING          → no significant motion, or unclassified active state

import numpy as np

# ── Activity labels — match assessment spec exactly ──────────────────────────
ACTIVITY_DIGGING  = "DIGGING"
ACTIVITY_SWINGING = "SWINGING/LOADING"   # spec says "Swinging/Loading"
ACTIVITY_DUMPING  = "DUMPING"
ACTIVITY_WAITING  = "WAITING"

ACTIVITIES = [ACTIVITY_DIGGING, ACTIVITY_SWINGING, ACTIVITY_DUMPING, ACTIVITY_WAITING]


class ActivityClassifier:
    def __init__(self):
        # Keep a short history of scores to smooth out noise
        # (avoids flickering between states every frame)
        self.history_len = 8
        self.score_history: dict = {}   # track_id → list of score dicts

    def classify(
        self,
        track_id:      int,
        motion_result: dict,
        flow_vectors=None
    ) -> str:
        """
        Classify the current activity of a tracked machine.

        Input:
            track_id      — unique ID of this machine from the tracker
            motion_result — output dict from MotionAnalyzer.analyze_bbox()
            flow_vectors  — reserved for future use

        Output:
            activity string: one of ACTIVITIES
        """
        # If machine is not active at all → WAITING
        if not motion_result["is_active"]:
            return ACTIVITY_WAITING

        scores = motion_result["region_scores"]
        top    = scores["top"]
        middle = scores["middle"]
        bottom = scores["bottom"]
        source = motion_result["motion_source"]

        # ── Store scores in history for smoothing ────────────────────────
        if track_id not in self.score_history:
            self.score_history[track_id] = []

        self.score_history[track_id].append({
            "top": top, "middle": middle,
            "bottom": bottom, "source": source
        })

        # Keep only last N frames
        if len(self.score_history[track_id]) > self.history_len:
            self.score_history[track_id].pop(0)

        # Use smoothed scores (average over history)
        avg = self._smooth(track_id)

        # ── Classification rules ─────────────────────────────────────────
        #
        # Rule 1: DUMPING
        # Very high top-region motion — arm releasing a heavy load quickly.
        # Threshold 4.5 is deliberately high to avoid false positives.
        if avg["top"] > 4.5:
            return ACTIVITY_DUMPING

        # Rule 2: DIGGING
        # Arm-only motion: tracks are still, only the upper arm region moves.
        # This is the core articulated-motion case from the assessment spec.
        if source == "arm_only" and avg["top"] > 1.5:
            return ACTIVITY_DIGGING

        # Rule 3: SWINGING / LOADING
        # Two sub-cases for the swinging action:
        #   a) Cab-only motion → cab rotating to position arm (start of swing)
        #   b) Top + middle both active → arm elevated + cab rotating (full swing)
        if source == "cab_only" and avg["middle"] > 1.0:
            return ACTIVITY_SWINGING

        if avg["top"] > 1.2 and avg["middle"] > 1.0:
            return ACTIVITY_SWINGING

        # Rule 4: Machine translating (tracks moving more than arm)
        # WAITING is the correct label here — the machine is repositioning,
        # not performing a defined work cycle activity.
        if avg["bottom"] > avg["top"] and avg["bottom"] > 1.2:
            return ACTIVITY_WAITING

        # Default: safe fallback — don't assert a specific work activity
        # when none of the motion patterns matched clearly.
        return ACTIVITY_WAITING

    def _smooth(self, track_id: int) -> dict:
        """
        Average the region scores over the history window.
        Prevents activity label from flickering every frame.
        """
        history = self.score_history[track_id]
        return {
            "top":    float(np.mean([h["top"]    for h in history])),
            "middle": float(np.mean([h["middle"] for h in history])),
            "bottom": float(np.mean([h["bottom"] for h in history])),
        }

    def remove_track(self, track_id: int):
        """Call this when a track disappears to free memory."""
        self.score_history.pop(track_id, None)