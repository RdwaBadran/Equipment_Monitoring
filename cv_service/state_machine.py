# cv_service/state_machine.py
# Tracks ACTIVE/INACTIVE state per machine and calculates:
#   - Each individual idle session (توقف) — duration + video timestamp
#   - Total accumulated idle time (sum of all sessions)
#   - Total active time
#   - Utilization percentage
#
# Time is tracked using frame-based dt (process_every / fps) instead of
# wall clock — this gives accurate VIDEO time regardless of processing speed.

from dataclasses import dataclass, field
from typing import List


# ── COCO class → readable prefix ──────────────────────────────────────────────
CLASS_PREFIX: dict = {
    "truck":     "TR",
    "bulldozer": "BD",
    "crane":     "CR",
    "excavator": "EX",
    "machinery": "MX",
}


@dataclass
class IdleSession:
    """One complete idle stop for a machine."""
    session_number:  int    # 1, 2, 3 ... (chronological)
    started_at_secs: float  # video time when idle started (seconds from start)
    duration_secs:   float  # how long this stop lasted (seconds)

    @property
    def started_at_ts(self) -> str:
        """Human-readable video timestamp: HH:MM:SS"""
        return _format_ts(self.started_at_secs)

    @property
    def duration_ts(self) -> str:
        """Human-readable duration: MM:SS"""
        m = int(self.duration_secs // 60)
        s = int(self.duration_secs  % 60)
        return f"{m:02d}:{s:02d}"

    def to_dict(self) -> dict:
        return {
            "session_number":  self.session_number,
            "started_at_secs": round(self.started_at_secs, 1),
            "started_at_ts":   self.started_at_ts,
            "duration_secs":   round(self.duration_secs, 1),
            "duration_ts":     self.duration_ts,
        }


@dataclass
class EquipmentState:
    """Complete time-tracking state for one machine."""

    track_id:          int
    equipment_class:   str

    # Current live state
    current_state:     str   = "INACTIVE"
    current_activity:  str   = "WAITING"
    motion_source:     str   = "none"

    # Cumulative time (VIDEO seconds)
    total_tracked:     float = 0.0
    total_active:      float = 0.0
    total_idle:        float = 0.0

    # Current idle session tracking
    current_idle_secs:    float = 0.0   # how long THIS idle session has lasted
    idle_session_start:   float = 0.0   # video time when THIS idle session began

    # History of all completed idle sessions
    idle_sessions: List[IdleSession] = field(default_factory=list)

    # Last completed idle session (for Kafka payload — only set on transition)
    last_completed_idle: dict = field(default_factory=dict)

    @property
    def utilization_percent(self) -> float:
        if self.total_tracked == 0:
            return 0.0
        return round((self.total_active / self.total_tracked) * 100, 1)

    @property
    def equipment_id(self) -> str:
        prefix = CLASS_PREFIX.get(
            self.equipment_class.lower(),
            self.equipment_class.upper()[:2]
        )
        return f"{prefix}-{self.track_id:03d}"

    @property
    def idle_sessions_count(self) -> int:
        return len(self.idle_sessions)

    @property
    def longest_idle_secs(self) -> float:
        if not self.idle_sessions:
            return 0.0
        return max(s.duration_secs for s in self.idle_sessions)

    @property
    def avg_idle_secs(self) -> float:
        if not self.idle_sessions:
            return 0.0
        total_completed = sum(s.duration_secs for s in self.idle_sessions)
        return round(total_completed / len(self.idle_sessions), 1)


def _format_ts(total_seconds: float) -> str:
    """Convert seconds → HH:MM:SS.mmm  (matches assessment spec format)."""
    hours   = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs    = int(total_seconds % 60)
    millis  = int(round((total_seconds % 1) * 1000))
    if millis >= 1000:
        millis = 0
        secs  += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class StateMachine:
    def __init__(self):
        self.states: dict = {}   # track_id → EquipmentState

    def update(
        self,
        track_id:        int,
        equipment_class: str,
        is_active:       bool,
        activity:        str,
        motion_source:   str   = "none",
        fps:             float = 25.0,
        process_every:   int   = 1,
    ) -> EquipmentState:
        """
        Update state for one machine. Returns the updated EquipmentState.

        Key behaviour:
          - When machine goes INACTIVE  → starts timing the idle session
          - When machine goes ACTIVE    → closes the idle session, records duration,
                                          appends to idle_sessions list
          - last_completed_idle         → set only on that transition frame so
                                          db_consumer can INSERT the idle record once
        """
        dt = process_every / fps   # video seconds per processed frame

        # ── First time we see this machine ────────────────────────────────────
        if track_id not in self.states:
            st = EquipmentState(
                track_id        = track_id,
                equipment_class = equipment_class,
                current_state   = "ACTIVE" if is_active else "INACTIVE",
                current_activity= activity,
                motion_source   = motion_source,
            )
            if not is_active:
                st.idle_session_start = 0.0   # started idle from frame 0
            self.states[track_id] = st
            return st

        state      = self.states[track_id]
        prev_state = state.current_state
        new_state  = "ACTIVE" if is_active else "INACTIVE"

        # Clear last_completed_idle every frame — only set on transition frame
        state.last_completed_idle = {}

        # ── Accumulate total tracked time ─────────────────────────────────────
        state.total_tracked += dt

        # ── State transition logic ────────────────────────────────────────────
        if prev_state == "ACTIVE" and new_state == "ACTIVE":
            # Staying active
            state.total_active      += dt
            state.current_idle_secs  = 0.0

        elif prev_state == "INACTIVE" and new_state == "INACTIVE":
            # Continuing idle session
            state.total_idle        += dt
            state.current_idle_secs += dt

        elif prev_state == "ACTIVE" and new_state == "INACTIVE":
            # ── Just went idle — start a new idle session ─────────────────────
            state.total_active       += dt
            state.current_idle_secs   = 0.0
            state.idle_session_start  = state.total_tracked  # video time
            print(
                f"[StateMachine] {state.equipment_id} "
                f"went INACTIVE at {_format_ts(state.total_tracked)}"
            )

        elif prev_state == "INACTIVE" and new_state == "ACTIVE":
            # ── Idle session just ENDED — record it ───────────────────────────
            state.total_idle += dt
            state.current_idle_secs += dt

            # Build the completed idle session record
            session = IdleSession(
                session_number  = state.idle_sessions_count + 1,
                started_at_secs = state.idle_session_start,
                duration_secs   = state.current_idle_secs,
            )
            state.idle_sessions.append(session)

            # Expose it for this frame so db_consumer saves it exactly once
            state.last_completed_idle = session.to_dict()

            print(
                f"[StateMachine] {state.equipment_id} "
                f"back ACTIVE — idle session #{session.session_number} "
                f"lasted {session.duration_ts} "
                f"(started at {session.started_at_ts})"
            )

            # Reset current idle tracking
            state.current_idle_secs  = 0.0
            state.idle_session_start = 0.0

        # ── Update current state ──────────────────────────────────────────────
        state.current_state    = new_state
        state.current_activity = activity if is_active else "WAITING"
        state.motion_source    = motion_source

        return state

    def get_payload(self, state: EquipmentState, frame_id: int) -> dict:
        """
        Build the Kafka JSON payload.

        idle_sessions summary is always included.
        completed_idle_session is only non-empty on the frame an idle session ends.
        """
        payload = {
            "frame_id":        frame_id,
            "equipment_id":    state.equipment_id,
            "equipment_class": state.equipment_class,
            "timestamp":       _format_ts(state.total_tracked),
            "utilization": {
                "current_state":    state.current_state,
                "current_activity": state.current_activity,
                "motion_source":    state.motion_source,
            },
            "time_analytics": {
                "total_tracked_seconds":  round(state.total_tracked, 1),
                "total_active_seconds":   round(state.total_active, 1),
                "total_idle_seconds":     round(state.total_idle, 1),
                "current_idle_seconds":   round(state.current_idle_secs, 1),
                "utilization_percent":    state.utilization_percent,
                "idle_sessions_count":    state.idle_sessions_count,
                "longest_idle_seconds":   round(state.longest_idle_secs, 1),
                "avg_idle_seconds":       round(state.avg_idle_secs, 1),
            },
            # Only populated on the exact frame an idle session completes.
            # db_consumer checks if this is non-empty → inserts one row.
            "completed_idle_session": state.last_completed_idle,
        }
        return payload

    def remove_lost_track(self, track_id: int):
        self.states.pop(track_id, None)