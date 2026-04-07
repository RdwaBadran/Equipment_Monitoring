# Technical Write-Up: Equipment Utilization & Activity Classification Prototype

**System:** Real-time construction equipment monitoring via microservices, Apache Kafka, and computer vision  

---

## 1. System Overview

The system processes video footage of construction sites frame-by-frame, automatically tracking every piece of equipment, determining whether it is actively working or idle at any given second, classifying what specific activity it is performing, and accumulating precise working vs. idle time statistics — all streamed in real time through Apache Kafka to a live web dashboard backed by TimescaleDB.

The pipeline has three independent microservices:

- **CV Service** - reads frames, runs YOLOv8 detection + tracking, optical flow motion analysis, rule-based activity classification, and state machine time tracking, then sends results to Kafka.
- **Consumer Service** — reads from Kafka and writes all events to TimescaleDB.
- **UI Service** - serves a FastAPI REST API and live MJPEG video stream; a browser dashboard polls every 2 seconds.

---

## 2. Solving the Articulated Motion Challenge

### The Problem

This was the core challenge of the project. Standard whole-bounding-box motion analysis computes the average optical flow magnitude across all pixels inside the detection box. For excavators, this fails critically: when the arm is digging, the cab and tracks may be completely stationary. The global average is pulled down by the stationary pixels and falls below the motion threshold — so the system incorrectly marks the machine as `INACTIVE` while it is actively working.

### The Solution: 3-Region Vertical Split

Instead of one global score, the bounding box is divided into three equal vertical thirds. Farneback dense optical flow is computed independently in each region:

```
┌─────────────────────┐  ← y1
│   TOP THIRD         │  motion_score_top    → arm / boom
├─────────────────────┤  ← y1 + bbox_h//3
│   MIDDLE THIRD      │  motion_score_middle → cab / upper body
├─────────────────────┤  ← y1 + 2*bbox_h//3
│   BOTTOM THIRD      │  motion_score_bottom → tracks / undercarriage
└─────────────────────┘  ← y2
```

Each region score is computed as:

```
score = mean(magnitude[magnitude > 0.5])
      only if len(moving_pixels) / total_pixels >= 0.08
      else 0.0
```

The `>= 0.08` ratio filter (8% of pixels must be moving) prevents noise from a handful of flickering pixels from triggering a false ACTIVE reading.

**Motion source determination** — all six region combinations are handled explicitly:

| Combination | `motion_source` | Physical meaning |
|---|---|---|
| Top only | `arm_only` | Arm digging / reaching |
| Middle only | `cab_only` | Cab rotating for swing |
| Bottom only | `tracks` | Machine repositioning |
| Any combination | `full_body` | Multiple parts moving |
| None | `none` | Machine is stationary |

If **any** region is active, `is_active = True`. This is how an excavator digging with stationary tracks is correctly classified as `ACTIVE`.

### Why Farneback Over Other Approaches

The project listed three acceptable approaches: region-based motion analysis, keypoint tracking, and instance segmentation. We chose region-based Farneback optical flow for the following reasons:

**Zero training data required.** Farneback is a classical algorithm implemented in C++ inside OpenCV. It runs in real-time on CPU without any labeled construction equipment videos. Keypoint tracking (e.g., MediaPipe Pose) would need to be fine-tuned on construction equipment joints — expensive to label. Instance segmentation (e.g., YOLOv8-seg) would require a segmentation-annotated dataset.

**Spatially rich output.** Farneback computes a dense 2D displacement vector for every pixel. This gives precise per-region motion scores that directly feed the activity classifier — more informative than sparse keypoints.

**Speed.** At 960×540 resolution cropped to bounding boxes, Farneback runs at ~50ms per bbox on CPU — compatible with real-time processing.

**Trade-off.** Vertical splitting assumes the machine is roughly upright in the frame. A camera mounted at an oblique angle (> ~45°) will misalign the region boundaries with the actual arm/cab/tracks geometry. A future improvement would be to use YOLOv8-pose to detect equipment keypoints and draw region boundaries from the actual joint positions.

---

## 3. Activity Classification

### Architecture

The classifier is rule-based, operating on smoothed optical flow region scores. No ML model was trained. This decision was made for three reasons:

1. No labeled dataset of excavator activities was available.
2. Rule-based logic is fully transparent and debuggable — the exact condition that triggered a classification can be traced directly.
3. Construction equipment activities have clear, physically grounded motion signatures that map reliably to optical flow patterns.

### Smoothing Window

Each frame's region scores are stored in a per-machine 8-frame sliding history. Classification uses the time-averaged scores over this window (`_smooth()`). At `PROCESS_EVERY=5` and 25fps, the window covers 1.6 seconds of video time. This prevents the activity label from flickering between states on every frame — a machine that briefly pauses mid-dig is not reclassified as WAITING.

### Classification Rules (in priority order)

**Rule 1 — DUMPING**
```python
if avg["top"] > 4.5:
    return "DUMPING"
```
Dumping is a fast, violent downward motion as the arm releases its load. The threshold 4.5 is deliberately high — significantly higher than the 1.5 threshold for digging — because dumping involves sudden, large-magnitude motion. Setting it too low causes fast digging strokes to be misclassified as dumping.

**Rule 2 — DIGGING**
```python
if source == "arm_only" and avg["top"] > 1.5:
    return "DIGGING"
```
This is the direct application of the articulated motion solution. If the motion source is `arm_only` (only the top region is moving) and the top score is above 1.5 (clear, sustained arm movement), the machine is digging. This correctly handles an excavator arm digging into the ground while the cab and tracks are stationary.

**Rule 3 — SWINGING/LOADING**
Two sub-cases:
```python
# Sub-case a: cab rotation only (start or end of swing)
if source == "cab_only" and avg["middle"] > 1.0:
    return "SWINGING/LOADING"

# Sub-case b: arm elevated + cab rotating (full swing arc)
if avg["top"] > 1.2 and avg["middle"] > 1.0:
    return "SWINGING/LOADING"
```
Swinging occurs when the cab rotates to move the loaded arm from dig position to dump position. Sub-case (a) catches pure cab rotation — the arm is elevated but relatively still. Sub-case (b) catches the full swing where both arm and cab are moving simultaneously.

**Rule 4 — Track motion (repositioning)**
```python
if avg["bottom"] > avg["top"] and avg["bottom"] > 1.2:
    return "WAITING"
```
If the tracks are moving more than the arm, the machine is repositioning — not performing a defined work cycle activity. `WAITING` is the correct label here rather than inventing a "moving" category not in the spec.

**Default → WAITING**

No explicit `ACTIVE` label exists for "motion that doesn't match any pattern." Falling through all rules means the motion pattern is ambiguous — safe to call WAITING rather than assert a specific work activity.

### Trade-offs

Rule-based classification is transparent and fast, but thresholds need manual tuning per camera angle, distance, and equipment type. A machine far from the camera will produce lower flow scores than a nearby machine performing the same action. An ML approach — such as a small 1D CNN or LSTM running over the 8-frame region score time-series — would learn these relationships from data and generalize better. The cost is needing a labeled dataset of `(flow_scores_sequence → activity_label)` pairs, which was not available for this prototype.

---

## 4. Stable Identity with HSV Re-ID

### The Problem

BoT-SORT maintains track IDs across frames using Kalman filtering. But when a machine goes behind a pile of dirt or another obstacle, BoT-SORT loses the track entirely. When the machine reappears, it gets a new track ID — and all accumulated dwell time is reset to zero.

### The Solution

A lost-track buffer stores every recently-disappeared track's **HSV color histogram** (96-dimensional: 32 bins each for Hue, Saturation, Value). When a new detection appears:

1. Compute its HSV histogram.
2. Compare using cosine similarity against all lost tracks of the same class.
3. If the best match scores ≥ 0.60, restore the original track ID — the machine is recognized.

HSV is used (not RGB) because it separates color (`H`) from brightness (`V`), making the fingerprint more robust to shadows and lighting changes. The histogram is L2-normalized, so the dot product directly equals cosine similarity.

The track histogram uses an Exponential Moving Average (`0.8 × old + 0.2 × new`) to stay stable across frames while slowly adapting to gradual appearance changes (dust accumulation, changing light conditions).

The lost-track buffer expires entries after 8 seconds of wall-clock time. If a machine has been occluded for longer than that, its accumulated dwell time is preserved in the state machine — only future frames are affected.

### Trade-offs

HSV histograms are fast (microseconds per comparison) but cannot distinguish two machines of similar color and size. A learned re-ID embedding (e.g., OSNet) would produce a 512-dimensional feature vector that encodes texture, shape, and color jointly — much stronger discrimination at the cost of an inference pass per detection.

---

## 5. Frame-Accurate Time Tracking

### Why Not Wall-Clock Time

Using `time.time()` for dwell time would measure how long the computer takes to process each frame — not how long the video actually shows. On a fast GPU, 1 processed frame might take 50ms of wall time but represent 200ms of video time. The reported utilization would be completely wrong.

### Solution

```python
dt = process_every / fps   # video seconds per processed frame
```

At `PROCESS_EVERY=5` and `fps=25.0`: `dt = 5/25 = 0.2` seconds per processed frame.

Every time the state machine is updated, it adds exactly `dt` to the appropriate accumulator regardless of wall-clock processing speed. The reported utilization reflects **actual video time** — the number that matters for construction site analysis.

### The 4-State Finite State Machine

```
          [ACTIVE → ACTIVE]        add dt to total_active
          [INACTIVE → INACTIVE]    add dt to total_idle, current_idle_secs
          [ACTIVE → INACTIVE]      start idle session timer
          [INACTIVE → ACTIVE]      close idle session, record to DB
```

The INACTIVE → ACTIVE transition is the most significant: it builds a complete `IdleSession` record (session number, start time, duration) and sets `last_completed_idle` on the state object. This field is cleared at the top of every `update()` call, so it is non-empty only on the single transition frame. The DB consumer checks this field and inserts one row into `equipment_idle_sessions` — exactly once per stop, with no deduplication logic needed.

---

## 6. What Was Added Beyond the project Spec

The project required: ACTIVE/INACTIVE tracking, articulated motion handling, activity classification (4 types), working vs. idle time calculation, Kafka streaming, TimescaleDB sink, and a UI showing live video + machine status + utilization dashboard.

Beyond these requirements, the following were added:

| Addition | Rationale |
|---|---|
| **Per-stop idle session tracking** (`equipment_idle_sessions`) | Total idle time alone is insufficient for cost attribution. Operators need to know *when* each stop occurred and *how long* it lasted. |
| **HSV appearance Re-ID** | Without it, dwell time resets every time a machine is briefly occluded — making utilization percentages unreliable on any real site footage. |
| **`motion_source` field** | Goes deeper than ACTIVE/INACTIVE — tells the operator which physical part of the machine triggered the active state. |
| **Idle session statistics** (`longest_idle_seconds`, `avg_idle_seconds`) | Fleet managers need distribution info, not just totals. |
| **Automatic schema migration** | Consumer runs `ADD COLUMN IF NOT EXISTS` on startup so existing databases upgrade safely. |
| **H.264 re-encoding via ffmpeg** | The annotated output video plays in all browsers and QuickTime, not just VLC. |
| **`health_check.py`** | End-to-end diagnostic that independently verifies all 6 pipeline components — useful for submission reproducibility and CI integration. |
| **`run.py` orchestrator** | Single command starts all services in the correct order, with staggered delays and graceful shutdown. |
| **MJPEG live stream** | The browser dashboard shows the annotated video feed live while processing — not just a static summary. |

---

## 7. Key Design Decisions Summary

| Decision | Choice | Alternative Considered | Reason |
|---|---|---|---|
| Motion analysis method | Farneback dense optical flow | Lucas-Kanade sparse flow, MediaPipe keypoints | Zero training data; dense per-pixel output; runs on CPU in real time |
| Activity classifier | Rule-based on region scores | LSTM / CNN on score time-series | No labeled dataset; fully transparent; fast to tune |
| Object detection | YOLOv8m (COCO-80) | Fine-tuned excavator detector | Pre-trained model available; excavators detected via `boat` class remapping |
| Tracker | BoT-SORT (built into Ultralytics) | ByteTrack, DeepSORT | BoT-SORT handles occlusions best of the options with `persist=True` |
| Re-ID | HSV histogram + cosine similarity | OSNet learned embedding | Fast, no training data needed; sufficient for single-equipment-type scenes |
| Time base | `process_every / fps` | `time.time()` | Must measure video time, not wall-clock processing time |
| Message broker | Apache Kafka | Redis pub/sub, RabbitMQ |  specified Kafka; correct choice for durable event streaming at scale |
| Database | TimescaleDB (PostgreSQL + hypertable) | InfluxDB, plain PostgreSQL |  specified; hypertables give efficient time-range queries as event volume grows |
| UI framework | FastAPI + vanilla JS | Streamlit, Gradio | FastAPI supports async, MJPEG streaming, and typed endpoints; Streamlit does not support MJPEG natively |
| Video feed | MJPEG over HTTP | WebSocket, WebRTC | Browser `<img src="/video">` natively renders MJPEG — no client-side JS required |

---

