import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.spatial import distance
from dataclasses import dataclass, asdict, field

# =========================
# CONFIG CONSTANTS
# =========================

# Model / IO
MODEL_PATH = "yolo11n.pt"
INPUT_VIDEO = "./videos/test3.mp4"
OUTPUT_DIR = "./cache"

# Frame sampling: analyze every N-th frame (e.g., 1, 5, 10, 30, 50...)
FRAME_STEP = 10  # at 30 FPS -> analyze once per second

# Detection settings
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.7
LANE_WIDTH_METERS = 3.5  # standard lane width (static physical width)

# Danger / TTC thresholds (legacy, still used in some fallbacks)
DANGER_DISTANCE_METERS = 2.0        # pedestrian–vehicle proximity in meters
DANGER_DISTANCE_PIXELS_FALLBACK = 50
TTC_PED_VEH_THRESHOLD = 2.0         # seconds
TTC_VEH_VEH_THRESHOLD = 1.5         # seconds
MAX_TTC_SECONDS = 5.0               # upper bound to consider TTC meaningful

STOPPED_SPEED_THRESHOLD_MPS = 0.5   # m/s considered "stopped"
STOPPED_DWELL_SECONDS = 5.0         # how long a vehicle must be stopped to raise event

# === Ego-collision detection sensitivity (dashcam car) ===
# Object must be within this distance ahead of the ego-vehicle
EGO_COLLISION_DISTANCE_METERS = 5.0      # longitudinal distance ahead of ego
# Time-to-collision threshold to flag an imminent collision
EGO_COLLISION_TTC_THRESHOLD = 2.0        # seconds
# Lateral tolerance around ego lane center (in addition to lane half-width)
EGO_LANE_TOLERANCE_METERS = 1.0

# Heatmap
HEATMAP_RADIUS = 20                 # pixels

# Event reporting
EVENT_COOLDOWN_SECONDS = 5.0        # report the same logical event at most once per 5s

# Mock world location: Tallinn University of Technology
MOCK_WORLD_LOCATION = {
    "name": "Tallinn University of Technology",
    "lat": 59.3949,
    "lon": 24.6648
}

# Mock video start datetime for overlay
VIDEO_START_DATETIME = datetime(2025, 11, 18, 12, 0, 0)

# COCO class mappings (YOLO11 on COCO)
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
PEDESTRIAN_CLASSES = {0: 'person'}
CYCLIST_CLASSES = {1: 'bicycle'}


# =========================
# DATA STRUCTURES
# =========================

@dataclass
class DangerousEvent:
    """Data class for dangerous events"""
    id: int
    timestamp: str
    frame_number: int
    event_type: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    objects_involved: List[int]
    ttc: Optional[float] = None
    location: Optional[Tuple[int, int]] = None  # pixel coords (for overlay)
    world_location: Optional[Dict[str, float]] = None  # mock geo-location

    def to_dict(self):
        data = asdict(self)
        data['objects_involved'] = [int(x) for x in data['objects_involved']]
        if data['location'] is not None:
            data['location'] = tuple(int(x) for x in data['location'])
        return data


@dataclass
class AnalyzerState:
    """Holds all mutable analysis state."""
    track_history: Dict[int, deque] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=30))
    )
    track_speeds: Dict[int, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    track_dwell_times: Dict[int, float] = field(default_factory=dict)
    track_first_seen: Dict[int, int] = field(default_factory=dict)

    object_counts_per_frame: List[Dict] = field(default_factory=list)
    heatmap_accumulator: Optional[np.ndarray] = None
    line_crossings: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    dangerous_events: List[DangerousEvent] = field(default_factory=list)
    event_counter: int = 0

    pixels_per_meter: Optional[float] = None

    # Ego-lane geometry (in image space)
    lane_width_pixels: Optional[float] = None
    lane_direction: Optional[Tuple[float, float]] = None  # unit vector along lane
    lane_center_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    lane_left_line: Optional[Tuple[float, float]] = None   # x = a*y + b
    lane_right_line: Optional[Tuple[float, float]] = None  # x = a*y + b

    regions: Dict[str, np.ndarray] = field(default_factory=dict)
    crossing_lines: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = field(
        default_factory=dict
    )

    # For throttling event reporting
    last_event_times: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """High-level result returned by analyze_video()."""
    state: AnalyzerState
    fps: float
    total_frames: int
    video_name: str


# =========================
# LANE WIDTH & GEOMETRY ESTIMATION
# =========================

def estimate_lane_geometry(frame: np.ndarray) -> Optional[dict]:
    """
    Estimate ego-lane geometry from a dashcam frame.

    Returns a dict with:
        - 'lane_width_pixels': float
        - 'left_line': (a_left, b_left) for x = a*y + b
        - 'right_line': (a_right, b_right)
        - 'lane_center_line': ((x_bottom, y_bottom), (x_top, y_top))
        - 'lane_direction': (dx, dy) unit vector along lane centerline
    or None if detection fails.
    """
    h, w = frame.shape[:2]

    # 1) Preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    # 2) Region of interest: bottom central trapezoid
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.45), int(h * 0.6)),
        (int(w * 0.55), int(h * 0.6)),
        (int(w * 0.95), h)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    edges_roi = cv2.bitwise_and(edges, mask)

    # 3) Hough transform
    lines = cv2.HoughLinesP(
        edges_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=int(w * 0.2),
        maxLineGap=40
    )

    if lines is None:
        return None

    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)

        # Left lane: negative slope, right lane: positive slope
        if slope < -0.3:
            left_lines.append((x1, y1, x2, y2, slope))
        elif slope > 0.3:
            right_lines.append((x1, y1, x2, y2, slope))

    if not left_lines or not right_lines:
        return None

    def fit_line(lines_list):
        xs = []
        ys = []
        for x1, y1, x2, y2, _ in lines_list:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        # Fit x = a*y + b (y varies more in perspective road view)
        A = np.vstack([ys, np.ones_like(ys)]).T
        a, b = np.linalg.lstsq(A, xs, rcond=None)[0]
        return a, b  # x = a*y + b

    a_left, b_left = fit_line(left_lines)
    a_right, b_right = fit_line(right_lines)

    # Evaluate lane width near the bottom of the image
    y_eval = int(h * 0.9)
    x_left = a_left * y_eval + b_left
    x_right = a_right * y_eval + b_right

    lane_width = abs(x_right - x_left)
    if lane_width < 10:  # clearly bogus
        return None

    # Define a centerline for the ego lane
    y_bottom = h
    y_top = int(h * 0.6)

    x_left_bottom = a_left * y_bottom + b_left
    x_right_bottom = a_right * y_bottom + b_right
    x_center_bottom = (x_left_bottom + x_right_bottom) / 2.0

    x_left_top = a_left * y_top + b_left
    x_right_top = a_right * y_top + b_right
    x_center_top = (x_left_top + x_right_top) / 2.0

    p_bottom = np.array([x_center_bottom, y_bottom], dtype=np.float32)
    p_top = np.array([x_center_top, y_top], dtype=np.float32)

    lane_vec = p_top - p_bottom
    norm = np.linalg.norm(lane_vec)
    if norm < 1e-6:
        return None
    lane_dir = (lane_vec / norm).astype(np.float32)

    lane_center_line = (
        (int(x_center_bottom), int(y_bottom)),
        (int(x_center_top), int(y_top))
    )

    return {
        "lane_width_pixels": float(lane_width),
        "left_line": (float(a_left), float(b_left)),
        "right_line": (float(a_right), float(b_right)),
        "y_eval": int(y_eval),
        "lane_center_line": lane_center_line,
        "lane_direction": (float(lane_dir[0]), float(lane_dir[1]))
    }


def estimate_lane_width_pixels(frame: np.ndarray) -> Optional[float]:
    """
    Backwards-compatible wrapper – use lane geometry, return only width.
    """
    geom = estimate_lane_geometry(frame)
    return geom["lane_width_pixels"] if geom is not None else None


# =========================
# GEOMETRY & REGIONS
# =========================

def setup_regions(state: AnalyzerState, frame_shape: Tuple[int, int]):
    """
    Setup analysis regions and crossing lines based on frame size.
    """
    h, w = frame_shape[:2]

    state.regions = {
        'left_lane': np.array([[0, h // 3], [w // 3, h // 3],
                               [w // 3, 2 * h // 3], [0, 2 * h // 3]]),
        'center_lane': np.array([[w // 3, h // 3], [2 * w // 3, h // 3],
                                 [2 * w // 3, 2 * h // 3], [w // 3, 2 * h // 3]]),
        'right_lane': np.array([[2 * w // 3, h // 3], [w, h // 3],
                                [w, 2 * h // 3], [2 * w // 3, 2 * h // 3]]),
        'crosswalk': np.array([[w // 4, h // 2], [3 * w // 4, h // 2],
                               [3 * w // 4, 2 * h // 3], [w // 4, 2 * h // 3]])
    }

    # Single horizontal counting line
    state.crossing_lines = {
        'main_line': ((0, int(h * 0.6)), (w, int(h * 0.6)))
    }


def calibrate_pixel_to_meter(state: AnalyzerState,
                             frame: np.ndarray,
                             lane_width_pixels: Optional[int] = None):
    """
    Calibrate pixel-to-meter conversion using ego-lane width and direction.

    Preferred:
        - Detect ego-lane width and centerline from the frame via lane detection.
    Fallback:
        - Use frame_width / 3 as lane width and no lane direction.
    """
    h, w = frame.shape[:2]

    lane_geom = None

    # Try automatic ego-lane estimation if not supplied explicitly
    if lane_width_pixels is None:
        lane_geom = estimate_lane_geometry(frame)
        if lane_geom is not None:
            lane_width_pixels = lane_geom["lane_width_pixels"]
            state.lane_width_pixels = lane_width_pixels
            state.lane_direction = lane_geom["lane_direction"]
            state.lane_center_line = lane_geom["lane_center_line"]
            state.lane_left_line = lane_geom["left_line"]
            state.lane_right_line = lane_geom["right_line"]

    # Fallback if detection failed
    if lane_width_pixels is None:
        lane_width_pixels = w // 3  # crude guess
        state.lane_width_pixels = lane_width_pixels
        state.lane_direction = None
        state.lane_center_line = None
        state.lane_left_line = None
        state.lane_right_line = None

    # Use static physical lane width (3.5 m) for scaling
    state.pixels_per_meter = lane_width_pixels / LANE_WIDTH_METERS


def point_in_region(state: AnalyzerState,
                    point: Tuple[int, int],
                    region_name: str) -> bool:
    if region_name not in state.regions:
        return False
    return cv2.pointPolygonTest(state.regions[region_name], point, False) >= 0


def line_intersection(p1: Tuple[int, int], p2: Tuple[int, int],
                      p3: Tuple[int, int], p4: Tuple[int, int]) -> bool:
    """Check if line segment p1-p2 intersects with p3-p4."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def check_line_crossing(state: AnalyzerState,
                        track_id: int,
                        current_pos: Tuple[int, int]):
    """
    Check if object crossed any counting lines.
    """
    history = state.track_history[track_id]
    if len(history) < 2:
        return

    prev_pos = history[-2]

    for line_name, (p1, p2) in state.crossing_lines.items():
        if line_intersection(prev_pos, current_pos, p1, p2):
            state.line_crossings[line_name] += 1


# =========================
# SPEED & TTC
# =========================

def calculate_speed(state: AnalyzerState,
                    track_id: int,
                    fps: float) -> Optional[float]:
    """
    Calculate speed in m/s using last two positions in history and FRAME_STEP.

    Uses the ego-lane direction (if available) so that motion is measured
    along the road; falls back to vertical motion if lane direction is unknown.

    Note: for dashcam footage, this is the speed of the object relative to
    the camera, not necessarily absolute world speed.
    """
    history = state.track_history[track_id]
    if len(history) < 2 or state.pixels_per_meter is None or fps <= 0:
        return None

    prev = np.array(history[-2], dtype=float)
    curr = np.array(history[-1], dtype=float)
    delta = curr - prev  # (dx, dy) in pixels

    # Prefer motion along lane centerline
    if state.lane_direction is not None:
        lane_dir = np.array(state.lane_direction, dtype=float)
        norm = np.linalg.norm(lane_dir)
        if norm > 1e-6:
            lane_dir /= norm
            # Scalar displacement along the lane (can be positive or negative)
            pixel_distance = abs(float(np.dot(delta, lane_dir)))
        else:
            # Degenerate lane direction, fall back
            pixel_distance = abs(delta[1])
    else:
        # Fallback: assume road aligned roughly with vertical axis
        pixel_distance = abs(delta[1])

    meter_distance = pixel_distance / state.pixels_per_meter
    dt_seconds = FRAME_STEP / fps  # because we process every FRAME_STEP-th frame

    if dt_seconds <= 0:
        return None

    speed_mps = meter_distance / dt_seconds
    return speed_mps


def calculate_ttc(state: AnalyzerState,
                  obj1: Dict,
                  obj2: Dict,
                  fps: float) -> Optional[float]:
    """
    Calculate Time-To-Collision between two objects using relative velocity
    projected along the line-of-sight. Returns TTC in seconds.

    (Kept for potential future use, but not used in ego-collision logic.)
    """
    track_id1 = obj1['track_id']
    track_id2 = obj2['track_id']

    history1 = list(state.track_history.get(track_id1, []))
    history2 = list(state.track_history.get(track_id2, []))

    if len(history1) < 2 or len(history2) < 2 or fps <= 0:
        return None

    dt = FRAME_STEP / fps
    if dt <= 0:
        return None

    pos1_prev = np.array(history1[-2], dtype=float)
    pos1_curr = np.array(history1[-1], dtype=float)
    pos2_prev = np.array(history2[-2], dtype=float)
    pos2_curr = np.array(history2[-1], dtype=float)

    # Positions at "now"
    pos1 = pos1_curr
    pos2 = pos2_curr

    # Velocities (px/s)
    v1 = (pos1_curr - pos1_prev) / dt
    v2 = (pos2_curr - pos2_prev) / dt

    rel_pos = pos2 - pos1        # from obj1 to obj2
    rel_vel = v2 - v1            # velocity of obj2 relative to obj1

    rel_speed_sq = float(np.dot(rel_vel, rel_vel))
    if rel_speed_sq < 1e-6:
        return None  # stationary relative to each other

    # TTC along relative motion:
    ttc = -np.dot(rel_pos, rel_vel) / rel_speed_sq

    if not (0 < ttc < MAX_TTC_SECONDS):
        return None

    return float(ttc)


# =========================
# HEATMAP
# =========================

def update_heatmap(state: AnalyzerState,
                   objects: List[Dict],
                   frame_shape: Tuple[int, int]):
    """
    Update occupancy heatmap with detected objects.
    """
    h, w = frame_shape[:2]

    if state.heatmap_accumulator is None:
        state.heatmap_accumulator = np.zeros((h, w), dtype=np.float32)

    for obj in objects:
        x, y = map(int, obj['center'])
        # Clamp to valid range
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        cv2.circle(state.heatmap_accumulator, (x, y), HEATMAP_RADIUS, 1.0, -1)


def generate_heatmap_visualization(state: AnalyzerState, output_path: str):
    """Generate and save heatmap visualization from accumulator."""
    if state.heatmap_accumulator is None:
        return

    acc = state.heatmap_accumulator.copy()

    # Smooth to get a more "continuous" heatmap
    acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=10, sigmaY=10)

    # Normalize to 0–255
    heatmap_norm = cv2.normalize(
        acc, None, 0, 255,
        cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # OpenCV color heatmap
    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, heatmap_colored)

    # Matplotlib version
    plt.figure(figsize=(12, 8))
    plt.imshow(acc, cmap='hot', interpolation='bilinear', origin='upper')
    plt.colorbar(label='Occupancy Density')
    plt.title('Traffic Occupancy Heatmap')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path.replace('.jpg', '_matplotlib.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# =========================
# EVENT LOGIC
# =========================

def make_event_key(event_type: str, objects_involved: List[int]) -> str:
    """Create a stable key for de-duplicating events per type + object set."""
    ids = sorted(int(x) for x in objects_involved)
    return f"{event_type}:" + ",".join(map(str, ids))


def should_report_event(state: AnalyzerState,
                        event_key: str,
                        current_time_sec: float) -> bool:
    """
    Enforce EVENT_COOLDOWN_SECONDS so we don't spam the same event.
    """
    last_time = state.last_event_times.get(event_key)
    if last_time is None:
        return True
    return (current_time_sec - last_time) >= EVENT_COOLDOWN_SECONDS


def register_event(state: AnalyzerState,
                   event: DangerousEvent,
                   event_key: str,
                   current_time_sec: float):
    state.dangerous_events.append(event)
    state.event_counter += 1
    state.last_event_times[event_key] = current_time_sec


def check_dangerous_events(state: AnalyzerState,
                           detected_objects: List[Dict],
                           frame_number: int,
                           timestamp: str,
                           fps: float,
                           frame_shape: Tuple[int, int]):
    """
    Detect dangerous traffic events where an object is about to collide
    with the ego vehicle (dashcam car).

    The ego vehicle is assumed to be at the bottom of the ego-lane centerline
    (if available) or at the center of the bottom image edge.

    Sensitivity is controlled via:
        - EGO_COLLISION_DISTANCE_METERS
        - EGO_COLLISION_TTC_THRESHOLD
        - EGO_LANE_TOLERANCE_METERS
    """
    current_time_sec = frame_number / fps if fps > 0 else 0.0
    h, w = frame_shape[:2]

    # Pixel thresholds derived from meter-based constants
    if state.pixels_per_meter is not None:
        collision_distance_px = EGO_COLLISION_DISTANCE_METERS * state.pixels_per_meter
        lane_tolerance_px = (
            (LANE_WIDTH_METERS / 2.0 + EGO_LANE_TOLERANCE_METERS)
            * state.pixels_per_meter
        )
    else:
        # Fallback thresholds when calibration is unavailable
        collision_distance_px = DANGER_DISTANCE_PIXELS_FALLBACK * 3.0
        lane_tolerance_px = w * 0.25

    # Ego reference point (approx front bumper of dashcam car)
    if state.lane_center_line is not None:
        ego_x, ego_y = state.lane_center_line[0]
    else:
        ego_x, ego_y = w // 2, h

    for obj in detected_objects:
        cls_id = obj['class_id']

        # Only care about road users that can collide with ego
        if cls_id not in VEHICLE_CLASSES and \
           cls_id not in PEDESTRIAN_CLASSES and \
           cls_id not in CYCLIST_CLASSES:
            continue

        track_id = obj['track_id']
        center_x, center_y = obj['center']

        # Object must be in front of the ego-vehicle in image space
        if center_y >= ego_y:
            continue

        # Lateral alignment with ego lane
        lateral_offset_px = abs(center_x - ego_x)
        if lateral_offset_px > lane_tolerance_px:
            continue  # object is in another lane / sidewalk far away

        # Longitudinal distance (ahead of ego)
        distance_ahead_px = ego_y - center_y
        if distance_ahead_px > collision_distance_px:
            continue  # too far ahead

        # Need history to compute approach speed
        history = list(state.track_history.get(track_id, []))
        if len(history) < 2 or fps <= 0:
            continue

        prev_x, prev_y = history[-2]
        curr_x, curr_y = history[-1]
        dt = FRAME_STEP / fps
        if dt <= 0:
            continue

        # Positive dy => moving downward (towards ego)
        dy = curr_y - prev_y
        if dy <= 0:
            continue  # not approaching

        approach_speed_px_per_s = dy / dt
        ttc = distance_ahead_px / approach_speed_px_per_s  # seconds

        if not (0 < ttc <= EGO_COLLISION_TTC_THRESHOLD):
            continue

        # All criteria satisfied => imminent collision with ego
        if state.pixels_per_meter is not None:
            distance_ahead_m = distance_ahead_px / state.pixels_per_meter
        else:
            distance_ahead_m = None

        if cls_id in PEDESTRIAN_CLASSES:
            obj_type = "pedestrian"
        elif cls_id in CYCLIST_CLASSES:
            obj_type = "cyclist"
        else:
            obj_type = "vehicle"

        event_type = f"imminent_collision_ego_{obj_type}"
        event_key = make_event_key(event_type, [track_id])

        if not should_report_event(state, event_key, current_time_sec):
            continue

        # Severity based on TTC
        if ttc < EGO_COLLISION_TTC_THRESHOLD * 0.5:
            severity = "critical"
        else:
            severity = "high"

        description_parts = [
            f"Imminent collision with {obj_type}",
            f"TTC={ttc:.2f}s"
        ]
        if distance_ahead_m is not None:
            description_parts.append(f"distance≈{distance_ahead_m:.1f}m")

        event = DangerousEvent(
            id=state.event_counter,
            timestamp=timestamp,
            frame_number=frame_number,
            event_type=event_type,
            description=", ".join(description_parts),
            severity=severity,
            objects_involved=[track_id],
            ttc=float(ttc),
            location=obj['center'],
            world_location=MOCK_WORLD_LOCATION.copy()
        )
        register_event(state, event, event_key, current_time_sec)


# =========================
# OVERLAYS
# =========================

def draw_overlays(state: AnalyzerState,
                  frame: np.ndarray,
                  detected_objects: List[Dict],
                  frame_number: int,
                  fps: float,
                  current_datetime_str: str) -> np.ndarray:
    """
    Draw all analysis overlays onto the frame.
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Ego-lane centerline (if detected)
    if state.lane_center_line is not None:
        p1, p2 = state.lane_center_line
        cv2.line(overlay, p1, p2, (255, 255, 0), 2)
        cv2.putText(overlay, "ego lane", (p1[0] + 10, p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Lane boundary lines (x = a*y + b form)
    if state.lane_left_line is not None:
        a_left, b_left = state.lane_left_line
        y0, y1 = int(h * 0.6), h
        x0 = int(a_left * y0 + b_left)
        x1 = int(a_left * y1 + b_left)
        cv2.line(overlay, (x0, y0), (x1, y1), (255, 0, 255), 2)

    if state.lane_right_line is not None:
        a_right, b_right = state.lane_right_line
        y0, y1 = int(h * 0.6), h
        x0 = int(a_right * y0 + b_right)
        x1 = int(a_right * y1 + b_right)
        cv2.line(overlay, (x0, y0), (x1, y1), (255, 0, 255), 2)

    # Ego reference position (approx front bumper of dashcam car)
    if state.lane_center_line is not None:
        ego_x, ego_y = state.lane_center_line[0]
    else:
        ego_x, ego_y = w // 2, h

    cv2.circle(overlay, (int(ego_x), int(ego_y)), 8, (255, 255, 255), -1)
    cv2.putText(overlay, "EGO", (int(ego_x) + 10, int(ego_y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Regions
    for region_name, polygon in state.regions.items():
        overlay_region = overlay.copy()
        cv2.polylines(overlay_region, [polygon], True, (0, 255, 255), 2)
        cv2.fillPoly(overlay_region, [polygon], (0, 255, 255))
        cv2.addWeighted(overlay_region, 0.1, overlay, 0.9, 0, overlay)

    # Counting lines
    for line_name, (p1, p2) in state.crossing_lines.items():
        cv2.line(overlay, p1, p2, (0, 255, 0), 3)
        cv2.putText(overlay, f"{line_name}: {state.line_crossings[line_name]}",
                    (p1[0], p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Detected objects (using latest known data)
    for obj in detected_objects:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        track_id = obj['track_id']
        class_name = obj['class_name']
        speed = obj.get('speed')

        # Color coding
        if obj['class_id'] in VEHICLE_CLASSES:
            color = (0, 255, 0)
        elif obj['class_id'] in PEDESTRIAN_CLASSES:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{track_id} {class_name}"
        if speed is not None:
            label += f" {speed * 3.6:.1f}km/h"
        cv2.putText(overlay, label, (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Trajectory
        history = state.track_history[track_id]
        if len(history) > 1:
            points = np.array(list(history), dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [points], False, color, 2)

    # Recent dangerous events (~ last 1 second)
    time_window_frames = int(fps) if fps > 0 else 0
    recent_events = [e for e in state.dangerous_events
                     if abs(e.frame_number - frame_number) < time_window_frames]

    for event in recent_events:
        if event.location:
            x, y = map(int, event.location)
            if event.severity == 'critical':
                color = (0, 0, 255)
            elif event.severity == 'high':
                color = (0, 165, 255)
            else:
                color = (0, 255, 255)

            cv2.circle(overlay, (x, y), 30, color, 3)
            cv2.putText(overlay, f"⚠ {event.event_type}", (x + 35, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Stats overlay (bottom-left)
    stats_y = 30
    cv2.putText(overlay, f"Frame: {frame_number}", (10, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    stats_y += 30
    cv2.putText(overlay, f"Objects: {len(detected_objects)}", (10, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    stats_y += 30
    cv2.putText(overlay, f"Events: {len(state.dangerous_events)}", (10, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mock datetime overlay (top-right)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = current_datetime_str
    (text_w, text_h), _ = cv2.getTextSize(text, font, 0.7, 2)
    x_pos = w - text_w - 10
    y_pos = 30
    cv2.putText(overlay, text, (x_pos, y_pos),
                font, 0.7, (255, 255, 255), 2)

    return overlay


# =========================
# REPORT GENERATION
# =========================

def generate_time_profile(state: AnalyzerState,
                          output_path: Path,
                          video_name: str,
                          fps: float):
    """Generate time-series plots of object counts."""
    if not state.object_counts_per_frame or fps <= 0:
        return

    frames = [d['frame'] for d in state.object_counts_per_frame]
    times = [f / fps for f in frames]
    totals = [d['total'] for d in state.object_counts_per_frame]

    # Aggregate by fixed time window in seconds
    window_seconds = 5.0
    aggregated_times = []
    aggregated_counts = []

    current_start = 0.0
    max_time = times[-1]

    while current_start <= max_time:
        current_end = current_start + window_seconds
        window_values = [
            d['total'] for d in state.object_counts_per_frame
            if current_start <= d['frame'] / fps < current_end
        ]
        if window_values:
            aggregated_times.append(current_start + window_seconds / 2.0)
            aggregated_counts.append(float(np.mean(window_values)))
        current_start += window_seconds

    plt.figure(figsize=(14, 6))

    # Left: per-analyzed-frame and 5s avg
    plt.subplot(1, 2, 1)
    plt.plot(times, totals, alpha=0.5, label='Per analyzed frame')
    if aggregated_times:
        plt.plot(aggregated_times, aggregated_counts, linewidth=2,
                 label='5s average')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Object Count')
    plt.title('Object Count Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Right: by class
    plt.subplot(1, 2, 2)
    class_series = defaultdict(list)
    for d in state.object_counts_per_frame:
        for class_name in ['car', 'truck', 'bus', 'person', 'bicycle']:
            class_series[class_name].append(d['by_class'].get(class_name, 0))

    for class_name, counts in class_series.items():
        if sum(counts) > 0:
            plt.plot(times, counts, label=class_name, alpha=0.7)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Object Count')
    plt.title('Object Count by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f"{video_name}_time_profile.png", dpi=150)
    plt.close()


def generate_speed_distribution(state: AnalyzerState,
                                output_path: Path,
                                video_name: str):
    """Generate speed distribution plots."""
    all_speeds_kmh = []
    for speeds in state.track_speeds.values():
        all_speeds_kmh.extend([s * 3.6 for s in speeds if s is not None])

    if not all_speeds_kmh:
        return

    plt.figure(figsize=(10, 6))
    plt.hist(all_speeds_kmh, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Frequency')
    plt.title('Vehicle Speed Distribution (relative to camera)')
    plt.grid(True, alpha=0.3)

    mean_speed = np.mean(all_speeds_kmh)
    median_speed = np.median(all_speeds_kmh)
    plt.axvline(mean_speed, color='r', linestyle='--',
                label=f'Mean: {mean_speed:.1f} km/h')
    plt.axvline(median_speed, color='g', linestyle='--',
                label=f'Median: {median_speed:.1f} km/h')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path / f"{video_name}_speed_distribution.png", dpi=150)
    plt.close()


def generate_summary_statistics(state: AnalyzerState,
                                output_path: Path,
                                video_name: str,
                                fps: float,
                                total_frames: int):
    """Generate summary statistics JSON + console print."""
    total_crossings = sum(state.line_crossings.values())
    total_dangerous_events = len(state.dangerous_events)

    event_types = defaultdict(int)
    for event in state.dangerous_events:
        event_types[event.event_type] += 1

    all_speeds_kmh = []
    for speeds in state.track_speeds.values():
        all_speeds_kmh.extend([s * 3.6 for s in speeds if s is not None])

    speed_stats = {
        'mean': float(np.mean(all_speeds_kmh)) if all_speeds_kmh else None,
        'median': float(np.median(all_speeds_kmh)) if all_speeds_kmh else None,
        'std': float(np.std(all_speeds_kmh)) if all_speeds_kmh else None,
        'min': float(np.min(all_speeds_kmh)) if all_speeds_kmh else None,
        'max': float(np.max(all_speeds_kmh)) if all_speeds_kmh else None
    }

    class_totals = defaultdict(int)
    for frame_data in state.object_counts_per_frame:
        for class_name, count in frame_data['by_class'].items():
            class_totals[class_name] += count

    summary = {
        'video_name': video_name,
        'total_frames': total_frames,
        'duration_seconds': total_frames / fps if fps > 0 else None,
        'fps': fps,
        'frame_step': FRAME_STEP,
        'total_line_crossings': total_crossings,
        'line_crossings_detail': dict(state.line_crossings),
        'total_dangerous_events': total_dangerous_events,
        'dangerous_events_by_type': dict(event_types),
        'unique_tracks': len(state.track_history),
        'object_class_totals': dict(class_totals),
        'speed_statistics_kmh': speed_stats
    }

    summary_file = output_path / f"{video_name}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary statistics saved to: {summary_file}")
    print(f"\n=== Analysis Summary ===")
    print(f"Total dangerous events detected: {total_dangerous_events}")
    print(f"Total line crossings: {total_crossings}")
    print(f"Unique tracked objects: {len(state.track_history)}")
    if all_speeds_kmh:
        print(f"Average relative speed: {speed_stats['mean']:.1f} km/h")


def generate_reports(result: AnalysisResult, output_path: Path):
    """Generate JSON, heatmap, plots and summary."""
    state = result.state
    video_name = result.video_name
    fps = result.fps
    total_frames = result.total_frames

    # 1. Dangerous events JSON
    events_file = output_path / f"{video_name}_dangerous_events.json"
    with open(events_file, 'w') as f:
        json.dump([e.to_dict() for e in state.dangerous_events], f, indent=2)
    print(f"Dangerous events saved to: {events_file}")

    # 2. Heatmap
    heatmap_file = output_path / f"{video_name}_heatmap.jpg"
    generate_heatmap_visualization(state, str(heatmap_file))
    print(f"Heatmap saved to: {heatmap_file}")

    # 3. Time profile
    generate_time_profile(state, output_path, video_name, fps)

    # 4. Speed distribution
    generate_speed_distribution(state, output_path, video_name)

    # 5. Summary statistics
    generate_summary_statistics(state, output_path, video_name, fps, total_frames)


# =========================
# MAIN ANALYSIS FUNCTION
# =========================

def analyze_video(video_path: str,
                  output_dir: str = OUTPUT_DIR) -> AnalysisResult:
    """
    Main analysis function - process entire video using YOLO11n.
    Only every FRAME_STEP-th frame is analyzed, but ALL frames are written
    to the output video with overlays based on the latest analysis.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {video_path}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, "
          f"Frames: {total_frames}, Frame step: {FRAME_STEP}")

    video_name = Path(video_path).stem
    output_video_path = output_path / f"{video_name}_analyzed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps,
                          (frame_width, frame_height))

    # Read first frame for calibration
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    state = AnalyzerState()
    setup_regions(state, (frame_height, frame_width))
    calibrate_pixel_to_meter(state, first_frame)

    model = YOLO(MODEL_PATH)

    frame_number = 0
    last_detected_objects: List[Dict] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # Compute mock datetime based on real video time
        if fps > 0:
            elapsed_seconds = frame_number / fps
        else:
            elapsed_seconds = 0.0
        current_dt = VIDEO_START_DATETIME + timedelta(seconds=elapsed_seconds)
        current_dt_str = current_dt.strftime("%d/%m/%Y %H:%M:%S")

        detected_objects: List[Dict] = last_detected_objects

        # Analyze only every N-th frame
        if FRAME_STEP > 0 and frame_number % FRAME_STEP == 0:
            timestamp_sec = frame_number / fps if fps > 0 else 0.0
            timestamp = str(timedelta(seconds=timestamp_sec))

            results = model.track(
                frame,
                persist=True,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False
            )

            detected_objects = []

            if results and results[0].boxes is not None and \
                    results[0].boxes.id is not None:

                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()

                for box, track_id, class_id, conf in zip(
                        boxes, track_ids, classes, confidences
                ):
                    x1, y1, x2, y2 = box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    class_name = model.names[int(class_id)]

                    # Update track history
                    state.track_history[track_id].append((center_x, center_y))

                    # Calculate speed (along ego-lane direction if available)
                    speed_mps = calculate_speed(state, track_id, fps)
                    if speed_mps is not None:
                        state.track_speeds[track_id].append(speed_mps)

                    # Line crossings
                    check_line_crossing(state, track_id, (center_x, center_y))

                    obj_data = {
                        'bbox': box,
                        'center': (center_x, center_y),
                        'track_id': track_id,
                        'class_id': int(class_id),
                        'class_name': class_name,
                        'confidence': float(conf),
                        'speed': speed_mps
                    }
                    detected_objects.append(obj_data)

                # Heatmap update
                update_heatmap(state, detected_objects, frame.shape)

                # Dangerous events (ego-collision only)
                check_dangerous_events(
                    state,
                    detected_objects,
                    frame_number,
                    timestamp,
                    fps,
                    frame.shape
                )

            # Per-analyzed-frame counts (for reports)
            class_counts = defaultdict(int)
            for obj in detected_objects:
                class_counts[obj['class_name']] += 1

            state.object_counts_per_frame.append({
                'frame': frame_number,
                'timestamp': timestamp,
                'total': len(detected_objects),
                'by_class': dict(class_counts)
            })

            last_detected_objects = detected_objects

            if frame_number % (FRAME_STEP * 100) == 0:
                print(f"Analyzed frame {frame_number}/{total_frames} "
                      f"({frame_number / total_frames * 100:.1f}%)")

        # Draw overlays using latest known detections (even on skipped frames)
        annotated_frame = draw_overlays(
            state,
            frame,
            last_detected_objects,
            frame_number,
            fps,
            current_datetime_str=current_dt_str
        )
        out.write(annotated_frame)

    cap.release()
    out.release()

    print(f"Video analysis complete. Output saved to: {output_video_path}")

    result = AnalysisResult(
        state=state,
        fps=fps,
        total_frames=total_frames,
        video_name=video_name
    )

    generate_reports(result, output_path)

    return result


# =========================
# COMPARISON FUNCTION
# =========================

def compare_time_periods(result1: AnalysisResult,
                         result2: AnalysisResult,
                         period1_name: str = "Morning",
                         period2_name: str = "Daytime",
                         output_path: str = "./comparison_report.txt"):
    """
    Compare traffic patterns between two analyzed periods (two videos).
    """
    state1, fps1 = result1.state, result1.fps
    state2, fps2 = result2.state, result2.fps

    report_lines = []
    report_lines.append(f"=== Traffic Comparison: {period1_name} vs "
                        f"{period2_name} ===\n")

    # Average objects per analyzed frame
    total1 = sum(d['total'] for d in state1.object_counts_per_frame)
    total2 = sum(d['total'] for d in state2.object_counts_per_frame)
    avg1 = total1 / len(state1.object_counts_per_frame) \
        if state1.object_counts_per_frame else 0
    avg2 = total2 / len(state2.object_counts_per_frame) \
        if state2.object_counts_per_frame else 0

    report_lines.append("Average Objects per Analyzed Frame:")
    report_lines.append(f"  {period1_name}: {avg1:.2f}")
    report_lines.append(f"  {period2_name}: {avg2:.2f}")
    if avg1 > 0:
        diff_pct = ((avg2 - avg1) / avg1 * 100.0)
        report_lines.append(f"  Difference: {diff_pct:.1f}% "
                            f"{'increase' if avg2 > avg1 else 'decrease'}\n")
    else:
        report_lines.append("  Difference: N/A (no data in first period)\n")

    # Speed comparison (relative speeds)
    speeds1 = [s * 3.6 for speeds in state1.track_speeds.values()
               for s in speeds if s]
    speeds2 = [s * 3.6 for speeds in state2.track_speeds.values()
               for s in speeds if s]

    if speeds1 and speeds2:
        report_lines.append("Speed Comparison (km/h, relative to camera):")
        report_lines.append(f"  {period1_name} avg: {np.mean(speeds1):.1f}")
        report_lines.append(f"  {period2_name} avg: {np.mean(speeds2):.1f}")
        report_lines.append(
            f"  Difference: "
            f"{(np.mean(speeds2) - np.mean(speeds1)):.1f} km/h\n"
        )

    # Dangerous events
    report_lines.append("Dangerous Events:")
    report_lines.append(f"  {period1_name}: {len(state1.dangerous_events)}")
    report_lines.append(f"  {period2_name}: {len(state2.dangerous_events)}\n")

    # Simple interpretation
    report_lines.append("=== Urban Planning Implications ===")
    if avg2 > avg1:
        report_lines.append(f"• Higher traffic volume during {period2_name} "
                            f"suggests peak congestion period.")
        report_lines.append("• May require additional traffic control "
                            "measures.")
    if speeds1 and speeds2 and np.mean(speeds1) > np.mean(speeds2):
        report_lines.append(f"• Slower relative speeds during {period2_name} "
                            f"indicate more congestion.")
        report_lines.append("• Consider traffic flow optimization or "
                            "alternative routes.")
    if len(state2.dangerous_events) > len(state1.dangerous_events):
        report_lines.append(f"• More ego-collision events during {period2_name}.")
        report_lines.append("• Enhanced safety measures recommended for this "
                            "period.")

    report_text = '\n'.join(report_lines)
    with open(output_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nComparison report saved to: {output_path}")


# =========================
# SCRIPT ENTRYPOINT
# =========================

if __name__ == "__main__":
    # Single video analysis
    analyze_video(
        video_path=INPUT_VIDEO,
        output_dir=OUTPUT_DIR
    )

    # Example for comparing two periods:
    # morning_result = analyze_video("./videos/morning_traffic.mp4",
    #                                "./morning_output")
    # day_result = analyze_video("./videos/daytime_traffic.mp4",
    #                            "./daytime_output")
    # compare_time_periods(morning_result, day_result,
    #                      "Morning", "Daytime",
    #                      "./comparison_report.txt")
