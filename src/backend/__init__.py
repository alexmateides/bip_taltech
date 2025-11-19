# main.py
import smtplib
import os
import base64
from collections import defaultdict
from typing import List, Dict, Any, Literal, Optional
from email.message import EmailMessage

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr

from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------
VIDEO_DIR = "videos"
CACHE_DIR = "cache"

CHUNK_DURATION = 5.0  # seconds
LANE_WIDTH_M = 3.5
LANE_WIDTH_PX = 600  # adjust per your video
SPEED_HIGH_THRESHOLD = 40.0  # km/h threshold for "high speed"

# Run YOLO only once every N frames
DETECTION_STRIDE_FRAMES = 60

# Static mock location (e.g. Tallinn)
STATIC_LOCATION = {
    "lat": 59.4370,
    "lon": 24.7536,
    "description": "Mock bus route, Tallinn",
}

# -----------------------------
# Pydantic models
# -----------------------------


class Event(BaseModel):
    id: str
    type: Literal["person", "vehicle"]
    timestamp: float


class Chunk(BaseModel):
    id: str
    timestamp_start: float
    timestamp_end: float
    frames: List[str]  # base64-encoded JPEG images
    events: List[Event]
    location: Dict[str, Any]


class CameraStream(BaseModel):
    camera_id: str
    chunks: List[Chunk]
    metrics: Dict[str, Any]


class Camera(BaseModel):
    id: str


class CamerasResponse(BaseModel):
    cameras: List[Camera]


class SendReportRequest(BaseModel):
    eventId: str
    email: EmailStr


class SendReportResponse(BaseModel):
    success: bool
    message: str


# -----------------------------
# Global in-memory storage
# -----------------------------
app = FastAPI(title="Traffic Emergency Monitor POC")

CAMERA_STREAMS: Dict[str, CameraStream] = {}

MOCK_EVENT_DATA: Dict[str, Dict[str, Any]] = {
    "evt1": {
        "id": "evt1",
        "type": "Vehicle collision",
        "timestamp_start": 2.0,
        "timestamp_end": 4.0,
        "location": "Tallinn Old Town",
        "confidence": 0.91,
        "occurred_at": "2024-05-01T10:15:00Z",
        "video_url": "https://example.com/videos/evt1.mp4",
        "description": "Collision involving two vehicles at Tallinn Old Town.",
    },
    "evt2": {
        "id": "evt2",
        "type": "Pedestrian near-miss",
        "timestamp_start": 32.0,
        "timestamp_end": 36.0,
        "location": "Kesklinn",
        "confidence": 0.88,
        "occurred_at": "2024-05-02T08:20:00Z",
        "video_url": "https://example.com/videos/evt2.mp4",
        "description": "Pedestrian detected dangerously close to roadway.",
    },
}

# Email configuration
EMAIL_SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "localhost")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "1025"))
EMAIL_SMTP_USERNAME = os.getenv("EMAIL_SMTP_USERNAME")
EMAIL_SMTP_PASSWORD = os.getenv("EMAIL_SMTP_PASSWORD")
EMAIL_SMTP_USE_TLS = os.getenv("EMAIL_SMTP_USE_TLS", "false").lower() == "true"
EMAIL_FROM_ADDRESS = os.getenv("REPORT_EMAIL_FROM", "stuber0016@gmail.com")
EMAIL_TIMEOUT_SECONDS = int(os.getenv("EMAIL_TIMEOUT_SECONDS", "15"))


# -----------------------------
# Utility functions
# -----------------------------
def encode_frame_to_base64(frame: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


def estimate_speed_kmh(
    prev_gray: Optional[np.ndarray],
    gray: np.ndarray,
    fps: float,
) -> Optional[float]:
    """
    Very rough ego-speed estimate using optical flow in lower half of frame.
    Returns speed in km/h or None if not available.
    """
    if prev_gray is None or fps <= 0:
        return None

    h, w = gray.shape
    roi_prev = prev_gray[h // 2 :, :]  # bottom half
    roi_curr = gray[h // 2 :, :]

    flow = cv2.calcOpticalFlowFarneback(
        roi_prev,
        roi_curr,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    px_per_frame = np.median(mag)
    if px_per_frame <= 0:
        return 0.0

    m_per_pixel = LANE_WIDTH_M / LANE_WIDTH_PX
    m_per_frame = px_per_frame * m_per_pixel
    m_per_s = m_per_frame * fps
    kmh = m_per_s * 3.6
    return float(kmh)


def map_yolo_class_to_type(cls_name: str) -> Optional[str]:
    """
    Map YOLO / COCO class to our event type.
    """
    if cls_name == "person":
        return "person"
    if cls_name in {"car", "truck", "bus", "motorbike"}:
        return "vehicle"
    return None


def is_person_near_road(box, frame_shape) -> bool:
    """
    box: (x1, y1, x2, y2)
    """
    _, h, _ = frame_shape
    x1, y1, x2, y2 = box
    return y2 > (h * 2 / 3)  # bottom third of frame


def is_vehicle_too_close(box, frame_shape) -> bool:
    """
    Simple heuristic: large box height implies closeness.
    """
    _, h, _ = frame_shape
    x1, y1, x2, y2 = box
    height = y2 - y1
    return height > 0.4 * h


# -----------------------------
# Cache helpers
# -----------------------------
def get_cache_path(camera_id: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{camera_id}.json")


def load_cached_stream(camera_id: str, video_path: str) -> Optional[CameraStream]:
    """
    Load cached analysis if it exists AND is newer than the video file.
    """
    cache_path = get_cache_path(camera_id)
    if not os.path.exists(cache_path):
        return None

    # Only use cache if it's newer than the video
    cache_mtime = os.path.getmtime(cache_path)
    video_mtime = os.path.getmtime(video_path)
    if cache_mtime < video_mtime:
        print(f"Cache for camera {camera_id} is older than video, recomputing...")
        return None

    print(f"Loading cached analysis for camera {camera_id} from {cache_path}")
    with open(cache_path, "r", encoding="utf-8") as f:
        data = f.read()

    # Pydantic v2 vs v1 compatibility
    if hasattr(CameraStream, "model_validate_json"):
        # Pydantic v2
        return CameraStream.model_validate_json(data)
    else:
        # Pydantic v1
        return CameraStream.parse_raw(data)


def save_cached_stream(camera_stream: CameraStream) -> None:
    cache_path = get_cache_path(camera_stream.camera_id)
    print(f"Saving analysis for camera {camera_stream.camera_id} to {cache_path}")

    if hasattr(camera_stream, "model_dump_json"):
        # Pydantic v2
        data = camera_stream.model_dump_json()
    else:
        # Pydantic v1
        data = camera_stream.json()

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(data)


# -----------------------------
# Video analysis
# -----------------------------
def analyze_video(video_path: str, camera_id: str, model: YOLO) -> CameraStream:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = float(fps)
    print(f"FPS: {fps}")

    # Run YOLO every N frames
    detection_stride = DETECTION_STRIDE_FRAMES

    frame_idx = 0
    event_counter = 0

    # Metrics
    total_counts = defaultdict(int)  # {class_name: count}
    second_counts = defaultdict(lambda: defaultdict(int))  # {sec: {class_name: count}}
    heatmap_w, heatmap_h = 32, 18
    heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
    speed_profile = []  # list of dicts: {"t": t, "speed_kmh": value}

    # Chunks
    chunks: List[Chunk] = []
    current_chunk_index = 0
    current_chunk_frames: List[str] = []
    current_chunk_events: List[Event] = []

    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        frame_idx += 1

        if frame_idx % DETECTION_STRIDE_FRAMES == 0:
            print(f"Frame: {frame_idx}")

        # Determine chunk index
        chunk_idx = int(timestamp // CHUNK_DURATION)
        if chunk_idx != current_chunk_index and frame_idx > 1:
            # finalize previous chunk
            chunk = Chunk(
                id=f"{camera_id}-chunk-{current_chunk_index}",
                timestamp_start=current_chunk_index * CHUNK_DURATION,
                timestamp_end=(current_chunk_index + 1) * CHUNK_DURATION,
                frames=current_chunk_frames,
                events=current_chunk_events,
                location=STATIC_LOCATION.copy(),
            )
            chunks.append(chunk)

            # start new chunk
            current_chunk_index = chunk_idx
            current_chunk_frames = []
            current_chunk_events = []

        # Convert to gray for speed estimation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        speed_kmh = estimate_speed_kmh(prev_gray, gray, fps)
        if speed_kmh is not None:
            speed_profile.append({"t": timestamp, "speed_kmh": speed_kmh})
        prev_gray = gray

        # Sample frames (1 fps) into chunk
        if int(timestamp) == 0 or (abs(timestamp - round(timestamp)) < 1e-3):
            current_chunk_frames.append(encode_frame_to_base64(frame))

        # Run detection every detection_stride frames
        if frame_idx % detection_stride == 0:
            results = model(frame, verbose=False)[0]

            frame_h, frame_w, _ = frame.shape
            dangerous_speed = (
                speed_kmh is not None and speed_kmh > SPEED_HIGH_THRESHOLD
            )

            for box in results.boxes:
                cls_idx = int(box.cls[0])
                cls_name = results.names.get(cls_idx, "")
                mapped_type = map_yolo_class_to_type(cls_name)
                if mapped_type is None:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Metrics: counts
                total_counts[cls_name] += 1
                sec = int(timestamp)
                second_counts[sec][cls_name] += 1

                # Metrics: heatmap
                grid_x = int((cx / frame_w) * heatmap_w)
                grid_y = int((cy / frame_h) * heatmap_h)
                grid_x = max(0, min(heatmap_w - 1, grid_x))
                grid_y = max(0, min(heatmap_h - 1, grid_y))
                heatmap[grid_y, grid_x] += 1.0

                # Dangerous events
                if dangerous_speed:
                    if mapped_type == "person" and is_person_near_road(
                        (x1, y1, x2, y2), frame.shape
                    ):
                        event_counter += 1
                        event = Event(
                            id=f"{camera_id}-event-{event_counter}",
                            type="person",
                            timestamp=timestamp,
                        )
                        current_chunk_events.append(event)
                    elif mapped_type == "vehicle" and is_vehicle_too_close(
                        (x1, y1, x2, y2), frame.shape
                    ):
                        event_counter += 1
                        event = Event(
                            id=f"{camera_id}-event-{event_counter}",
                            type="vehicle",
                            timestamp=timestamp,
                        )
                        current_chunk_events.append(event)

    cap.release()

    # Final chunk
    if current_chunk_frames or current_chunk_events:
        chunk = Chunk(
            id=f"{camera_id}-chunk-{current_chunk_index}",
            timestamp_start=current_chunk_index * CHUNK_DURATION,
            timestamp_end=(current_chunk_index + 1) * CHUNK_DURATION,
            frames=current_chunk_frames,
            events=current_chunk_events,
            location=STATIC_LOCATION.copy(),
        )
        chunks.append(chunk)

    # Build metrics payload
    object_counts = {cls: int(count) for cls, count in total_counts.items()}

    # time profile: list of {t: sec, counts: {...}}
    time_profile = []
    for sec, counts in sorted(second_counts.items()):
        time_profile.append(
            {
                "t": sec,
                "counts": {cls: int(c) for cls, c in counts.items()},
            }
        )

    # heatmap as nested list
    heatmap_list = heatmap.tolist()

    metrics = {
        "fps": fps,
        "object_counts": object_counts,
        "time_profile": time_profile,
        "heatmap": heatmap_list,
        "speed_profile": speed_profile,
        "lane_width_px": LANE_WIDTH_PX,
        "lane_width_m": LANE_WIDTH_M,
        "speed_threshold_kmh": SPEED_HIGH_THRESHOLD,
    }

    return CameraStream(camera_id=camera_id, chunks=chunks, metrics=metrics)


def compose_event_report(event_data: Dict[str, Any]) -> str:
    lines = [
        f"Event ID: {event_data['id']}",
        f"Type: {event_data['type']}",
        f"Confidence: {event_data['confidence']}",
        f"Window: {event_data['timestamp_start']}s - {event_data['timestamp_end']}s",
        f"Location: {event_data['location']}",
        f"Occurred at: {event_data['occurred_at']}",
        f"Video: {event_data['video_url']}",
        "",
        event_data.get("description", "No description provided."),
    ]
    return "\n".join(lines)


def send_email_via_smtp(recipient: str, subject: str, body: str) -> None:
    message = EmailMessage()
    message["From"] = EMAIL_FROM_ADDRESS
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)

    try:
        with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, timeout=EMAIL_TIMEOUT_SECONDS) as server:
            if EMAIL_SMTP_USE_TLS:
                server.starttls()
            if EMAIL_SMTP_USERNAME and EMAIL_SMTP_PASSWORD:
                server.login(EMAIL_SMTP_USERNAME, EMAIL_SMTP_PASSWORD)
            server.send_message(message)
    except Exception as exc:
        raise RuntimeError(f"Failed to send email: {exc}") from exc


# -----------------------------
# Startup: analyze all videos (with caching)
# -----------------------------
@app.on_event("startup")
def startup_event():
    global CAMERA_STREAMS

    videos = [
        f for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith(".mp4")
    ]
    if not videos:
        print(f"No videos found in {VIDEO_DIR}")
        return

    # Load YOLO model once
    model = YOLO("yolov8n.pt")

    for filename in videos:
        camera_id = os.path.splitext(filename)[0]  # "1.mp4" -> "1"
        video_path = os.path.join(VIDEO_DIR, filename)

        # Try to load cached result first
        cached = load_cached_stream(camera_id, video_path)
        if cached is not None:
            CAMERA_STREAMS[camera_id] = cached
            continue

        # Otherwise compute and cache
        print(f"Analyzing video for camera {camera_id}: {video_path}")
        camera_stream = analyze_video(video_path, camera_id, model)
        CAMERA_STREAMS[camera_id] = camera_stream
        save_cached_stream(camera_stream)

    print("Startup analysis complete.")


# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/api/v1/cameras", response_model=CamerasResponse)
def get_cameras():
    cameras = [Camera(id=cid) for cid in CAMERA_STREAMS.keys()]
    return CamerasResponse(cameras=cameras)


@app.get("/api/v1/cameras/{camera_id}/stream", response_model=CameraStream)
def get_camera_stream(camera_id: str):
    stream = CAMERA_STREAMS.get(camera_id)
    if not stream:
        raise HTTPException(status_code=404, detail="Camera not found")
    return stream


@app.post("/api/v1/reports/email", response_model=SendReportResponse)
def send_report_email(payload: SendReportRequest):
    event_data = MOCK_EVENT_DATA.get(payload.eventId)
    if not event_data:
        raise HTTPException(status_code=404, detail="Event not found")

    subject = f"Traffic event report: {event_data['type']} ({event_data['id']})"
    body = compose_event_report(event_data)

    try:
        send_email_via_smtp(payload.email, subject, body)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return SendReportResponse(success=True, message="Report sent")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
