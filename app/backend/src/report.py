from typing import List, Dict, Any
import os
import smtplib
import json
from pathlib import Path
from fastapi.exceptions import HTTPException
from fastapi.routing import APIRouter
from src.models import SendReportResponse, SendReportRequest
from email.message import EmailMessage

router = APIRouter()

VIDEOS_DIR = Path(__file__).parent.parent / "videos"

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
EMAIL_SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
EMAIL_SMTP_USERNAME = os.getenv("EMAIL_SMTP_USERNAME")
EMAIL_SMTP_PASSWORD = os.getenv("EMAIL_SMTP_PASSWORD")
EMAIL_SMTP_USE_TLS = os.getenv("EMAIL_SMTP_USE_TLS", "true").lower() == "true"
EMAIL_FROM_ADDRESS = os.getenv("REPORT_EMAIL_FROM", "allpueli.ai@gmail.com")
EMAIL_TIMEOUT_SECONDS = int(os.getenv("EMAIL_TIMEOUT_SECONDS", "15"))


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
            server.ehlo()
            if EMAIL_SMTP_USE_TLS:
                server.starttls()
                server.ehlo()

            auth_supported = "auth" in (server.esmtp_features or {})
            if EMAIL_SMTP_USERNAME and EMAIL_SMTP_PASSWORD:
                if auth_supported:
                    server.login(EMAIL_SMTP_USERNAME, EMAIL_SMTP_PASSWORD)
                else:
                    raise RuntimeError("SMTP server does not support AUTH; unset credentials to send without login")

            server.send_message(message)
    except Exception as exc:
        raise RuntimeError(f"Failed to send email: {exc}") from exc



@router.post("/reports/email", response_model=SendReportResponse)
def send_report_email(payload: SendReportRequest):
    events_path = VIDEOS_DIR / payload.camera_id / "events.json"
    if not events_path.is_file():
        raise HTTPException(status_code=404, detail="Events file not found")

    with events_path.open("r") as f:
        events_blob = json.load(f)

    events = events_blob.get("events", [])
    event_data = next(
        (
            evt
            for evt in events
            if evt.get("id") == payload.event_id
            and evt.get("camera_id") == payload.camera_id
        ),
        None,
    )
    if event_data is None:
        raise HTTPException(status_code=404, detail="Event not found")

    subject = f"Traffic event report: {event_data['type']} ({event_data['id']})"
    body = compose_event_report(event_data)

    try:
        send_email_via_smtp(payload.email, subject, body)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return SendReportResponse(success=True, message="Report sent")
