from pydantic import BaseModel, EmailStr
from typing import Dict, List, Any

class Camera(BaseModel):
    id: str


class CamerasResponse(BaseModel):
    cameras: List[Camera]

class SendReportResponse(BaseModel):
    success: bool
    message: str

class SendReportRequest(BaseModel):
    camera_id: str
    event_id: str
    email: EmailStr
