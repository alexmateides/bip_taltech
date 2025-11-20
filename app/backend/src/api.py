from fastapi.routing import APIRouter
from fastapi.exceptions import HTTPException
from src.models import Camera, CamerasResponse, CameraStream

CAMERA_STREAMS={
    "camera1": "56b913ce-3c10-4792-aef0-e5950ab1acd3",
    "camera2": "d1ff2b0c-8750-46c9-a228-68ab69723eeb",
}

router = APIRouter()

@router.get("", response_model=CamerasResponse)
def get_cameras():
    cameras = [Camera(id=cid) for cid in CAMERA_STREAMS.keys()]
    return CamerasResponse(cameras=cameras)


@router.get("/{camera_id}/stream", response_model=CameraStream)
def get_camera_stream(camera_id: str):
    stream = CAMERA_STREAMS.get(camera_id)
    if not stream:
        raise HTTPException(status_code=404, detail="Camera not found")
    return stream



