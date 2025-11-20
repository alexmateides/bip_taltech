from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
import os

router = APIRouter()

# Adjust this if your folder lives somewhere else
VIDEOS_DIR = Path(__file__).parent.parent / "videos"


def open_file_range(file_path: Path, start: int, end: int, chunk_size: int = 1024 * 1024):
    """Generator that reads a file chunk by chunk within [start, end]."""
    with file_path.open("rb") as f:
        f.seek(start)
        bytes_left = end - start + 1
        while bytes_left > 0:
            read_size = min(chunk_size, bytes_left)
            data = f.read(read_size)
            if not data:
                break
            bytes_left -= len(data)
            yield data


@router.get("")
async def get_cameras():
    return os.listdir(VIDEOS_DIR)

@router.get("/{name}")
async def get_video(name: str, range: str | None = Header(default=None)):
    """
    Stream .mp4 video with Range support for seeking.
    Usage (frontend):
      <video src="/videos/my_video.mp4" controls></video>
    """
    # Ensure we always serve .mp4; tweak if you want to allow any extension
    if not name.endswith(".mp4"):
        name = f"{name}.mp4"

    video_path = VIDEOS_DIR / name

    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")

    file_size = video_path.stat().st_size
    content_type = "video/mp4"

    # No Range header -> send full file
    if range is None:
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        }
        return StreamingResponse(
            open_file_range(video_path, 0, file_size - 1),
            media_type=content_type,
            headers=headers,
            status_code=200,
        )

    # Example Range header: "bytes=0-1023"
    try:
        units, _, range_spec = range.partition("=")
        if units.strip().lower() != "bytes":
            raise ValueError("Only 'bytes' unit is supported")

        start_str, _, end_str = range_spec.partition("-")

        if not start_str and not end_str:
            raise ValueError("Invalid Range header")

        if start_str:
            start = int(start_str)
        else:
            # bytes=-500  (last 500 bytes)
            suffix_length = int(end_str)
            start = max(file_size - suffix_length, 0)

        if end_str:
            end = int(end_str)
        else:
            # bytes=500-  (from 500 to end)
            end = file_size - 1

        if start > end or start >= file_size:
            raise ValueError("Invalid byte range")

    except ValueError:
        # Malformed or unsatisfiable range
        raise HTTPException(status_code=416, detail="Invalid Range header")

    chunk_size = (end - start) + 1

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
    }

    return StreamingResponse(
        open_file_range(video_path, start, end),
        media_type=content_type,
        headers=headers,
        status_code=206,  # Partial Content
    )
