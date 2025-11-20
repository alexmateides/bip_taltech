# main.py
import os
import base64
from collections import defaultdict
from typing import List, Dict, Any, Literal, Optional
from email.message import EmailMessage
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr

from ultralytics import YOLO

from src.camera import router as camera_router
from src.report import router as report_router

app = FastAPI(title="Traffic Emergency Monitor POC")
app.include_router(camera_router, prefix="/api/v1/cameras")
app.include_router(report_router, prefix="/api/v1/reports")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allows all origins
    allow_credentials=True,
    allow_methods=['*'],  # Allows all methods
    allow_headers=['*'],  # Allows all headers
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
