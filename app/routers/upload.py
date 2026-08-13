import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.models import Upload, ProcessingJob
from app.queue import image_queue

router = APIRouter(prefix="/api/v1", tags=["Upload"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")


class UploadResponse(BaseModel):
    processing_id: str
    upload_id: str
    filename: str
    status: str
    message: str


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate allowed image extensions
    allowed_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types are: {', '.join(allowed_extensions)}"
        )

    # Ensure target upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    upload_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    # Sanitize filename & form local file path
    safe_filename = os.path.basename(file.filename)
    stored_filename = f"{upload_id}_{safe_filename}"
    file_path = os.path.join(UPLOAD_DIR, stored_filename)

    # Save file content locally
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}",
        )

    file_size = os.path.getsize(file_path)
    content_type = file.content_type or "application/octet-stream"

    # Create Database Records
    db_upload = Upload(
        id=upload_id,
        filename=safe_filename,
        file_path=file_path,
        content_type=content_type,
        file_size=file_size,
    )
    db.add(db_upload)

    db_job = ProcessingJob(
        id=job_id,
        upload_id=upload_id,
        status="pending",
    )
    db.add(db_job)
    db.commit()

    # Queue async processing job via abstract QueueService
    try:
        image_queue.enqueue_job(
            "worker.worker.process_image_job",
            job_id,
        )
    except Exception as e:
        # Update job status if enqueuing fails
        db_job.status = "failed"
        db_job.error_message = f"Failed to enqueue task: {str(e)}"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image uploaded but failed to queue for analysis: {str(e)}",
        )

    return UploadResponse(
        processing_id=job_id,
        upload_id=upload_id,
        filename=safe_filename,
        status="pending",
        message="Image uploaded and queued successfully for analysis",
    )
