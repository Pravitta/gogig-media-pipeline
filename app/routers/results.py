from typing import Optional, Any, Dict
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.models import ProcessingJob

router = APIRouter(prefix="/api/v1", tags=["Results"])


class ProcessingJobResponse(BaseModel):
    processing_id: str
    upload_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class StatusResponse(BaseModel):
    status: str

class FailureReasonResponse(BaseModel):
    error_message: Optional[str]


@router.get("/results/{processing_id}", response_model=ProcessingJobResponse)
def get_job_results(processing_id: str, db: Session = Depends(get_db)):
    job = db.query(ProcessingJob).filter(ProcessingJob.id == processing_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Processing job '{processing_id}' not found",
        )

    return ProcessingJobResponse(
        processing_id=job.id,
        upload_id=job.upload_id,
        status=job.status,
        result=job.result,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )

@router.get("/results/{processing_id}/status", response_model=StatusResponse)
def get_job_status(processing_id: str, db: Session = Depends(get_db)):
    job = db.query(ProcessingJob).filter(ProcessingJob.id == processing_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Processing job '{processing_id}' not found",
        )

    return StatusResponse(status=job.status)

@router.get("/results/{processing_id}/failure-reason", response_model=FailureReasonResponse)
def get_job_failure_reason(processing_id: str, db: Session = Depends(get_db)):
    job = db.query(ProcessingJob).filter(ProcessingJob.id == processing_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Processing job '{processing_id}' not found",
        )
    if job.status != "failed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Processing job '{processing_id}' has not failed.",
        )

    return FailureReasonResponse(error_message=job.error_message)
