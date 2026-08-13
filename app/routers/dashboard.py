import os
import glob
from fastapi import APIRouter, Request, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import ProcessingJob, Upload, ImageHash

router = APIRouter(tags=["Dashboard"])
templates = Jinja2Templates(directory="app/templates")

@router.get("/dashboard", response_class=HTMLResponse)
def get_dashboard(request: Request, db: Session = Depends(get_db)):
    jobs = db.query(ProcessingJob).order_by(ProcessingJob.created_at.desc()).limit(20).all()
    response = templates.TemplateResponse(request=request, name="dashboard.html", context={"request": request, "jobs": jobs})
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@router.get("/api/v1/jobs")
def get_jobs_json(db: Session = Depends(get_db)):
    jobs = db.query(ProcessingJob).order_by(ProcessingJob.created_at.desc()).limit(20).all()
    serialized = []
    for job in jobs:
        upload_info = None
        if job.upload:
            upload_info = {
                "id": job.upload.id,
                "filename": job.upload.filename
            }
        serialized.append({
            "id": job.id,
            "upload_id": job.upload_id,
            "status": job.status,
            "result": job.result,
            "error_message": job.error_message,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "updated_at": job.updated_at.isoformat() if job.updated_at else None,
            "upload": upload_info
        })
    return serialized

@router.post("/api/v1/reset")
def reset_all_pipeline_data(db: Session = Depends(get_db)):
    """Wipes all database records and uploaded files from the running server."""
    try:
        db.query(ProcessingJob).delete()
        db.query(ImageHash).delete()
        db.query(Upload).delete()
        db.commit()
    except Exception:
        db.rollback()

    upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
    if os.path.exists(upload_dir):
        files = glob.glob(os.path.join(upload_dir, "*"))
        for f in files:
            try:
                os.remove(f)
            except Exception:
                pass

    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
