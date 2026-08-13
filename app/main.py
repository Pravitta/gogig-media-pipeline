import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.database import engine, Base
from app.routers import upload, results, dashboard

# Create database tables automatically on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Async Image Processing API",
    description="FastAPI service for async image uploading, queuing, and result retrieval",
    version="1.0.0",
)

# Mount static directory for serving images to the dashboard
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

# Include API Routers
app.include_router(upload.router)
app.include_router(results.router)
app.include_router(dashboard.router)


from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/dashboard")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Async Image Processing API",
        "version": "1.0.0",
    }
