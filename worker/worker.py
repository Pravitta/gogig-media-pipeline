import time
import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from rq import Worker, Queue, get_current_job

from app.database import SessionLocal
from app.models import ProcessingJob, Upload, ImageHash
import os
from redis import Redis
redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_conn = Redis.from_url(redis_url)
from worker.checks.screenshot import analyze_screenshot
from worker.checks.duplicate import analyze_duplicate
from worker.checks.plate import analyze_plate
from worker.checks.brightness import analyze_brightness
from worker.checks.blur import analyze_blur
from worker.checks.dimensions import analyze_dimensions
from worker.checks.editing import analyze_editing
from worker.checks.vehicle import analyze_vehicle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("worker")

def execute_check_with_retry(check_name: str, check_fn, *args, max_retries: int = 3, **kwargs):
    job_id = "unknown"
    current_rq_job = get_current_job()
    if current_rq_job:
        job_id = current_rq_job.id
        
    start_time = time.time()
    
    for attempt in range(1, max_retries + 1):
        try:
            result = check_fn(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(
                f"STRUCTURING_LOG | job_id={job_id} | check={check_name} | "
                f"attempt={attempt} | duration={duration:.4f}s | outcome=success"
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.warning(
                f"STRUCTURING_LOG | job_id={job_id} | check={check_name} | "
                f"attempt={attempt} | duration={duration:.4f}s | outcome=failure | error={str(e)}"
            )
            if attempt == max_retries:
                logger.error(
                    f"STRUCTURING_LOG | job_id={job_id} | check={check_name} | "
                    f"attempts_exhausted | duration={duration:.4f}s | outcome=failed"
                )
                return {
                    "name": check_name,
                    "score": 0.0,
                    "signal": f"failed: {str(e)}",
                    "verdict": "unknown",
                    "confidence": 0.0
                }
            time.sleep(0.5)

import importlib

# The worker functions for DAG
def run_blur(image_path: str):
    import worker.checks.blur as blur_mod
    importlib.reload(blur_mod)
    return execute_check_with_retry("blur", blur_mod.analyze_blur, image_path)

def run_duplicate(image_path: str, upload_id: str):
    import worker.checks.duplicate as dup_mod
    importlib.reload(dup_mod)
    def run_dup_attempt():
        db: Session = SessionLocal()
        try:
            return dup_mod.analyze_duplicate(image_path, db, upload_id)
        finally:
            db.close()
    return execute_check_with_retry("duplicate", run_dup_attempt)

def run_screenshot(image_path: str):
    import worker.checks.screenshot as screen_mod
    importlib.reload(screen_mod)
    return execute_check_with_retry("screenshot", screen_mod.analyze_screenshot, image_path)

def run_plate(image_path: str):
    import worker.checks.plate as plate_mod
    importlib.reload(plate_mod)
    return execute_check_with_retry("ocr_plate", plate_mod.analyze_plate, image_path)

def run_brightness(image_path: str):
    import worker.checks.brightness as bright_mod
    importlib.reload(bright_mod)
    return execute_check_with_retry("brightness", bright_mod.analyze_brightness, image_path)

def run_dimensions(image_path: str):
    import worker.checks.dimensions as dim_mod
    importlib.reload(dim_mod)
    return execute_check_with_retry("dimensions", dim_mod.analyze_dimensions, image_path)

def run_editing(image_path: str):
    import worker.checks.editing as edit_mod
    importlib.reload(edit_mod)
    return execute_check_with_retry("editing", edit_mod.analyze_editing, image_path)

def run_vehicle(image_path: str):
    import worker.checks.vehicle as veh_mod
    importlib.reload(veh_mod)
    return execute_check_with_retry("vehicle", veh_mod.analyze_vehicle, image_path)

def aggregate_results(job_id: str, upload_id: str):
    logger.info(f"Aggregating results for job {job_id}")
    db: Session = SessionLocal()
    try:
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if not job:
            return

        current_rq_job = get_current_job()
        
        results = []
        weight_sum = 0
        weighted_score_sum = 0
        verdict = "clean"
        current_hash = None

        # Defined check weights
        WEIGHTS = {
            "duplicate": 0.25,
            "ocr_plate": 0.20,
            "screenshot": 0.15,
            "editing": 0.15,
            "blur": 0.15,
            "brightness": 0.05,
            "dimensions": 0.05
        }

        if current_rq_job and current_rq_job.dependency_ids:
            queue = Queue("image_processing", connection=redis_conn)
            for dep_id in current_rq_job.dependency_ids:
                dep_job = queue.fetch_job(dep_id)
                if dep_job and dep_job.result:
                    res = dep_job.result
                    results.append(res)
                    
                    name = res.get("name")
                    if name == "duplicate" and res.get("current_hash"):
                        current_hash = res["current_hash"]
                    
                    weight = WEIGHTS.get(name, 0.0)
                    score = res.get("score", 0.0)
                    
                    weight_sum += weight
                    weighted_score_sum += score * weight

                    # Verdict cascade logic
                    check_verdict = res.get("verdict", "clean")
                    if check_verdict == "rejected":
                        verdict = "rejected"
                    elif check_verdict == "needs_review" and verdict != "rejected":
                        verdict = "needs_review"

        composite_score = weighted_score_sum / weight_sum if weight_sum > 0 else 0.0

        final_report = {
            "verdict": verdict,
            "score": round(composite_score, 2),
            "checks": results
        }

        job.status = "completed"
        job.result = final_report
        job.updated_at = datetime.now(timezone.utc)
        
        # Save hash for duplicate detection on future uploads
        if current_hash:
            # Check if hash already exists just to be safe
            existing = db.query(ImageHash).filter(ImageHash.upload_id == upload_id).first()
            if not existing:
                new_hash = ImageHash(upload_id=upload_id, phash=current_hash)
                db.add(new_hash)

        db.commit()
    except Exception as e:
        logger.exception(f"Error in aggregate_results {job_id}: {str(e)}")
        db.rollback()
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            job.updated_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


def process_image_job(job_id: str):
    """
    Entry point. Updates job status to processing and fans out tasks.
    Supports both RQ Redis queuing and standalone in-memory execution.
    """
    logger.info(f"Dispatching DAG for job_id: {job_id}")
    db: Session = SessionLocal()

    try:
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if not job:
            return

        upload = db.query(Upload).filter(Upload.id == job.upload_id).first()
        image_path = upload.file_path

        job.status = "processing"
        job.updated_at = datetime.now(timezone.utc)
        db.commit()

        use_in_memory = os.getenv("USE_IN_MEMORY_QUEUE", "true").lower() == "true"
        if not use_in_memory:
            try:
                redis_conn.ping()
            except Exception:
                logger.warning("Redis is unreachable. Falling back to In-Memory Queue execution...")
                use_in_memory = True

        if use_in_memory:
            logger.info("Executing quality checks via In-Memory Queue...")
            res_blur = run_blur(image_path)
            res_dup = run_duplicate(image_path, upload.id)
            res_screen = run_screenshot(image_path)
            res_plate = run_plate(image_path)
            res_brightness = run_brightness(image_path)
            res_dimensions = run_dimensions(image_path)
            res_editing = run_editing(image_path)
            res_vehicle = run_vehicle(image_path)

            # Manual aggregation for in-memory execution
            results = [res_blur, res_dup, res_screen, res_plate, res_brightness, res_dimensions, res_editing, res_vehicle]
            WEIGHTS = {
                "duplicate": 0.20,
                "vehicle": 0.15,
                "ocr_plate": 0.15,
                "screenshot": 0.15,
                "editing": 0.15,
                "blur": 0.10,
                "brightness": 0.05,
                "dimensions": 0.05
            }
            weight_sum = 0
            weighted_score_sum = 0
            verdict = "clean"
            current_hash = None

            for res in results:
                name = res.get("name")
                if name == "duplicate" and res.get("current_hash"):
                    current_hash = res["current_hash"]
                weight = WEIGHTS.get(name, 0.0)
                score = res.get("score", 0.0)
                weight_sum += weight
                weighted_score_sum += score * weight
                check_verdict = res.get("verdict", "clean")
                if check_verdict == "rejected":
                    verdict = "rejected"
                elif check_verdict == "needs_review" and verdict != "rejected":
                    verdict = "needs_review"

            composite_score = weighted_score_sum / weight_sum if weight_sum > 0 else 0.0
            job.status = "completed"
            job.result = {
                "verdict": verdict,
                "score": round(composite_score, 2),
                "checks": results
            }
            job.updated_at = datetime.now(timezone.utc)
            if current_hash:
                existing = db.query(ImageHash).filter(ImageHash.upload_id == upload.id).first()
                if not existing:
                    db.add(ImageHash(upload_id=upload.id, phash=current_hash))
            db.commit()
            return

        # Default: RQ Redis Queue Execution
        queue = Queue("image_processing", connection=redis_conn)
        j_blur = queue.enqueue(run_blur, image_path)
        j_dup = queue.enqueue(run_duplicate, image_path, upload.id)
        j_screen = queue.enqueue(run_screenshot, image_path)
        j_plate = queue.enqueue(run_plate, image_path, job_timeout=600)
        j_brightness = queue.enqueue(run_brightness, image_path)
        j_dimensions = queue.enqueue(run_dimensions, image_path)
        j_editing = queue.enqueue(run_editing, image_path)
        j_vehicle = queue.enqueue(run_vehicle, image_path)

        queue.enqueue(
            aggregate_results,
            job_id,
            upload.id,
            depends_on=[j_blur, j_dup, j_screen, j_plate, j_brightness, j_dimensions, j_editing, j_vehicle]
        )

    except Exception as e:
        logger.exception(f"Error dispatching job {job_id}: {str(e)}")
        db.rollback()
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            job.updated_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    logger.info("Starting RQ Worker listening on queue 'image_processing'...")
    queue = Queue("image_processing", connection=redis_conn)
    worker = Worker([queue], connection=redis_conn)
    worker.work()
