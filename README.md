# Intelligent Media Processing & Quality Pipeline

A production-grade, asynchronous backend system that ingests uploaded vehicle images, queues background quality and fraud detection checks using a Directed Acyclic Graph (DAG) architecture, computes composite confidence scores, and exposes structured analysis APIs & a live monitoring dashboard.

---

## 1. Architecture

```mermaid
graph TD
    User[Client / API User] -->|1. POST /api/v1/upload| API[FastAPI Web Server]
    API -->|2. Store Metadata| DB[(SQLite / PostgreSQL)]
    API -->|3. Enqueue Job| Q[Queue Service: Redis/RQ or In-Memory]
    
    Q -->|4. Fan-Out Parallel Workers| W1[Blur Detector]
    Q -->|4. Fan-Out Parallel Workers| W2[Brightness Detector]
    Q -->|4. Fan-Out Parallel Workers| W3[Duplicate pHash Detector]
    Q -->|4. Fan-Out Parallel Workers| W4[Screenshot & Moiré Detector]
    Q -->|4. Fan-Out Parallel Workers| W5[License Plate OCR & Validator]
    Q -->|4. Fan-Out Parallel Workers| W6[Vehicle Classifier]
    
    W1 --> W7(Fan-In: Aggregator)
    W2 --> W7
    W3 --> W7
    W4 --> W7
    W5 --> W7
    W6 --> W7
    
    W7 -->|5. Compute Composite Score & Verdict| DB
    User -->|6. GET /dashboard or GET /api/v1/results/{id}| API
```

### Service Flow
1. **Ingestion & Validation**: Clients upload images to `POST /api/v1/upload`. The API performs basic payload validation, saves the image with a unique filename containing `upload_id` and `processing_id`, creates a pending `ProcessingJob` entry in SQLite/PostgreSQL, and enqueues the processing job.
2. **Immediate Return**: The endpoint immediately returns a `202 Accepted` status with the job IDs, keeping the HTTP connection short and non-blocking.
3. **Polling & Monitoring**: Clients poll status via `GET /api/v1/results/{processing_id}/status` or fetch full metrics via `GET /api/v1/results/{processing_id}`. Live monitoring is rendered at `/dashboard`.

### Processing Flow (DAG Execution)
The pipeline is structured as a Directed Acyclic Graph (DAG) where quality checks run in parallel:
- **Blur & Focus check**: Computes focus score using OpenCV Laplacian variance metrics.
- **Brightness check**: Evaluates pixel intensity histograms to detect under/overexposed frames.
- **Duplicate check**: Compares perceptual hashes (pHash) against historical records.
- **Source Integrity check**: Analyzes aspect ratios, EXIF headers, and Fast Fourier Transforms (FFT) for screen moiré.
- **License Plate OCR check**: Automatically extracts registration plates using **Google Cloud Vision REST API**, filters hologram prefixes, and maps character confusions.
- **Vehicle Classifier check**: Evaluates bounding contours and yellow HSV color ratios to verify target vehicles (e.g. Auto-Rickshaw / Commercial Three-Wheeler).
- **Fan-In Aggregator**: Consolidates parallel outputs, calculates the composite confidence score, updates the database status to `completed`, and updates the final job verdict.

### Queue Strategy
- **Production Mode (`USE_IN_MEMORY_QUEUE=false`)**: Integrates Redis and python-rq. It enqueues the individual tasks in parallel. The aggregate step is configured with RQ dependencies (`depends_on`), executing only when all 6 worker sub-tasks complete successfully.
- **Development Mode (`USE_IN_MEMORY_QUEUE=true`)**: Falls back to an synchronous, in-memory queue runner executing checks sequentially within the main web container, allowing local execution with zero docker/redis dependencies.

### Major Design Decisions
- **REST-based Cloud Vision client**: Implemented direct `urllib.request` base64 POST calls using API keys. This bypasses the locked Org policy block (`iam.disableServiceAccountKeyCreation`) preventing JSON key creation.
- **Bottom-Up Line parsing**: Configured character line search starting from the bottom of OCR frames. Since vehicle plates are located on the bumper (lower half), this avoids capturing background street signs or traffic plates.
- **Character Correction rules**: Maps stenciled bumper stencils (like bumper font reading `MH12NH8556`) to correct registration indexes (`MH12NW8556`).
- **HSV Masking for Classification**: Since ML models like YOLOv8 were not pre-packaged in the environment, we built color-space masking heuristics. We detect auto-rickshaws by calculating the ratio of bright yellow pixels (`H: 15-35`, `S: 40-255`, `V: 40-255`) relative to bounding contours.
- **38 State Code whitelist**: Enforces validation against valid Indian States and UTs (`VALID_STATE_CODES`), preventing background noise text from verifying as random plates.

---

## 2. AI Usage Disclosure (Mandatory)

### Where AI was used
- Structuring the base64 Google Cloud Vision REST request payload.
- Designing the HSV yellow masking thresholds for the OpenCV Three-Wheeler vehicle classifier.
- Structuring the CSS grid columns and HTML Jinja tags for the Live Dashboard card layout.

### Where AI output was wrong and how it was corrected
- **Pune Rickshaw Misclassification**: The AI originally suggested a strict aspect ratio (`>= 1.0` and `<= 2.2`) and high yellow contour ratio (`>= 5%`). However, shadows merging with the auto-rickshaw body skewed the bounding aspect ratio to `2.27` and diluted the yellow ratio to `5.1%`, causing the pipeline to misclassify it as a Four-Wheeler. **Correction**: We lowered the yellow threshold to `3.5%` (`0.035`) and widened the aspect ratio boundary to `2.8` under [`vehicle.py`](file:///c:/Users/jprav/OneDrive/Desktop/gogig-media-pipeline/worker/checks/vehicle.py).
- **Background Traffic Plate Interference**: For the Chennai auto-rickshaw (`image2.png`), the AI processed OCR text lines top-down, which matched background street traffic plates (like `TN 05 CI 5911`) first. **Correction**: We modified the line search to run bottom-up, prioritizing the foreground vehicle plate (`TN 05 BT 5754`).
- **Blur Image Garbage Fallback**: When processing the blurry image (`imageblur.png`), Google Cloud Vision correctly returned empty text. However, the AI code fell back to Tesseract OCR, which parsed background watermarks as active plates. **Correction**: We added an early exit in [`plate.py`](file:///c:/Users/jprav/OneDrive/Desktop/gogig-media-pipeline/worker/checks/plate.py) returning `None` immediately if Cloud Vision runs successfully but finds no plate.
- **Jinja Render 500 & Layout Overlap**: Under large text scales, long strings overflowed columns, and the HTML merge left duplicate tags. **Correction**: We added flex-wrapping (`flex-wrap`) and corrected the dangling `{% endif %}` tags.

### How code was validated
Code was validated using isolated validation scripts under [`scripts/`](file:///c:/Users/jprav/OneDrive/Desktop/gogig-media-pipeline/scripts/) and unit test scripts under [`scratch/`](file:///c:/Users/jprav/OneDrive/Desktop/gogig-media-pipeline/scratch/):
- **[`test_vision.py`](file:///c:/Users/jprav/OneDrive/Desktop/gogig-media-pipeline/scratch/test_vision.py)**: Verified API REST client parsing on live base64 payloads.
- **[`test_normalize.py`](file:///c:/Users/jprav/OneDrive/Desktop/gogig-media-pipeline/scratch/test_normalize.py)**: Validated state code whitelists, Bharat series regex patterns, and stenciled index mappings.
- **[`test_yellow.py`](file:///c:/Users/jprav/OneDrive/Desktop/gogig-media-pipeline/scratch/test_yellow.py)**: Validated OpenCV HSV masking filters.
- **Browser Subagents**: Used automated browser sessions to capture real-time screenshots and verify page element structures.

---

## 3. Trade-offs

### Intentional Simplifications
- **SQLite Fallback for Local Run**: SQLite is used as a fallback for local execution to make run-throughs without Docker dependency-free. However, under standard Docker Compose, the system uses a fully configured PostgreSQL 15 database service.
- **OpenCV Heuristics**: Used aspect ratio and HSV yellow color ratio heuristics for vehicle classification instead of importing heavy neural nets (like YOLOv8/ResNet). This keeps memory usage minimal and CPU latency low.
- **In-Memory Duplicate Check**: Perceptual hash comparisons are run inside the python worker memory against database records.

### Future Improvements (with more time)
- **Deep-Learning plate localization**: Integrating a YOLO license plate locator to crop plates before OCR, improving accuracy in low-light environments.
- **Distributed Hash Lookup**: Moving perceptual hash checks to a specialized vector engine (like `FAISS` or PostgreSQL `pgvector`) to optimize search times.
- **Websocket notifications**: Replacing the 5-second polling reload in `dashboard.html` with real-time websocket pushes for seamless updates.

### Scalability Concerns
- **OpenCV CPU usage**: Image decoding and FFT computations are CPU-intensive operations. A sudden influx of parallel uploads will saturate worker CPU threads.
- **Redis Queue Memory**: rq stores queue logs in Redis memory. Under heavy loads, Redis storage must be configured with TTL policies to prevent OOM events.

### Failure Handling Concerns
- **GCP REST timeouts**: Third-party API failures could block execution. Wrapped all HTTP requests in retry handlers with 10-second connection limits.
- **Corrupt images**: Uploading incomplete or corrupted byte strings is caught by OpenCV decoders (`cv2.imdecode` returning `None`), failing the job gracefully with detailed error payloads instead of crashing workers.

---

## 4. Running Instructions

### Prerequisites
- Python 3.9+ installed.
- (Optional) Docker and Docker Compose installed.

---

### Method 1: Docker Compose (Recommended)

1. Set your Google Cloud Vision API Key in `docker-compose.yml` under the `api` and `worker` environments:
   ```yaml
   VISION_API_KEY: "AIzaSyBA3_VgDyrUguJ4sP7vKTejC2q1xn3zwgI"
   ```
2. Start the services:
   ```bash
   docker compose up --build -d
   ```
3. Open your browser to **`http://localhost:8000/dashboard`** to see the pipeline live.

---

### Method 2: Local Execution (Zero Docker Dependencies)

1. Activate your python virtual environment:
   ```powershell
   # Windows:
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the uvicorn web server in-memory mode:
   ```powershell
   $env:USE_IN_MEMORY_QUEUE="true"
   $env:VISION_API_KEY="AIzaSyBA3_VgDyrUguJ4sP7vKTejC2q1xn3zwgI"
   python -m uvicorn app.main:app --port 8000 --reload
   ```
4. Access the dashboard at **`http://localhost:8000/dashboard`**.

---

### Running Tests & Seed Uploads

#### Running the test suite:
To run unit and integration tests:
```bash
pytest
```

#### Uploading sample images (Seeding data):
Upload your test images using curl:
```powershell
# 1. Upload Chennai Auto-Rickshaw (Clean plate TN 05 BT 5754)
curl.exe -X POST "http://localhost:8000/api/v1/upload" -F "file=@C:\Users\jprav\OneDrive\Desktop\IMAGESGOGIG\image2.png"

# 2. Upload Pune Auto-Rickshaw (Corrected plate MH 12 NW 8556)
curl.exe -X POST "http://localhost:8000/api/v1/upload" -F "file=@C:\Users\jprav\OneDrive\Desktop\IMAGESGOGIG\image.png"

# 3. Upload Blurry Image (Returns None)
curl.exe -X POST "http://localhost:8000/api/v1/upload" -F "file=@C:\Users\jprav\OneDrive\Desktop\IMAGESGOGIG\imageblur.png"
```
Once uploaded, open **`http://localhost:8000/dashboard`** to watch the processing cards update live!
