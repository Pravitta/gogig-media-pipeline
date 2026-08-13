import os
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageFilter, ImageDraw
from app.main import app
from app.database import get_db, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from worker.checks.blur import analyze_blur
from worker.checks.duplicate import analyze_duplicate
from app.models import ImageHash

# Use a local test SQLite database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test.db"):
        os.remove("./test.db")

@pytest.fixture
def db_session():
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()

@pytest.fixture(scope="module")
def sample_images(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("data")
    
    # 1. Sharp Image (White background with black grid lines)
    sharp_img = Image.new("RGB", (300, 300), color="white")
    draw = ImageDraw.Draw(sharp_img)
    for i in range(0, 300, 15):
        draw.line([(i, 0), (i, 300)], fill="black", width=2)
    sharp_path = tmp_dir / "sharp.png"
    sharp_img.save(sharp_path)
    
    # 2. Blurry Image (Apply strong Gaussian Blur)
    blurry_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=15))
    blurry_path = tmp_dir / "blurry.png"
    blurry_img.save(blurry_path)
    
    # 3. Duplicate Image (Save exact copy of sharp image)
    dup_path = tmp_dir / "duplicate.png"
    sharp_img.save(dup_path)
    
    return {
        "sharp": str(sharp_path),
        "blurry": str(blurry_path),
        "duplicate": str(dup_path)
    }

# --- API Endpoint Tests ---

def test_upload_valid_image(client):
    file_data = io.BytesIO()
    Image.new("RGB", (100, 100), color="red").save(file_data, format="PNG")
    file_data.seek(0)
    
    response = client.post(
        "/api/v1/upload",
        files={"file": ("test.png", file_data, "image/png")}
    )
    assert response.status_code == 202
    json_data = response.json()
    assert "processing_id" in json_data
    assert "upload_id" in json_data
    assert json_data["status"] == "pending"

def test_upload_invalid_file_type(client):
    file_data = io.BytesIO(b"not an image file")
    response = client.post(
        "/api/v1/upload",
        files={"file": ("test.txt", file_data, "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

def test_status_unknown_job_id(client):
    response = client.get("/api/v1/results/non-existent-job-id")
    assert response.status_code == 404

# --- Worker Quality Checks Unit Tests ---

def test_blur_detection(sample_images):
    # Sharp image should score high
    sharp_res = analyze_blur(sample_images["sharp"])
    # Blurry image should score low
    blurry_res = analyze_blur(sample_images["blurry"])
    
    assert sharp_res["score"] > blurry_res["score"]
    assert blurry_res["verdict"] == "needs_review"
    assert sharp_res["verdict"] == "clean"

def test_duplicate_detection(db_session, sample_images):
    # Insert first image's hash into the database
    import imagehash
    sharp_hash = str(imagehash.phash(Image.open(sample_images["sharp"])))
    
    db_hash = ImageHash(upload_id="original-upload-id", phash=sharp_hash)
    db_session.add(db_hash)
    db_session.commit()
    
    # Process the duplicate upload. It should match the original (distance 0 <= threshold 5)
    dup_res = analyze_duplicate(sample_images["duplicate"], db_session, "new-upload-id")
    
    assert dup_res["verdict"] == "rejected"
    assert "hash_distance=0" in dup_res["signal"]
    assert dup_res["match_id"] == "original-upload-id"


def test_dimensions_check(sample_images):
    from worker.checks.dimensions import analyze_dimensions
    res = analyze_dimensions(sample_images["sharp"])
    assert res["name"] == "dimensions"
    assert res["score"] > 0.0
    assert "mp=" in res["signal"]


def test_editing_check(sample_images):
    from worker.checks.editing import analyze_editing
    res = analyze_editing(sample_images["sharp"])
    assert res["name"] == "editing"
    assert "signal" in res

