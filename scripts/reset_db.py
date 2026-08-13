"""
Database Reset Script: reset_db.py
Clears all jobs, uploads, and image hashes from the database and deletes uploads directory files.
"""

import os
import sys
import glob

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Upload, ProcessingJob, ImageHash

def reset_all_data():
    db_urls = ["sqlite:///./dev.db"]
    if os.getenv("DATABASE_URL"):
        db_urls.append(os.getenv("DATABASE_URL"))

    for url in set(db_urls):
        try:
            print(f"Clearing database tables for {url}...")
            connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
            eng = create_engine(url, connect_args=connect_args)
            session_factory = sessionmaker(bind=eng)
            db = session_factory()
            db.query(ProcessingJob).delete()
            db.query(ImageHash).delete()
            db.query(Upload).delete()
            db.commit()
            db.close()
            print(f"Database {url} cleared successfully!")
        except Exception as e:
            print(f"Note for {url}: {e}")

    # Clear uploads folder
    upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
    if os.path.exists(upload_dir):
        files = glob.glob(os.path.join(upload_dir, "*"))
        for f in files:
            try:
                os.remove(f)
            except Exception:
                pass
        print("Uploads folder emptied successfully!")

if __name__ == "__main__":
    reset_all_data()
