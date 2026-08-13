import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dev.db")

connect_args = {"check_same_thread": False, "timeout": 15} if DATABASE_URL.startswith("sqlite") else {"connect_timeout": 10}
pool_kwargs = {} if DATABASE_URL.startswith("sqlite") else {"pool_pre_ping": True, "pool_timeout": 10}

engine = create_engine(DATABASE_URL, connect_args=connect_args, **pool_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency for obtaining database sessions in API requests."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

