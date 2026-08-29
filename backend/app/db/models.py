"""
SQLite database setup via SQLAlchemy. Stores every analysis result so the
history endpoint can retrieve past analyses.
"""

import os
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "database", "app.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# check_same_thread=False is required for SQLite when used with FastAPI's
# threaded request handling; standard, documented pattern for this combo.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    stored_image_path = Column(String, nullable=False)
    quality_score = Column(Integer, nullable=False)
    quality_label = Column(String, nullable=False)
    issues = Column(JSON, nullable=False)          # list of {type, severity, confidence}
    image_stats = Column(JSON, nullable=False)      # raw feature values
    model_breakdown = Column(JSON, nullable=False)  # per-model probs, blend weights
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()