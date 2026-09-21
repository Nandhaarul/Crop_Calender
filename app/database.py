from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH      = os.path.join(BASE_DIR, "crop_calendar.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─────────────────────────────────────────
# CALENDAR TABLE
# ─────────────────────────────────────────
class CalendarRecord(Base):
    __tablename__ = "calendars"

    id            = Column(Integer, primary_key=True, index=True)
    calendar_id   = Column(Integer, unique=True, index=True)
    farmer_name   = Column(String,  nullable=False)
    crop          = Column(String,  nullable=False)
    season        = Column(String,  nullable=False)
    location      = Column(String,  nullable=False)
    soil_type     = Column(String,  nullable=True)
    sowing_date   = Column(String,  nullable=True)
    calendar_json = Column(Text,    nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────
# PROGRESS TABLE (NEW)
# One row per progress update
# ─────────────────────────────────────────
class ProgressRecord(Base):
    __tablename__ = "progress"

    id               = Column(Integer, primary_key=True, index=True)
    calendar_id      = Column(Integer, index=True, nullable=False)
    stage            = Column(String,  nullable=False)
    progress_percent = Column(Integer, nullable=False)
    notes            = Column(Text,    nullable=True)
    logged_at        = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────
# INIT + DEPENDENCY
# ─────────────────────────────────────────
def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()