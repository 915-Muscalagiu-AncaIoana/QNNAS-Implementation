from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Enum
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum

Base = declarative_base()

class TrainingStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"

class TrainingSession(Base):
    __tablename__ = "training_sessions"

    id = Column(Integer, primary_key=True)
    dataset = Column(String, nullable=False)
    gates = Column(JSON, nullable=False)
    discount_rate = Column(Float, nullable=False)
    learning_rate = Column(Float, nullable=False)
    max_architecture_length = Column(Integer, nullable=False)
    autoencoder_path = Column(String, nullable=True)
    status = Column(Enum(TrainingStatus), default=TrainingStatus.pending)
    start_time = Column(DateTime, default=datetime)
    end_time = Column(DateTime, nullable=True)