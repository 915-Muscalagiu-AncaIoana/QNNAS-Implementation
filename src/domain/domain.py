from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
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

    architectures = relationship("QuantumArchitecture", back_populates="session")

class QuantumArchitecture(Base):
    __tablename__ = "quantum_architectures"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("training_sessions.id"), nullable=False)
    epoch = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=False)
    loss = Column(Float, nullable=False)
    num_layers = Column(Integer, nullable=False)
    gate_sequence = Column(JSON, nullable=False)
    circuit_diagram_path = Column(String, nullable=False)
    loss_plot_path = Column(String, nullable=False)

    session = relationship("TrainingSession", back_populates="architectures")
