from domain.domain import TrainingStatus
from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime

class TrainingSessionDTO(BaseModel):
    id: int
    dataset: str
    gates: Any
    discount_rate: float
    learning_rate: float
    max_architecture_length: int
    autoencoder_path: Optional[str]
    status: TrainingStatus
    start_time: datetime
    end_time: Optional[datetime]

    model_config = {
        "from_attributes": True
    }
