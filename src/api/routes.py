import os
import subprocess

from litestar import post, get, Router
from litestar.di import Provide
from litestar.exceptions import HTTPException
from litestar.params import Parameter
from litestar.status_codes import HTTP_404_NOT_FOUND
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.dtos.training_session_dto import TrainingSessionDTO
from domain.db import get_db_session
from repositories.training_session_repo import TrainingSessionRepository
from config.settings import settings


class TrainRequest(BaseModel):
    dataset: str
    gates: list[str]
    discount: float
    lr: float
    max_length: int
    autoencoder_path: str | None = None

@post("")
async def start_training(data: TrainRequest, db: Session = Provide(get_db_session)) -> dict:
    repo = TrainingSessionRepository(db)

    session = repo.create(
        dataset=data.dataset,
        gates=data.gates,
        discount_rate=data.discount,
        learning_rate=data.lr,
        max_architecture_length=data.max_length,
        autoencoder_path=data.autoencoder_path,
    )

    cmd = [
        ".venv/bin/python", "src/training/train.py",
        "--session_id", str(session.id),
        "--dataset", data.dataset,
        "--gates", *data.gates,
        "--discount", str(data.discount),
        "--lr", str(data.lr),
        "--max_length", str(data.max_length)
    ]
    if data.autoencoder_path:
        cmd += ["--autoencoder_path", data.autoencoder_path]


    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/session_{session.id}.log"

    with open(log_path, "w") as log_file:
        subprocess.Popen(
            cmd,
            cwd=settings.project_root,
            stdout=log_file,
            stderr=log_file
        )

    return {"status": "started", "training_id": session.id}


@get("")
async def list_sessions(db: Session = Provide(get_db_session)) -> list[TrainingSessionDTO]:
    repo = TrainingSessionRepository(db)
    try:
        sessions = repo.get_all()
        return [TrainingSessionDTO.model_validate(s) for s in sessions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@get("{session_id:int}")
async def get_session(session_id: int = Parameter(), db: Session = Provide(get_db_session)) -> TrainingSessionDTO:
    repo = TrainingSessionRepository(db)
    session = repo.get_by_id(session_id)
    if not session:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")
    return TrainingSessionDTO.model_validate(session)


sessions_router = Router(
    path="/sessions/",
    route_handlers=[
        start_training,
        list_sessions,
        get_session
    ]
)

