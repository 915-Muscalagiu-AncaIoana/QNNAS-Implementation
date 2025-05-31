import subprocess

from litestar import post, get, Router
from litestar.exceptions import HTTPException
from litestar.params import Dependency
from pydantic import BaseModel

from api.dtos.training_session_dto import TrainingSessionDTO
from repositories.training_session_repo import TrainingSessionRepository


class TrainRequest(BaseModel):
    dataset: str
    gates: list[str]
    discount: float
    lr: float
    max_length: int
    autoencoder_path: str | None = None

@post("/start-training")
async def start_training(data: TrainRequest, repo: TrainingSessionRepository = Dependency()) -> dict:
    cmd = [
        ".venv/bin/python", "src/training/train.py",
        "--dataset", data.dataset,
        "--gates", *data.gates,
        "--discount", str(data.discount),
        "--lr", str(data.lr),
        "--max_length", str(data.max_length)
    ]
    if data.autoencoder_path:
        cmd += ["--autoencoder_path", data.autoencoder_path]

    # Save output to file for debugging
    with open("training_output.log", "w") as log_file:
        subprocess.Popen(cmd,cwd="/Users/ancaioanamuscalagiu/Documents/QNNAS-Implementation", stdout=log_file, stderr=log_file)

    session = repo.create(
        dataset=data.dataset,
        gates=data.gates,
        discount_rate=data.discount,
        learning_rate=data.lr,
        max_architecture_length=data.max_length,
        autoencoder_path=data.autoencoder_path,
    )
    print(session)
    return {"status": "started", "training_id": session.id}

@get("/sessions")
async def list_sessions(repo: TrainingSessionRepository = Dependency()) -> list[TrainingSessionDTO]:
    try:
        sessions = repo.get_all()
        print(sessions)  # For debugging
        return [TrainingSessionDTO.model_validate(s) for s in sessions]
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

start_training_router = Router(path="/", route_handlers=[start_training])
