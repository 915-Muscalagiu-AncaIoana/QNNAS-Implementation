from litestar import post, Router
from pydantic import BaseModel
import subprocess

class TrainRequest(BaseModel):
    dataset: str
    gates: list[str]
    discount: float
    lr: float
    max_length: int
    autoencoder_path: str | None = None

@post("/start-training")
async def start_training(data: TrainRequest) -> dict:
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

    return {"status": "started", "cmd": " ".join(cmd)}

start_training_router = Router(path="/", route_handlers=[start_training])
