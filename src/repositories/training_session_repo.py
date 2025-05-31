import datetime
from contextlib import contextmanager

from sqlalchemy.orm import Session

from domain.db import get_db_session
from domain.domain import TrainingSession, TrainingStatus


class TrainingSessionRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        dataset: str,
        gates: list[str],
        discount_rate: float,
        learning_rate: float,
        max_architecture_length: int,
        autoencoder_path: str | None = None
    ) -> TrainingSession:
        session = TrainingSession(
            dataset=dataset,
            gates=gates,
            discount_rate=discount_rate,
            learning_rate=learning_rate,
            max_architecture_length=max_architecture_length,
            autoencoder_path=autoencoder_path,
            status=TrainingStatus.pending,
            start_time=datetime.datetime.now(datetime.UTC)
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def get_all(self):
        return self.db.query(TrainingSession).all()

    def get_by_id(self, session_id: int):
        return self.db.query(TrainingSession).filter(TrainingSession.id == session_id).first()

    def update_status(self, session_id: int, new_status: TrainingStatus) -> TrainingSession:
        session = self.db.query(TrainingSession).filter_by(id=session_id).first()
        if not session:
            raise ValueError(f"TrainingSession with id {session_id} not found.")
        session.status = new_status
        self.db.commit()
        self.db.refresh(session)
        return session

@contextmanager
def get_training_session_repository():
    db = next(get_db_session())
    try:
        yield TrainingSessionRepository(db)
    finally:
        db.close()