from pydantic import  Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    max_qubits_available: int = Field(16, alias="MAX_QUBITS_AVAILABLE")
    database_url: str = Field("postgresql://postgresql:postgresql@localhost:5433/qnnas", alias="DATABASE_URL")
    api_base_url: str = Field("http://localhost:8000", alias="API_BASE_URL")
    project_root: str = Field("/absolute/path/to/QNNAS-Implementation", alias="PROJECT_ROOT")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
