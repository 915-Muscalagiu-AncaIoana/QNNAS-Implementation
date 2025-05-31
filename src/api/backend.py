from litestar import Litestar
from api.routes import start_training_router, list_sessions
from repositories.training_session_repo import get_training_session_repository

app = Litestar(
    route_handlers=[start_training_router, list_sessions],
    dependencies={"repo": get_training_session_repository}
)