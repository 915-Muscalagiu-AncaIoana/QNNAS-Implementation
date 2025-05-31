from litestar import Litestar
from litestar.di import Provide

from api.routes import start_training_router, list_sessions
from domain.db import get_db_session

app = Litestar(
    route_handlers=[start_training_router, list_sessions],
    dependencies={
        "db": Provide(get_db_session)
    }
)