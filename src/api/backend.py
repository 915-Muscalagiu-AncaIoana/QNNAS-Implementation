from litestar import Litestar
from litestar.di import Provide

from api.routes import sessions_router
from domain.db import get_db_session

app = Litestar(
    route_handlers=[
        sessions_router
    ],
    dependencies={
        "db": Provide(get_db_session)
    }
)