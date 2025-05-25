from litestar import Litestar
from api.routes import start_training_router

app = Litestar(route_handlers=[start_training_router])
