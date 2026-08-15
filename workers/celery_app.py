
from pathlib import Path
import os


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from celery import Celery

from config.settings import settings
from workers.queues import TASK_ROUTES


_broker_root = Path(settings.celery_broker_data_folder)
(_broker_root / "out").mkdir(parents=True, exist_ok=True)
(_broker_root / "processed").mkdir(parents=True, exist_ok=True)

_sqlite_path = settings.celery_result_backend.removeprefix("db+sqlite:///")
Path(_sqlite_path).parent.mkdir(parents=True, exist_ok=True)

app = Celery("nyayai")

app.conf.broker_url = settings.celery_broker_url #type: ignore
app.conf.broker_transport_options = {
    # producer and consumer run on the same machine here, so "in" and "out"
    # both point at the same directory - see Kombu's filesystem transport docs
    "data_folder_in": f"{settings.celery_broker_data_folder}/out",
    "data_folder_out": f"{settings.celery_broker_data_folder}/out",
    "data_folder_processed": f"{settings.celery_broker_data_folder}/processed",
}
app.conf.result_backend = settings.celery_result_backend #type: ignore
app.conf.task_routes = TASK_ROUTES

# task results (the report dict) are small JSON, not files - fine to keep
# in the sqlite backend indefinitely for now. revisit if this grows.
app.conf.result_expires = None


app.conf.task_soft_time_limit = 300  # seconds
app.conf.task_time_limit = 360  # seconds

app.conf.broker_connection_retry_on_startup = True


app.conf.imports = ("workers.tasks",)