
import asyncio
import logging
from contextlib import asynccontextmanager
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from api.routes import upload, jobs, health
from api.middleware.timing import TimingMiddleware
from api.routes import chat as chat_routes
from api.routes import analysis as analysis_routes

# Move the ML models import to the global scope
from api.services.ml_service import NyayAI_Models

logger = logging.getLogger(__name__)

IDLE_UNLOAD_SECONDS = 600  # 10 min

async def _idle_unloader():
    from api.services.ml_service import NyayAI_Models
    while True:
        await asyncio.sleep(60)
        if not NyayAI_Models._last_used:
            continue
        idle_for = time.monotonic() - max(NyayAI_Models._last_used.values())
        if idle_for > IDLE_UNLOAD_SECONDS and NyayAI_Models._cache:
            logger.info(f"LSI/RR/CJPE idle for {idle_for:.0f}s — unloading")
            NyayAI_Models.unload_all()


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # ml_models = {}
    # try:
    #     logger.info("Loading InLegalBERT models...")

    #     try:
    #         ml_models["lsi"] = NyayAI_Models.load_lsi_model()
    #         logger.info("  LSI model loaded")
    #     except Exception as e:
    #         logger.warning(f"  LSI model failed to load: {e}")

    #     try:
    #         ml_models["rr"] = NyayAI_Models.load_rr_model()
    #         logger.info("  RR model loaded")
    #     except Exception as e:
    #         logger.warning(f"  RR model failed to load: {e}")

    #     try:
    #         ml_models["cjpe"] = NyayAI_Models.load_cjpe_model()
    #         logger.info("  CJPE model loaded")
    #     except Exception as e:
    #         logger.warning(f"  CJPE model failed to load: {e}")

    #     if ml_models:
    #         app.state.ml_models = ml_models
    #         logger.info(f"{len(ml_models)}/3 InLegalBERT models ready")
    #     else:
    #         logger.warning("No ML models loaded — /analyze/* will return 503")
    #         app.state.ml_models = None
            
    # except Exception as e:
    #     logger.warning(f"ML dependencies failed: {e}")
    #     app.state.ml_models = None

    
    yield 

    
    from api.services.ml_service import NyayAI_Models
    NyayAI_Models.unload_all()
    logger.info("ML models unloaded")

    # yield  # --- app runs here ---

    # # Cleanup (release GPU memory)
    # if hasattr(app.state, "ml_models") and app.state.ml_models:
    #     app.state.ml_models.clear()
    #     logger.info("ML models unloaded")


# app = FastAPI(title="NyayAI", lifespan=lifespan)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173", "http://localhost:3000"], 
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# app.add_middleware(TimingMiddleware)

# # --- Existing routes (OCR error detection) ---
# app.include_router(upload.router)
# app.include_router(jobs.router)
# app.include_router(health.router)

# # debug routes poke at Celery internals with no auth in front of them -
# # only mount when settings.debug is True, never by default in a real deployment
# if settings.debug:
#     from api.routes import debug
#     app.include_router(debug.router)

# # --- New routes (Chat + Analysis) ---
# app.include_router(chat_routes.router)
# app.include_router(analysis_routes.router)

# app.mount("/files", StaticFiles(directory=settings.outputs_dir), name="files") # /files is the URL path where the static files will be served from. StaticFiles is a FastAPI class that serves static files from a specified directory. The directory parameter specifies the local directory where the static files are located, which is set to settings.outputs_dir. The name parameter assigns a name to this static files route, which can be used for reverse URL lookups within the application. So a file saved as data/outputs/<job_id>_annotated.pdf is reachable as /files/<job_id>_annotated.pdf.



app = FastAPI(title="NyayAI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"], 
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TimingMiddleware)


app.include_router(upload.router)
app.include_router(jobs.router)
app.include_router(health.router)


if settings.debug:
    from api.routes import debug
    app.include_router(debug.router)


app.include_router(chat_routes.router)
app.include_router(analysis_routes.router)

app.mount("/files", StaticFiles(directory=settings.outputs_dir), name="files")
