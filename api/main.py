"""
FastAPI app entrypoint. wires the three route modules together, serves
outputs/ as static files (so the frontend can download the annotated PDF
and HTML report directly), and allows the Vite dev server's origin so
local frontend development isn't blocked by CORS.

also adds the timing middleware (X-Process-Time header on every response)
and, only when settings.debug is True, mounts the /debug/* routes -
they poke at Celery internals directly and there's no auth in front of
this API, so they're opt-in rather than always-on.

no auth - matches the roadmap's own known-limitations list ("no
authentication on the API... must be added before any deployment").
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from api.routes import upload, jobs, health
from api.middleware.timing import TimingMiddleware

app = FastAPI(title="NyayAI")
from api.routes import chat as chat_routes
from api.routes import analysis as analysis_routes

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: preload InLegalBERT models into app.state so every request
    reuses the same GPU-resident weights. Graceful degradation — if model
    files are missing the server still starts, just the /analyze/* routes
    return 503.
    """
    ml_models = {}
    try:
        from api.services.ml_service import NyayAI_Models

        logger.info("⏳ Loading InLegalBERT models...")

        try:
            ml_models["lsi"] = NyayAI_Models.load_lsi_model()
            logger.info("  ✅ LSI model loaded")
        except Exception as e:
            logger.warning(f"  ⚠️ LSI model failed to load: {e}")

        try:
            ml_models["rr"] = NyayAI_Models.load_rr_model()
            logger.info("  ✅ RR model loaded")
        except Exception as e:
            logger.warning(f"  ⚠️ RR model failed to load: {e}")

        try:
            ml_models["cjpe"] = NyayAI_Models.load_cjpe_model()
            logger.info("  ✅ CJPE model loaded")
        except Exception as e:
            logger.warning(f"  ⚠️ CJPE model failed to load: {e}")

        if ml_models:
            app.state.ml_models = ml_models
            logger.info(f"🧠 {len(ml_models)}/3 InLegalBERT models ready")
        else:
            logger.warning("⚠️ No ML models loaded — /analyze/* will return 503")
            app.state.ml_models = None
    except ImportError as e:
        logger.warning(f"⚠️ ML dependencies not available: {e}")
        app.state.ml_models = None

    yield  # --- app runs here ---

    # Cleanup (release GPU memory)
    if hasattr(app.state, "ml_models") and app.state.ml_models:
        app.state.ml_models.clear()
        logger.info("🧹 ML models unloaded")


app = FastAPI(title="NyayAI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite dev server + common React port
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TimingMiddleware)

# --- Existing routes (OCR error detection) ---
app.include_router(upload.router)
app.include_router(jobs.router)
app.include_router(health.router)

# debug routes poke at Celery internals with no auth in front of them -
# only mount when settings.debug is True, never by default in a real deployment
if settings.debug:
    from api.routes import debug
    app.include_router(debug.router)

app.mount("/files", StaticFiles(directory=settings.outputs_dir), name="files") # /files is the URL path where the static files will be served from. StaticFiles is a FastAPI class that serves static files from a specified directory. The directory parameter specifies the local directory where the static files are located, which is set to settings.outputs_dir. The name parameter assigns a name to this static files route, which can be used for reverse URL lookups within the application. So a file saved as data/outputs/<job_id>_annotated.pdf is reachable as /files/<job_id>_annotated.pdf.
# --- New routes (Chat + Analysis) ---
app.include_router(chat_routes.router)
app.include_router(analysis_routes.router)

app.mount("/files", StaticFiles(directory=settings.outputs_dir), name="files")
