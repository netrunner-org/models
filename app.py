import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv


load_dotenv()

from src.config import settings
from src.services import model_service
from src.api import router



# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
  logger.info("Loading models...")
  for model_config in settings.DEFAULT_MODELS:
    model_id = model_config["id"]
    quantize = model_config.get("quantize", False)
    try:
      model_service.load_model(model_id, quantize=quantize)
    except Exception as e:
      logger.error(f"Error loading model {model_id}: {str(e)}", exc_info=True)
  
  logger.info(f"Loaded models: {model_service.list_loaded_models()}")

  yield

  logger.info(f"Clearing models...")

# Create FastAPI app
app = FastAPI(
  title=settings.API_TITLE,
  description=settings.API_DESCRIPTION,
  version=settings.API_VERSION,
  lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
  CORSMiddleware,
  allow_origins=settings.CORS_ORIGINS,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
  start_time = time.time()
  try:
    response = await call_next(request)
    process_time = time.time() - start_time
    print("REQUEST: ", request)
    logger.info(
      f"{request.method} {request.url.path} - "
      f"Status: {response.status_code} - "
      f"Time: {process_time:.4f}s"
    )
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response
  except Exception as e:
    process_time = time.time() - start_time
    logger.error(
      f"{request.method} {request.url.path} - "
      f"Error: {str(e)} - "
      f"Time: {process_time:.4f}s"
    )
    raise

# Add exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
  logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
  return JSONResponse(
    status_code=500,
    content={"detail": f"Internal Server Error: {str(exc)}"}
  )

# Include router
app.include_router(router)

if __name__ == "__main__":
  uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)