"""
FastAPI application exposing the council request classifier.
Run locally with:  uvicorn app.main:app --reload
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.models import ModelsNotFoundError, get_store
from app.predictor import classify_request
from app.schemas import PredictionOut, RequestIn

logger = logging.getLogger("council-classifier")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the model store on startup so the first request isn't slow."""
    try:
        get_store()
        logger.info("Model store loaded.")
    except ModelsNotFoundError as exc:
        logger.warning("Starting without models: %s", exc)
    yield


app = FastAPI(
    title="UK Council Service Request Classifier",
    description=(
        "Automated classification, prioritisation and decision support "
        "for local council service requests."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
@app.get("/")
def health():
    """Health check — confirms the API is running."""
    return {"status": "online", "service": "council-request-classifier"}


@app.post("/classify", response_model=PredictionOut)
def classify(request: RequestIn):
    """
    Classify a service request.
    Takes request text (and optionally day_of_week and month),
    returns department, priority, suggested response, and similar requests.
    """
    try:
        return classify_request(
            text=request.text,
            day_of_week=request.day_of_week,
            month=request.month,
        )
    except ModelsNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
