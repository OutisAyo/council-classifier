"""
FastAPI application exposing the council request classifier.
Run locally with:  uvicorn app.main:app --reload
"""
from fastapi import FastAPI
from app.schemas import RequestIn, PredictionOut
from app.predictor import classify_request

app = FastAPI(
    title="UK Council Service Request Classifier",
    description="Automated classification, prioritisation and decision support for local council service requests.",
    version="1.0.0",
)


@app.get("/")
def root():
    """Health check — confirms the API is running."""
    return {"status": "online", "service": "council-request-classifier"}


@app.post("/classify", response_model=PredictionOut)
def classify(request: RequestIn):
    """
    Classify a service request.
    Takes request text (and optionally day_of_week and month),
    returns department, priority, suggested response, and similar requests.
    """
    result = classify_request(
        text=request.text,
        day_of_week=request.day_of_week,
        month=request.month,
    )
    return result