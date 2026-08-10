"""
Defines the structure of data going into and out of the API using Pydantic.
FastAPI uses these to validate inputs and document the API automatically.
"""
from pydantic import BaseModel


class RequestIn(BaseModel):
    """What the user sends to the API."""
    text: str
    day_of_week: int = 0   # Monday=0 ... Sunday=6; defaults to Monday
    month: int = 1         # 1-12; defaults to January


class SimilarRequest(BaseModel):
    request_text: str
    assigned_department: str
    priority: str
    similarity_score: float


class PredictionOut(BaseModel):
    """What the API sends back."""
    model_config = {"protected_namespaces": ()}   # <-- add this line

    department: str
    model_priority: str
    final_priority: str
    urgency_escalated: bool
    urgency_terms: list[str]
    suggested_response: str
    similar_requests: list[SimilarRequest]