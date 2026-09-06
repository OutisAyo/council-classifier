"""
Defines the structure of data going into and out of the API using Pydantic.
FastAPI uses these to validate inputs and document the API automatically.
"""
from pydantic import BaseModel, Field


class RequestIn(BaseModel):
    """What the user sends to the API."""

    text: str = Field(..., min_length=1, description="The service request text.")
    day_of_week: int = Field(0, ge=0, le=6, description="Monday=0 ... Sunday=6.")
    month: int = Field(1, ge=1, le=12, description="Calendar month, 1-12.")


class SimilarRequest(BaseModel):
    request_text: str
    assigned_department: str
    priority: str
    similarity_score: float


class PredictionOut(BaseModel):
    """What the API sends back."""

    model_config = {"protected_namespaces": ()}

    department: str
    model_priority: str
    final_priority: str
    urgency_escalated: bool
    urgency_terms: list[str]
    suggested_response: str
    similar_requests: list[SimilarRequest]
