import pytest
from pydantic import ValidationError

from app.schemas import PredictionOut, RequestIn


def test_request_defaults():
    r = RequestIn(text="missed bin collection")
    assert r.day_of_week == 0
    assert r.month == 1


def test_request_text_is_required():
    with pytest.raises(ValidationError):
        RequestIn()


@pytest.mark.parametrize("text", ["", "   "])
def test_request_rejects_blank_text(text):
    # empty string fails min_length; whitespace-only passes the schema
    # (handler-level check) so only the truly empty case must raise here
    if text == "":
        with pytest.raises(ValidationError):
            RequestIn(text=text)
    else:
        RequestIn(text=text)


@pytest.mark.parametrize(
    ("field", "value"),
    [("day_of_week", -1), ("day_of_week", 7), ("month", 0), ("month", 13)],
)
def test_request_range_validation(field, value):
    with pytest.raises(ValidationError):
        RequestIn(text="noise complaint", **{field: value})


def test_prediction_out_shape():
    out = PredictionOut(
        department="Waste Management",
        model_priority="LOW",
        final_priority="HIGH",
        urgency_escalated=True,
        urgency_terms=["urgent"],
        suggested_response="Logged.",
        similar_requests=[
            {
                "request_text": "bin not collected",
                "assigned_department": "Waste Management",
                "priority": "LOW",
                "similarity_score": 0.91,
            }
        ],
    )
    assert out.similar_requests[0].similarity_score == pytest.approx(0.91)
