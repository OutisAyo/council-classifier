"""
Core prediction logic: department classification, priority prediction
with hybrid urgency escalation, response generation, and duplicate detection.
"""
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from app.models import store


def detect_urgency(text):
    """Return the list of urgency terms found in the text."""
    text_lower = text.lower()
    return [word for word in store.all_urgency_words if word in text_lower]


def find_similar_requests(text, top_n=5):
    """Find the most similar historical requests and their average resolution time."""
    new_vector = store.retrieval_vectorizer.transform([text])
    similarities = cosine_similarity(new_vector, store.historical_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    similar = store.dataset.iloc[top_indices].copy()
    similar["similarity_score"] = similarities[top_indices]

    valid = similar[
        (similar["date_closed"].notna())
        & ((similar["date_closed"] - similar["date_received"]).dt.days >= 0)
    ]
    avg_days = (
        (valid["date_closed"] - valid["date_received"]).dt.days.mean()
        if len(valid) > 0
        else None
    )

    results = similar[["request_text", "assigned_department", "priority", "similarity_score"]]
    return results.to_dict(orient="records"), avg_days


def classify_request(text, day_of_week, month):
    """
    Full prediction pipeline: department, hybrid priority, response, and context.
    Returns a dictionary ready to be sent as an API response.
    """
    # Department prediction
    department = store.department_pipeline.predict([text])[0]

    # Priority prediction (model)
    priority_input = pd.DataFrame(
        {"request_text": [text], "day_of_week": [day_of_week], "month": [month]}
    )
    model_priority = store.priority_pipeline.predict(priority_input)[0]

    # Hybrid urgency override
    urgency_matches = detect_urgency(text)
    if urgency_matches:
        final_priority = "HIGH"
    else:
        final_priority = model_priority

    # Response template
    template = store.templates.get(department, {}).get(
        final_priority,
        "Thank you for your request. It has been logged and will be reviewed.",
    )
    similar, avg_resolution = find_similar_requests(text)
    if avg_resolution is not None:
        context = f" Based on similar requests, this typically takes around {avg_resolution:.0f} days to resolve."
    else:
        context = ""

    return {
        "department": department,
        "model_priority": model_priority,
        "final_priority": final_priority,
        "urgency_escalated": len(urgency_matches) > 0,
        "urgency_terms": urgency_matches,
        "suggested_response": template + context,
        "similar_requests": similar,
    }
