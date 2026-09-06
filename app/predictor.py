"""
Core prediction logic: department classification, priority prediction
with hybrid urgency escalation, response generation, and duplicate detection.
"""
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.models import get_store

DEFAULT_RESPONSE = "Thank you for your request. It has been logged and will be reviewed."


def detect_urgency(text):
    """Return the list of urgency terms found in the text."""
    store = get_store()
    text_lower = text.lower()
    return [word for word in store.all_urgency_words if word in text_lower]


def find_similar_requests(text, top_n=5):
    """Find the most similar historical requests and their average resolution time."""
    store = get_store()
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


def check_for_duplicates(text, recent_days=7, similarity_threshold=0.7, top_n=5):
    """
    Flag whether a near-identical request was filed in the recent window.
    Returns (matching_records, is_duplicate).
    """
    store = get_store()
    most_recent = store.dataset["date_received"].max()
    cutoff = most_recent - pd.Timedelta(days=recent_days)
    recent = store.dataset[store.dataset["date_received"] >= cutoff].copy()
    if len(recent) == 0:
        return [], False

    recent_vectors = store.retrieval_vectorizer.transform(recent["request_text"])
    new_vector = store.retrieval_vectorizer.transform([text])
    recent["similarity_score"] = cosine_similarity(new_vector, recent_vectors).flatten()

    matches = (
        recent[recent["similarity_score"] >= similarity_threshold]
        .sort_values("similarity_score", ascending=False)
        .head(top_n)
    )
    records = matches[
        ["request_text", "assigned_department", "date_received", "similarity_score"]
    ].to_dict(orient="records")
    return records, len(matches) > 0


def classify_request(text, day_of_week, month):
    """
    Full prediction pipeline: department, hybrid priority, response, and context.
    Returns a dictionary ready to be sent as an API response.
    """
    store = get_store()

    # Department prediction
    department = store.department_pipeline.predict([text])[0]

    # Priority prediction (model)
    priority_input = pd.DataFrame(
        {"request_text": [text], "day_of_week": [day_of_week], "month": [month]}
    )
    model_priority = store.priority_pipeline.predict(priority_input)[0]

    # Hybrid urgency override
    urgency_matches = detect_urgency(text)
    final_priority = "HIGH" if urgency_matches else model_priority

    # Response template
    template = store.templates.get(department, {}).get(final_priority, DEFAULT_RESPONSE)
    similar, avg_resolution = find_similar_requests(text)
    if avg_resolution is not None:
        context = (
            f" Based on similar requests, this typically takes around "
            f"{avg_resolution:.0f} days to resolve."
        )
    else:
        context = ""

    return {
        "department": str(department),
        "model_priority": str(model_priority),
        "final_priority": str(final_priority),
        "urgency_escalated": len(urgency_matches) > 0,
        "urgency_terms": urgency_matches,
        "suggested_response": template + context,
        "similar_requests": similar,
    }
