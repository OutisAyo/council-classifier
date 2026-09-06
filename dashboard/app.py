"""
Streamlit dashboard for the council request classifier.

Run from the repository root:
    streamlit run dashboard/app.py

Reuses the exact prediction logic that backs the FastAPI service
(``app.predictor``) so the dashboard and the API can never drift apart.
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import ModelsNotFoundError, get_store  # noqa: E402
from app.predictor import (  # noqa: E402
    check_for_duplicates,
    classify_request,
)

st.set_page_config(
    page_title="Council Request Classifier",
    page_icon="🏛️",
    layout="wide",
)


@st.cache_resource
def load_store():
    return get_store()


try:
    store = load_store()
except ModelsNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.title("🏛️ UK Council Service Request Classifier")
st.markdown(
    "**Automated classification, prioritisation and decision support "
    "for local council service requests**"
)
st.markdown("---")

tab1, tab2 = st.tabs(["📝 Classify a Request", "📊 Data Insights"])

# ------------------------------------------------------------------ classify
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        request_text = st.text_area(
            "Enter the service request text:",
            placeholder="e.g. live rat in house urgent",
            height=100,
        )
    with col2:
        submission_date = st.date_input("Submission date:", datetime.today())

    if st.button("Classify Request", type="primary"):
        if not request_text.strip():
            st.error("Please enter a request text first.")
        else:
            result = classify_request(
                text=request_text,
                day_of_week=submission_date.weekday(),
                month=submission_date.month,
            )

            m1, m2 = st.columns(2)
            m1.metric("Predicted Department", result["department"])
            m2.metric("Predicted Priority", result["final_priority"])

            if result["urgency_escalated"]:
                st.warning(
                    "⚠️ Escalated to HIGH — urgency language detected: "
                    + ", ".join(result["urgency_terms"])
                )

            st.subheader("Suggested Response")
            st.info(result["suggested_response"])

            duplicates, is_duplicate = check_for_duplicates(request_text)
            st.subheader("Duplicate Check")
            if is_duplicate:
                st.warning(
                    f"⚠️ {len(duplicates)} similar request(s) submitted recently "
                    "— possible duplicate."
                )
                st.dataframe(pd.DataFrame(duplicates), use_container_width=True)
            else:
                st.success("✅ No similar recent requests found.")

            st.subheader("Most Similar Historical Requests")
            st.dataframe(
                pd.DataFrame(result["similar_requests"]), use_container_width=True
            )

# -------------------------------------------------------------- data insights
with tab2:
    dataset = store.dataset
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{len(dataset):,}")
    c2.metric("Departments", dataset["assigned_department"].nunique())
    c3.metric(
        "Date Range",
        f"{dataset['date_received'].min().year} - {dataset['date_received'].max().year}",
    )

    st.subheader("Requests by Department")
    st.bar_chart(dataset["assigned_department"].value_counts())

    st.subheader("Priority Distribution")
    st.bar_chart(dataset["priority"].value_counts())

    st.subheader("Monthly Request Volume")
    monthly = dataset.groupby(dataset["date_received"].dt.to_period("M")).size()
    monthly.index = monthly.index.astype(str)
    st.line_chart(monthly)
