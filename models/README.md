# Model artifacts

The trained pipelines are **not stored in git** — one is 167 MB, over GitHub's
100 MB per-file limit. They are published as assets on the
[`v1.0.0` release](https://github.com/OutisAyo/council-classifier-/releases/tag/v1.0.0).

```bash
python scripts/download_models.py      # or: make models
```

| File | Size | Built in | Contents |
|---|---|---|---|
| `department_pipeline_rf.pkl` | ~10 MB | nb 07 | `Pipeline(TfidfVectorizer -> RandomForestClassifier)` — predicts `assigned_department` from request text |
| `priority_pipeline_final.pkl` | ~167 MB | nb 14 | `Pipeline(ColumnTransformer[TF-IDF text + passthrough day_of_week/month] -> class-weighted RandomForestClassifier)` — predicts `priority` |
| `retrieval_vectorizer.pkl` | ~24 KB | nb 15 | `TfidfVectorizer` (max 1500 features) used to embed requests for similarity search |
| `dataset_for_retrieval.pkl` | ~16 MB | nb 15 | cleaned 254k-row DataFrame (`request_text`, `assigned_department`, `priority`, `date_received`, `date_closed`, ...) for retrieval + duplicate checks |
| `response_templates.pkl` | ~4 KB | nb 15 | `dict[department][priority] -> str` acknowledgement templates |
| `urgency_lexicon.pkl` | ~4 KB | nb 18 | `dict` with `explicit` / `hazard` / `distress` term lists for the rule-based escalation layer |

## Regenerating from scratch

Run the notebooks in order against your own copy of the Swansea Council FOI
dataset (see the project README). Notebooks 07, 14, 15 and 18 write these files;
point their `MODELS_DIR` at this folder.
