# Model artifacts

The trained pipelines are stored in this repository via **[Git LFS](https://git-lfs.com/)**
(the priority pipeline is 167 MB, over GitHub's 100 MB per-file limit for regular files).

```bash
git lfs install      # once per machine
git lfs pull         # fetch the actual files after cloning
```

`git clone` on a machine with Git LFS installed pulls them automatically. `make models`
runs `git lfs pull` for you.

| File | Size | Built in | Contents |
|---|---|---|---|
| `department_pipeline_rf.pkl` | ~10 MB | nb 07 | `Pipeline(TfidfVectorizer -> RandomForestClassifier)` predicting `assigned_department` from request text |
| `priority_pipeline_final.pkl` | ~167 MB | nb 14 | `Pipeline(ColumnTransformer[TF-IDF text + passthrough day_of_week/month] -> class-weighted RandomForestClassifier)` predicting `priority` |
| `retrieval_vectorizer.pkl` | ~24 KB | nb 15 | `TfidfVectorizer` (max 1500 features) used to embed requests for similarity search |
| `dataset_for_retrieval.pkl` | ~16 MB | nb 15 | cleaned 254k-row DataFrame (`request_text`, `assigned_department`, `priority`, `date_received`, `date_closed`, ...) for retrieval + duplicate checks |
| `response_templates.pkl` | ~4 KB | nb 15 | `dict[department][priority] -> str` acknowledgement templates |
| `urgency_lexicon.pkl` | ~4 KB | nb 18 | `dict` with `explicit` / `hazard` / `distress` term lists for the rule-based escalation layer |

## Regenerating from scratch

Run the notebooks in order against your own copy of the Swansea Council FOI
dataset (see the project README). Notebooks 07, 14, 15 and 18 write these files;
point their `MODELS_DIR` at this folder.
