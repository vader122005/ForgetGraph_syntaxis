# FORGETGRAPH

Dependency-aware data deletion verification for an ML pipeline (GDPR-style
"right to be forgotten" demo).

## Dependency chain

```
users (raw rows) -> features (derived) -> predictions (cached) -> model (Ridge)
```

Deleting a user must remove their influence from every layer of the chain.

## Files

| File | Purpose |
|---|---|
| `schema.py` | SQLite schema creation + synthetic data generation (200 users, seed=42) |
| `pipeline_core.py` | Model training, feature computation, deletion pipeline, re-ingestion guard |
| `verify.py` | Independent verification (fresh DB connection, reference model) |
| `app.py` | Streamlit dashboard |
| `test_forgetgraph.py` | pytest suite |
| `train_original.py` | One-off script: train and save the original model |

## Usage

```bash
python schema.py           # create forgetgraph.db with 200 synthetic users
python train_original.py   # train original model -> original_model.pkl
streamlit run app.py        # dashboard
pytest test_forgetgraph.py # test suite
```

## Model

`sklearn.linear_model.Ridge(alpha=1.0, random_state=42)` everywhere a model
is trained.
