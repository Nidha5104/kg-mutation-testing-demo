# KG Mutation Testing — Interactive Research Dashboard

Reuses your notebook's actual implementation. No metrics, formulas, mutation operators,
recommender, or thresholds were changed or invented.

## Folder structure
```
kg-mutation-dashboard/
├── NOTEBOOK_EXPORT_CELL.py   # paste into your Colab notebook (Step 1)
├── backend/
│   ├── kg_testing_lib.py     # VERBATIM copy of your notebook's Section 7B module
│   ├── behavioral_analysis.py# VERBATIM logic from Sections 18/19 (BR1-BR5, anomaly score)
│   ├── app.py                # Flask API wiring the UI to your real functions
│   ├── requirements.txt
│   └── demo_bundle.pkl       # <- you generate this (Step 1), not included here
└── frontend/
    └── index.html            # single-file dashboard (no build step)
```

## Why a "bundle" file at all?
Your recommender/mutation/metric code needs runtime data (product/KG maps, preference
profiles, similarity tables, and the **frozen, development-calibrated thresholds** from
Sections 18A/19A) that only exist after your notebook queries BigQuery and runs its
pipeline. `NOTEBOOK_EXPORT_CELL.py` is the **only new code**: it serializes those
already-computed objects to `demo_bundle.pkl`. It does not recompute, resample, or alter
anything.

## Step 1 — Generate the real demo bundle (in Colab)
1. Run your notebook normally through Section 34B (i.e. `DEV_MODE=False`, held-out run complete).
2. Paste the contents of `NOTEBOOK_EXPORT_CELL.py` as a new final cell and run it.
3. Download the resulting `demo_bundle.pkl` from the Colab file browser.
4. Place it at `backend/demo_bundle.pkl`.

## Step 2 — Install & run the backend
```bash
cd kg-mutation-dashboard/backend
pip install -r requirements.txt
python app.py
```
Runs at `http://localhost:5050`.

## Step 3 — Open the frontend
Just open `frontend/index.html` in a browser (double-click, or `open frontend/index.html`).
It calls the backend at `http://localhost:5050/api/*`.

## What each screen does
- **Left panel:** pick an eligible demo user → pick an operator (M1–M5) → "Find valid
  candidates" runs your own `get_candidate_pool` / `apply_mutation` / `validate_mutation`
  to list real, domain-valid mutation targets → pick one → "Run mutation test".
- **Right panel:** original vs mutated KG (Product–Category–Brand–Department), original vs
  mutated Top-10 (unchanged / new / disappeared / moved), the actual behavioral metrics
  (`ranking_change`, `semantic_shift`, `diversity_shift`, `locality`), BR1–BR5 flags, the
  statistical anomaly score plotted against your frozen 90th-percentile cutoff, and a
  verdict: **ANOMALOUS (warrants investigation)** or **NON-ANOMALOUS** — never phrased as a
  confirmed bug.

## Demonstrating in your viva
1. Point out the pipeline stepper at the top — matches your methodology diagram exactly.
2. Pick a user, choose **M3 (Category Relation Removal)** — dramatic, easy to explain: the
   category edge is deleted in the "Mutated KG" panel.
3. Run the test; walk through: KG diff → Top-10 diff → behavioral metrics → BR flags →
   anomaly score vs. the frozen cutoff → interpretation text.
4. Repeat with **M1** or **M5** for a case that stays NON-ANOMALOUS, to show the framework
   doesn't just flag everything.
5. Mention explicitly: recommender, mutation operators, metrics, BR1–BR5, and the anomaly
   cutoff are all your notebook's own code (`kg_testing_lib.py`, `behavioral_analysis.py`) —
   the dashboard is a UI layer, not a reimplementation.

## Note on the demo subset
The dashboard only offers users drawn from your held-out evaluation sample (a small subset,
not the full pool) so every user shown already has real computed results in that run. This
is clearly separate from your full research experiment/results in the notebook itself.
