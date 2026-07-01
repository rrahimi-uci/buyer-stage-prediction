# Example: buyer-stage (real-estate)

The worked example that reconstructs the original AWS *buyer-stage-ml* problem on this
open-source framework. **Everything domain-specific lives here** — the framework under
`src/automl_template/` knows nothing about real estate.

| File | Role |
|---|---|
| `pipeline.yaml` | target=`buyer_stage`, entity=`member_id`, 4 classes, windows, budgets, thresholds |
| `feature_spec.yaml` | one-hot / 180s-clip / coalesce ported from `src_etl/config/constants.py` |
| `label_join.sql` | ±2-day symmetric Qualtrics survey join (port of `buyer_intent_label_data_set.sql`) |
| `label_collapse.py` | `stage_label_{1,2,3}` → `buyer_stage` |
| `seed_raw.py` | loads + reconciles the 229-col train / 230-col test CSVs, `member_id`→`entity_id` |
| `golden/` | checked-in derived parity fixtures (column order + reshape golden) |
| `sample_data/` | **not committed** — drop the CSVs here (see below) |

## The classes

| Stage | Sample rows |
|---|---|
| Active Searcher | 2076 |
| Ready to Transact | 684 |
| Dreamer | 642 |
| Casual Explorer | 598 |

(~3.5:1 imbalance — handled via `metric=macro_f1` + class weighting.)

## Getting the data

The sample CSVs are de-identified real Move/Realtor.com behavioral data. They live in
`sample_data/` for local runs but are **git-ignored** — never published (data provenance; see
[docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)). The seeder uses them when present and falls
back to synthetic data when absent, so the example always runs:

```bash
python -m examples.buyer_stage.seed_raw    # real CSVs if present in sample_data/, else synthetic
```

For a publishable demo, replace these CSVs with a synthetic or public dataset of the same shape.
