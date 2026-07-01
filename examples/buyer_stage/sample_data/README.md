# sample_data — git-ignored

This folder holds the buyer-stage sample CSVs for local runs. They are **git-ignored**
(`.gitignore` excludes `*.csv`/`*.parquet` here) because they are de-identified real behavioral
data — a data-provenance concern independent of this repo's code license. They are present for
development but must not be committed/published.

- `buyer_stage_tr_val_set.csv` — 229 cols, label column `buyer_stage` (last), no `member_id`
- `buyer_stage_test_set.csv` — 230 cols, `member_id` first, label column `stage_label_1`

If absent, `seed_raw` falls back to synthetic data, so the example still runs. To point at a
different source: `python -m examples.buyer_stage.seed_raw --source-dir /path/to/csvs`.

For a publishable demo, replace these with a **synthetic or public** dataset of the same shape.
