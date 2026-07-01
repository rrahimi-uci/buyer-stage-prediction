-- Port of buyer_intent_label_data_set.sql.
-- Joins per-member feature snapshots to Qualtrics survey labels on member_id within a
-- SYMMETRIC ±2-day window. The label may be up to 2 days in the FUTURE relative to the
-- snapshot — this is intentional and is why the PIT contract is a window, not feature<=label.
--
-- Placeholders {{features}} / {{survey}} are registered relations (DuckDB) or tables.
SELECT
    f.member_id,
    f.snapshot_date_mst_yyyymmdd        AS snapshot_date,
    s.start_date                         AS label_start_date,
    s.stage_label_1,
    s.stage_label_2,
    s.stage_label_3
FROM {{features}} AS f
INNER JOIN {{survey}} AS s
    ON f.member_id = s.member_id
   AND abs(date_diff('day',
            strptime(CAST(f.snapshot_date_mst_yyyymmdd AS VARCHAR), '%Y%m%d'),
            s.start_date)) <= 2;
