"""Feature ETL: the shared reshape code path used IDENTICALLY by training and scoring.

This single-code-path property is what guarantees no train/serve skew (ARCHITECTURE §5).
"""
