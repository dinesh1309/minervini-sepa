"""
Multi-source data layer for SEPA Trading Workflow.

Architecture:
    DataProvider (ABC) ← YFinanceProvider, OpenBBProvider, ...
         ↓
    DataAggregator (fallback chain, bad data detection, logging)
         ↓
    CacheLayer (Parquet-based, daily/weekly expiry)
"""
