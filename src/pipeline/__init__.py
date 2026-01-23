"""
SEPA Trading Workflow - Pipeline Package

This package provides the pipeline orchestration for
running the complete SEPA agent workflow.
"""

from .sepa_pipeline import (
    SEPAPipeline,
    PipelineResult,
    run_daily_scan
)

__all__ = [
    'SEPAPipeline',
    'PipelineResult',
    'run_daily_scan'
]
