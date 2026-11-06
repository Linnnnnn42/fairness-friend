"""Structured binary-decision fairness assessments, independent of any UI."""

from .assessment import analyze_attribute, analyze_dataset
from .io import read_dataset
from .models import (
    AttributeAssessment,
    ComponentResult,
    CriterionResult,
    Dataset,
    FairnessReport,
    GroupRate,
    Status,
)
from .validation import DatasetValidationError

__version__ = "0.0.1"

__all__ = [
    "AttributeAssessment", "ComponentResult", "CriterionResult", "Dataset",
    "DatasetValidationError", "FairnessReport", "GroupRate", "Status",
    "analyze_attribute", "analyze_dataset", "read_dataset",
]
