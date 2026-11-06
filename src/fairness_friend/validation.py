"""Validate at the library boundary, including calls without a frontend."""

import math
import re
from numbers import Real

import numpy as np
import pandas as pd

from .models import Dataset

_ATTRIBUTE = re.compile(r"A[1-9][0-9]*\Z")


class DatasetValidationError(ValueError):
    """Machine-readable input error; UI code should branch on code/details."""

    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code = code
        self.details = details
        self.missing_roles = details.get("missing_roles", [])
        self.missing_columns = details.get("missing_columns", [])

    def to_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "details": self.details.copy()}


def _validate_column_names(names) -> None:
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise DatasetValidationError("invalid_column_names", "Column names must be non-empty strings.")
    if len(set(names)) != len(names):
        raise DatasetValidationError("duplicate_columns", "Column names must be unique.")


def _validate_mapping(mapping) -> None:
    if not isinstance(mapping, dict):
        raise DatasetValidationError("invalid_mapping", "column_map must be a dictionary.")
    for role, column in mapping.items():
        if not isinstance(role, str) or (role not in ("Y", "R") and not _ATTRIBUTE.fullmatch(role)):
            raise DatasetValidationError("invalid_role", "Roles must be A1, A2, ..., Y or R.")
        if not isinstance(column, str) or not column.strip():
            raise DatasetValidationError("invalid_mapping", "Mapped column names must be non-empty strings.")


def _validate_epsilon(epsilon) -> float:
    if (
        isinstance(epsilon, (bool, np.bool_))
        or not isinstance(epsilon, Real)
        or not math.isfinite(epsilon)
        or not 0 <= epsilon <= 1
    ):
        raise DatasetValidationError("invalid_epsilon", "epsilon must be a finite number between 0 and 1.")
    return float(epsilon)


def _native_group(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise DatasetValidationError(
        "invalid_group", "Sensitive groups must be strings, booleans or finite numbers."
    )


def _validate_dataset(dataset: Dataset, attribute: str | None = None) -> list[str]:
    if not isinstance(dataset, Dataset) or not isinstance(dataset.data, pd.DataFrame):
        raise DatasetValidationError("invalid_dataset", "Expected a Dataset containing a DataFrame.")
    data, mapping = dataset.data, dataset.column_map
    _validate_column_names(list(data.columns))
    _validate_mapping(mapping)
    if data.empty:
        raise DatasetValidationError("empty_data", "The dataset contains no rows.")
    if attribute is None:
        attributes = sorted((key for key in mapping if _ATTRIBUTE.fullmatch(key)), key=lambda key: int(key[1:]))
        if not attributes:
            raise DatasetValidationError(
                "missing_roles", "Map at least one sensitive attribute (A1, A2, ...).",
                missing_roles=["A"],
            )
    else:
        if not isinstance(attribute, str) or not _ATTRIBUTE.fullmatch(attribute):
            raise DatasetValidationError("invalid_attribute", "attribute must be a role such as A1.")
        attributes = [attribute]
    roles = [*attributes, "Y", "R"]
    missing = [role for role in roles if role not in mapping]
    if missing:
        raise DatasetValidationError("missing_roles", "Required analysis roles are not mapped.", missing_roles=missing)
    columns = [mapping[role] for role in roles]
    missing_columns = list(dict.fromkeys(col for col in columns if col not in data.columns))
    if missing_columns:
        raise DatasetValidationError("missing_columns", "Mapped columns do not exist.", missing_columns=missing_columns)
    if len(set(columns)) != len(columns):
        raise DatasetValidationError("duplicate_roles", "Analysis roles must refer to distinct columns.")
    missing_values = [col for col in columns if data[col].isna().any()]
    if missing_values:
        raise DatasetValidationError("missing_values", "Analysis columns cannot contain missing values.", columns=missing_values)
    for role in ("Y", "R"):
        if not data[mapping[role]].isin([0, 1]).all():
            raise DatasetValidationError(
                "non_binary_labels", "Y and R must contain only numeric 0/1 values.", role=role, column=mapping[role]
            )
    for role in attributes:
        for value in data[mapping[role]]:
            _native_group(value)
    return attributes
