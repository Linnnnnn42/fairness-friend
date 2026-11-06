"""Input and result objects. Results contain JSON-compatible scalar values."""

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TypeAlias

import pandas as pd

GroupLabel: TypeAlias = str | int | float | bool


class Status(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNDEFINED = "UNDEFINED"


@dataclass
class Dataset:
    """A table and logical-role → column-name mapping; incomplete mappings are allowed.

    Use A1, A2, ... for sensitive attributes, Y for truth, and R for decisions.
    The caller may prepare the table/mapping before passing it to an analysis API.
    Analysis never modifies this object or its DataFrame.
    """

    data: pd.DataFrame
    column_map: dict[str, str] = field(default_factory=dict)


class _Result:
    def to_dict(self) -> dict:
        """Return independent dictionaries/lists with string statuses and None."""
        def plain(value):
            if isinstance(value, Status):
                return value.value
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, list):
                return [plain(item) for item in value]
            return value

        return plain(asdict(self))


@dataclass(frozen=True)
class GroupRate(_Result):
    group: GroupLabel
    numerator: int
    denominator: int
    value: float | None
    reason: str | None = None


@dataclass(frozen=True)
class ComponentResult(_Result):
    gap: float | None
    status: Status
    groups: list[GroupRate]
    reason: str | None = None


@dataclass(frozen=True)
class CriterionResult(_Result):
    gap: float | None
    status: Status
    components: dict[str, ComponentResult]


@dataclass(frozen=True)
class AttributeAssessment(_Result):
    attribute: str
    column_name: str
    epsilon: float
    independence: CriterionResult
    separation: CriterionResult
    sufficiency: CriterionResult


@dataclass(frozen=True)
class FairnessReport(_Result):
    epsilon: float
    n_rows: int
    attributes: dict[str, AttributeAssessment]
