"""Fairlearn-based computation, separate from file parsing and presentation."""

from functools import partial
from math import isclose

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame, count, false_positive_rate, selection_rate, true_positive_rate,
)

from .models import (
    AttributeAssessment, ComponentResult, CriterionResult, Dataset, FairnessReport,
    GroupRate, Status,
)
from .validation import _native_group, _validate_dataset, _validate_epsilon


def _conditional_prediction_rate(y_true, y_pred, *, outcome):
    """Guard the conditional denominator before invoking Fairlearn's TPR/FPR."""
    if not np.any(np.asarray(y_true) == outcome):
        return float("nan")
    metric = true_positive_rate if outcome == 1 else false_positive_rate
    return metric(y_true, y_pred)


def _outcome_given_decision(y_true, y_pred, *, decision):
    selected = np.asarray(y_pred) == decision
    return float(np.mean(np.asarray(y_true)[selected] == 1)) if selected.any() else float("nan")


_METRICS = {
    "selection_rate": selection_rate,
    "tpr": partial(_conditional_prediction_rate, outcome=1),
    "fpr": partial(_conditional_prediction_rate, outcome=0),
    "y_positive_given_r0": partial(_outcome_given_decision, decision=0),
    "y_positive_given_r1": partial(_outcome_given_decision, decision=1),
    "count": count,
}


def _combine(components: dict[str, ComponentResult]) -> CriterionResult:
    states = [component.status for component in components.values()]
    if Status.FAIL in states:
        status = Status.FAIL
    elif Status.UNDEFINED in states:
        status = Status.UNDEFINED
    else:
        status = Status.PASS
    gaps = [component.gap for component in components.values()]
    gap = None if None in gaps else max(gaps)
    return CriterionResult(gap=gap, status=status, components=components)


def _assess(dataset: Dataset, attribute: str, epsilon: float) -> AttributeAssessment:
    data, mapping = dataset.data, dataset.column_map
    y = data[mapping["Y"]].to_numpy(dtype=int)
    r = data[mapping["R"]].to_numpy(dtype=int)
    # Codes avoid sorting mixed string/numeric group labels in Fairlearn.
    # Public labels retain their original type and first-observed order.
    codes, labels = pd.factorize(data[mapping[attribute]], sort=False)
    frame = MetricFrame(metrics=_METRICS, y_true=y, y_pred=r, sensitive_features=codes)
    rates = frame.by_group
    differences = frame.difference(method="between_groups")
    conditions = {
        "selection_rate": (np.ones(len(y), dtype=bool), r == 1),
        "tpr": (y == 1, r == 1),
        "fpr": (y == 0, r == 1),
        "y_positive_given_r0": (r == 0, y == 1),
        "y_positive_given_r1": (r == 1, y == 1),
    }
    components = {}
    for name, (condition, positive) in conditions.items():
        groups = []
        for code, label in enumerate(labels):
            subset = (codes == code) & condition
            denominator = int(rates.loc[code, "count"]) if name == "selection_rate" else int(subset.sum())
            numerator = int((subset & positive).sum())
            groups.append(GroupRate(
                group=_native_group(label), numerator=numerator, denominator=denominator,
                value=float(rates.loc[code, name]) if denominator else None,
                reason=None if denominator else "zero_denominator",
            ))
        reason = None
        if len(groups) < 2:
            reason = "insufficient_groups"
        elif any(group.denominator == 0 for group in groups):
            reason = "zero_denominator"
        if reason:
            components[name] = ComponentResult(gap=None, status=Status.UNDEFINED, groups=groups, reason=reason)
        else:
            gap = float(differences[name])
            passed = gap <= epsilon or isclose(gap, epsilon, rel_tol=0, abs_tol=1e-12)
            components[name] = ComponentResult(gap=gap, status=Status.PASS if passed else Status.FAIL, groups=groups)
    return AttributeAssessment(
        attribute=attribute, column_name=mapping[attribute], epsilon=epsilon,
        independence=_combine({"selection_rate": components["selection_rate"]}),
        separation=_combine({name: components[name] for name in ("tpr", "fpr")}),
        sufficiency=_combine({name: components[name] for name in ("y_positive_given_r0", "y_positive_given_r1")}),
    )


def analyze_attribute(dataset: Dataset, attribute: str, *, epsilon: float = 0.05) -> AttributeAssessment:
    """Assess one mapped A-role; validate that attribute and Y/R first."""
    threshold = _validate_epsilon(epsilon)
    _validate_dataset(dataset, attribute)
    return _assess(dataset, attribute, threshold)


def analyze_dataset(dataset: Dataset, *, epsilon: float = 0.05) -> FairnessReport:
    """Assess every declared A-role independently, not their intersections.

    Validate all selected inputs before computation. Missing roles or invalid
    labels raise DatasetValidationError; empty conditional groups yield UNDEFINED
    statistics. No combined verdict across the three criteria is made.
    """
    threshold = _validate_epsilon(epsilon)
    attributes = _validate_dataset(dataset)
    return FairnessReport(
        epsilon=threshold, n_rows=len(dataset.data),
        attributes={attribute: _assess(dataset, attribute, threshold) for attribute in attributes},
    )
