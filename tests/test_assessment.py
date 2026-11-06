import json
from fractions import Fraction

import pandas as pd
import pytest
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

from fairness_friend import Dataset, Status, analyze_attribute, analyze_dataset


def test_reference_csv_expected_results_and_fairlearn(simple_dataset):
    report = analyze_dataset(simple_dataset)
    expected = {
        "A1": [(Status.PASS, 0.006802721088435374), (Status.FAIL, 0.055), (Status.FAIL, 0.19696969696969696)],
        "A2": [(Status.FAIL, 0.5467148884870404), (Status.FAIL, 0.5590277777777778), (Status.PASS, 0.025735294117647058)],
        "A3": [(Status.FAIL, 0.18888888888888888), (Status.FAIL, 0.29444444444444445), (Status.FAIL, 0.20714285714285716)],
    }
    assert report.n_rows == 100
    data = simple_dataset.data
    for role, expected_criteria in expected.items():
        attribute = report.attributes[role]
        assert attribute == analyze_attribute(simple_dataset, role)
        for actual, (status, gap) in zip(
            [attribute.independence, attribute.separation, attribute.sufficiency], expected_criteria
        ):
            assert actual.status == status
            assert actual.gap == pytest.approx(gap)
        kwargs = dict(y_true=data.Outcome, y_pred=data.Decision, sensitive_features=data[attribute.column_name])
        assert attribute.independence.gap == pytest.approx(demographic_parity_difference(**kwargs))
        assert attribute.separation.gap == pytest.approx(equalized_odds_difference(**kwargs, agg="worst_case"))


def test_hand_computed_rates_and_counts(balanced_dataset):
    result = analyze_attribute(balanced_dataset, "A1")
    assert result.independence.gap == 0
    assert result.separation.gap == 0.25
    assert result.sufficiency.gap == 0.25
    for criterion, key, expected in [
        (result.independence, "selection_rate", [(4, 8), (4, 8)]),
        (result.separation, "tpr", [(3, 4), (2, 4)]),
        (result.separation, "fpr", [(1, 4), (2, 4)]),
        (result.sufficiency, "y_positive_given_r0", [(1, 4), (2, 4)]),
        (result.sufficiency, "y_positive_given_r1", [(3, 4), (2, 4)]),
    ]:
        for group, (numerator, denominator) in zip(criterion.components[key].groups, expected):
            assert (group.numerator, group.denominator) == (numerator, denominator)
            assert group.value == float(Fraction(numerator, denominator))


def test_three_groups_use_max_minus_min(balanced_dataset):
    extra = pd.DataFrame([(2, 1, 1)] + [(2, 1, 0)] * 3 + [(2, 0, 1)] * 3 + [(2, 0, 0)],
                         columns=balanced_dataset.data.columns)
    balanced_dataset.data = pd.concat([balanced_dataset.data, extra], ignore_index=True)
    result = analyze_attribute(balanced_dataset, "A1")
    assert result.separation.gap == 0.5
    assert result.sufficiency.gap == 0.5
    assert len(result.independence.components["selection_rate"].groups) == 3


def test_threshold_equality_and_just_above():
    rows = [(group, i % 2, int(i < positives)) for group, positives in [(0, 10), (1, 11)] for i in range(20)]
    data = Dataset(pd.DataFrame(rows, columns=["a", "y", "r"]), {"A1": "a", "Y": "y", "R": "r"})
    assert analyze_attribute(data, "A1", epsilon=0.05).independence.status == Status.PASS
    assert analyze_attribute(data, "A1", epsilon=0.05 - 1e-9).independence.status == Status.FAIL


def test_epsilon_endpoints(balanced_dataset):
    strict = analyze_attribute(balanced_dataset, "A1", epsilon=0)
    assert strict.independence.status == Status.PASS
    assert strict.separation.status == Status.FAIL
    loose = analyze_attribute(balanced_dataset, "A1", epsilon=1)
    assert all(c.status == Status.PASS for c in [loose.independence, loose.separation, loose.sufficiency])


def test_one_group_is_not_automatically_fair(balanced_dataset):
    balanced_dataset.data = balanced_dataset.data[balanced_dataset.data.group == 0]
    result = analyze_attribute(balanced_dataset, "A1")
    for criterion in [result.independence, result.separation, result.sufficiency]:
        assert criterion.status == Status.UNDEFINED
        assert criterion.gap is None
        assert all(c.reason == "insufficient_groups" for c in criterion.components.values())


def test_missing_conditional_group_is_retained(balanced_dataset):
    data = balanced_dataset.data
    balanced_dataset.data = data[~((data.group == 1) & (data.truth == 1))]
    result = analyze_attribute(balanced_dataset, "A1")
    tpr = result.separation.components["tpr"]
    assert tpr.status == Status.UNDEFINED
    assert tpr.gap is None
    assert [(g.group, g.denominator) for g in tpr.groups] == [(0, 4), (1, 0)]
    assert tpr.groups[1].value is None
    assert tpr.groups[1].reason == "zero_denominator"
    # FPR disparity still proves failure, though the full separation gap is unknown.
    assert result.separation.components["fpr"].status == Status.FAIL
    assert result.separation.status == Status.FAIL
    assert result.separation.gap is None


@pytest.mark.parametrize("decision", [0, 1])
def test_constant_prediction_leaves_sufficiency_undefined(balanced_dataset, decision):
    balanced_dataset.data["decision"] = decision
    report = analyze_dataset(balanced_dataset)
    result = report.attributes["A1"]
    assert result.independence.status == Status.PASS
    assert result.separation.status == Status.PASS
    assert result.sufficiency.status == Status.UNDEFINED
    assert result.sufficiency.components[f"y_positive_given_r{1-decision}"].gap is None
    encoded = json.dumps(report.to_dict(), allow_nan=False)
    assert "null" in encoded
    assert type(report.to_dict()["attributes"]["A1"]["sufficiency"]["status"]) is str


@pytest.mark.parametrize("outcome", [0, 1])
def test_constant_truth_has_undefined_separation(balanced_dataset, outcome):
    balanced_dataset.data["truth"] = outcome
    result = analyze_attribute(balanced_dataset, "A1")
    assert result.separation.status == Status.UNDEFINED
    assert result.sufficiency.status == Status.PASS


def test_mixed_group_labels_remain_distinct_and_input_is_unchanged(balanced_dataset):
    balanced_dataset.data["group"] = balanced_dataset.data.group.map({0: 0, 1: "0"})
    before = balanced_dataset.data.copy(deep=True)
    result = analyze_attribute(balanced_dataset, "A1")
    groups = result.independence.components["selection_rate"].groups
    assert [g.group for g in groups] == [0, "0"]
    pd.testing.assert_frame_equal(before, balanced_dataset.data)
    json.dumps(result.to_dict(), allow_nan=False)


def test_unused_categorical_levels_do_not_create_phantom_groups(balanced_dataset):
    balanced_dataset.data["group"] = pd.Categorical(balanced_dataset.data.group, categories=[0, 1, 2])
    groups = analyze_attribute(balanced_dataset, "A1").independence.components["selection_rate"].groups
    assert [g.group for g in groups] == [0, 1]


def test_arbitrary_column_order_and_index_do_not_change_result(balanced_dataset):
    expected = analyze_attribute(balanced_dataset, "A1")
    balanced_dataset.data = balanced_dataset.data[["decision", "group", "truth"]]
    balanced_dataset.data.index = ["repeated"] * len(balanced_dataset.data)
    assert analyze_attribute(balanced_dataset, "A1") == expected
