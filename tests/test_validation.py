import json

import pandas as pd
import pytest

from fairness_friend import Dataset, DatasetValidationError, analyze_attribute, analyze_dataset


@pytest.mark.parametrize("epsilon", [-0.1, 1.1, float("nan"), float("inf"), "0.05", True, None])
def test_invalid_epsilon(balanced_dataset, epsilon):
    with pytest.raises(DatasetValidationError) as caught:
        analyze_dataset(balanced_dataset, epsilon=epsilon)
    assert caught.value.code == "invalid_epsilon"


@pytest.mark.parametrize("role", ["A1", "Y", "R"])
def test_missing_role_is_structured(balanced_dataset, role):
    del balanced_dataset.column_map[role]
    with pytest.raises(DatasetValidationError) as caught:
        analyze_attribute(balanced_dataset, "A1")
    assert caught.value.code == "missing_roles"
    assert caught.value.missing_roles == [role]
    assert json.loads(json.dumps(caught.value.to_dict()))["details"]["missing_roles"] == [role]


def test_mapped_prediction_column_does_not_exist(balanced_dataset):
    balanced_dataset.column_map["R"] = "not_uploaded"
    with pytest.raises(DatasetValidationError) as caught:
        analyze_dataset(balanced_dataset)
    assert caught.value.code == "missing_columns"
    assert caught.value.missing_columns == ["not_uploaded"]


@pytest.mark.parametrize("column", ["group", "truth", "decision"])
def test_missing_analysis_values_are_rejected(balanced_dataset, column):
    balanced_dataset.data.loc[0, column] = float("nan")
    with pytest.raises(DatasetValidationError) as caught:
        analyze_dataset(balanced_dataset)
    assert caught.value.code == "missing_values"


@pytest.mark.parametrize("value", [2, -1, 0.2, "1", "approved"])
@pytest.mark.parametrize("role", ["Y", "R"])
def test_binary_labels_are_required(balanced_dataset, role, value):
    column = balanced_dataset.column_map[role]
    balanced_dataset.data[column] = balanced_dataset.data[column].astype(object)
    balanced_dataset.data.loc[0, column] = value
    with pytest.raises(DatasetValidationError) as caught:
        analyze_dataset(balanced_dataset)
    assert caught.value.code == "non_binary_labels"


def test_roles_must_be_distinct(balanced_dataset):
    balanced_dataset.column_map["R"] = "truth"
    with pytest.raises(DatasetValidationError) as caught:
        analyze_dataset(balanced_dataset)
    assert caught.value.code == "duplicate_roles"


def test_single_attribute_does_not_require_other_attributes(balanced_dataset):
    balanced_dataset.column_map["A2"] = "absent"
    assert analyze_attribute(balanced_dataset, "A1").attribute == "A1"
    with pytest.raises(DatasetValidationError) as caught:
        analyze_dataset(balanced_dataset)
    assert caught.value.missing_columns == ["absent"]


def test_no_sensitive_attributes(balanced_dataset):
    del balanced_dataset.column_map["A1"]
    with pytest.raises(DatasetValidationError) as caught:
        analyze_dataset(balanced_dataset)
    assert caught.value.missing_roles == ["A"]


def test_empty_dataframe_is_input_error():
    dataset = Dataset(pd.DataFrame(columns=["a", "y", "r"]), {"A1": "a", "Y": "y", "R": "r"})
    with pytest.raises(DatasetValidationError) as caught:
        analyze_dataset(dataset)
    assert caught.value.code == "empty_data"
