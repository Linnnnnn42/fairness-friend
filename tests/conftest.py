from pathlib import Path

import pandas as pd
import pytest

from fairness_friend import Dataset, read_dataset

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"


@pytest.fixture
def simple_dataset():
    return read_dataset(
        TEST_DATA / "simple_data/model_outcome.csv",
        column_names=["Sex", "Age", "Ethnicity", "Outcome", "Decision"],
        format="csv", has_header=True, index_col=0,
        column_map={"A1": "Sex", "A2": "Age", "A3": "Ethnicity", "Y": "Outcome", "R": "Decision"},
    )


@pytest.fixture
def balanced_dataset():
    # Group 0: TP=3, FN=1, FP=1, TN=3; group 1: all four counts are 2.
    rows = []
    for group, counts in [(0, [3, 1, 1, 3]), (1, [2, 2, 2, 2])]:
        for (y, r), count in zip([(1, 1), (1, 0), (0, 1), (0, 0)], counts):
            rows.extend([(group, y, r)] * count)
    return Dataset(pd.DataFrame(rows, columns=["group", "truth", "decision"]),
                   {"A1": "group", "Y": "truth", "R": "decision"})
