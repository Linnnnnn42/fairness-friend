from io import BytesIO, StringIO

import pandas as pd
import pytest

from fairness_friend import DatasetValidationError, analyze_dataset, read_dataset
from conftest import TEST_DATA


@pytest.mark.parametrize("format,body", [
    ("csv", "0,1,0\n1,0,1\n"),
    ("whitespace", "  0   1\t0\n1 0    1\n"),
])
@pytest.mark.parametrize("has_header", [True, False])
def test_formats_with_explicit_headers(format, body, has_header):
    if has_header:
        body = ("a,b,c\n" if format == "csv" else "a b c\n") + body
    data = read_dataset(StringIO(body), column_names=["A", "Y", "R"], format=format, has_header=has_header)
    assert data.data.to_dict("list") == {"A": [0, 1], "Y": [1, 0], "R": [0, 1]}
    assert data.column_map == {}


def test_uploaded_bytes_repeated_read_restores_position():
    stream = BytesIO("\ufeffa,b\n0,1\n1,0\n".encode("utf-8"))
    stream.seek(5)
    arguments = dict(column_names=["Group", "Truth"], format="csv", has_header=True)
    first = read_dataset(stream, **arguments)
    second = read_dataset(stream, **arguments)
    pd.testing.assert_frame_equal(first.data, second.data)
    assert stream.tell() == 5
    assert not stream.closed


def test_stream_restored_on_failure():
    stream = BytesIO(b"0,1,2\n")
    stream.seek(2)
    with pytest.raises(DatasetValidationError):
        read_dataset(stream, column_names=["a"], format="csv", has_header=False)
    assert stream.tell() == 2
    assert not stream.closed


def test_simple_file_explicit_index(simple_dataset):
    assert simple_dataset.data.shape == (100, 5)
    assert list(simple_dataset.data.index) == list(range(1, 101))
    assert simple_dataset.data.iloc[0].tolist() == [1, 0, 1, 0, 0]
    assert simple_dataset.column_map["R"] == "Decision"


@pytest.mark.parametrize("filename,width", [("german.data", 21), ("german.data-numeric", 25)])
def test_german_loads_but_analysis_requires_predictions(filename, width):
    names = [f"feature_{i}" for i in range(1, width)] + ["label"]
    data = read_dataset(
        TEST_DATA / "german_credit" / filename, column_names=names,
        format="whitespace", has_header=False, column_map={"A1": names[0], "Y": "label"},
    )
    assert data.data.shape == (1000, width)
    assert set(data.data.label) == {1, 2}
    with pytest.raises(DatasetValidationError) as caught:
        analyze_dataset(data)
    assert caught.value.code == "missing_roles"
    assert caught.value.missing_roles == ["R"]


@pytest.mark.parametrize("body,code", [
    ("", "empty_data"), ("\n\n", "empty_data"),
    ("1,2,3\n", "column_count_mismatch"),
    ("1\n", "column_count_mismatch"),
    ("1,2\n3\n", "column_count_mismatch"),
    ('"unclosed,2\n', "parse_error"),
])
def test_bad_records_are_not_silently_dropped(body, code):
    with pytest.raises(DatasetValidationError) as caught:
        read_dataset(StringIO(body), column_names=["a", "b"], format="csv", has_header=False)
    assert caught.value.code == code


def test_header_only_is_empty():
    with pytest.raises(DatasetValidationError, match="no data"):
        read_dataset(StringIO("a,b\n"), column_names=["a", "b"], format="csv", has_header=True)


@pytest.mark.parametrize("kwargs,code", [
    ({"column_names": ["x", "x"]}, "duplicate_columns"),
    ({"column_names": ["", "b"]}, "invalid_column_names"),
    ({"column_names": "ab"}, "invalid_column_names"),
    ({"index_col": True}, "invalid_index"),
    ({"index_col": -1}, "invalid_index"),
    ({"index_col": 3}, "invalid_index"),
    ({"format": "auto"}, "invalid_format"),
    ({"has_header": "infer"}, "invalid_header"),
])
def test_invalid_options(kwargs, code):
    options = dict(column_names=["a", "b"], format="csv", has_header=False)
    options.update(kwargs)
    with pytest.raises(DatasetValidationError) as caught:
        read_dataset(StringIO("0,1\n"), **options)
    assert caught.value.code == code


def test_quoted_csv_and_unused_missing_feature():
    dataset = read_dataset(
        StringIO('"group,one",1,0,\n"group\ntwo",0,1,x\n'),
        column_names=["group", "y", "r", "unused"], format="csv", has_header=False,
        column_map={"A1": "group", "Y": "y", "R": "r"},
    )
    assert dataset.data.group.tolist() == ["group,one", "group\ntwo"]
    assert pd.isna(dataset.data.unused.iloc[0])
    assert analyze_dataset(dataset).n_rows == 2


def test_index_can_be_in_middle_and_names_apply_after_removal():
    dataset = read_dataset(StringIO("1,row-a,0\n0,row-b,1"),
                           column_names=["y", "r"], index_col=1, format="csv", has_header=False)
    assert dataset.data.index.tolist() == ["row-a", "row-b"]
    assert dataset.data.to_dict("list") == {"y": [1, 0], "r": [0, 1]}
