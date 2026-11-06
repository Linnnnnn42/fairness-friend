"""Strict CSV/whitespace loading without a Streamlit dependency."""

import csv
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import BinaryIO, Literal, TextIO

import pandas as pd

from .models import Dataset
from .validation import DatasetValidationError, _validate_column_names, _validate_mapping


def _read_text(source, encoding: str) -> str:
    try:
        if isinstance(source, (str, PathLike)):
            return Path(source).read_text(encoding=encoding)
        if not hasattr(source, "read"):
            raise DatasetValidationError("invalid_source", "source must be a path or readable file-like object.")
        position = None
        try:
            if hasattr(source, "seekable") and source.seekable():
                position = source.tell()
                source.seek(0)
            content = source.read()
        finally:
            if position is not None:
                source.seek(position)
        if isinstance(content, bytes):
            return content.decode(encoding)
        if isinstance(content, str):
            return content.removeprefix("\ufeff")
        raise DatasetValidationError("invalid_source", "read() must return text or bytes.")
    except (OSError, UnicodeError, LookupError) as exc:
        raise DatasetValidationError("read_error", str(exc)) from exc


def read_dataset(
    source: str | PathLike | BinaryIO | TextIO,
    *,
    column_names: list[str],
    format: Literal["csv", "whitespace"],
    has_header: bool,
    column_map: dict[str, str] | None = None,
    index_col: int | None = None,
    encoding: str = "utf-8-sig",
) -> Dataset:
    """Load a table; column_names names data columns AFTER excluding index_col.

    format and has_header are explicit to avoid losing a sample through inference.
    Incomplete role mappings are allowed for previews; analyze_* validates them.
    Seekable streams are read from the beginning and their position is restored.
    Nonseekable streams are consumed from their current position. Streams are never
    closed by this function. No malformed rows/columns are silently discarded.
    """
    if isinstance(column_names, str):
        raise DatasetValidationError("invalid_column_names", "Provide a sequence of column names, not a string.")
    names = list(column_names)
    _validate_column_names(names)
    mapping = {} if column_map is None else column_map
    _validate_mapping(mapping)
    if format not in ("csv", "whitespace"):
        raise DatasetValidationError("invalid_format", "format must be csv or whitespace.")
    if not isinstance(has_header, bool):
        raise DatasetValidationError("invalid_header", "has_header must be True or False.")
    width = len(names) + (index_col is not None)
    if index_col is not None and (type(index_col) is not int or not 0 <= index_col < width):
        raise DatasetValidationError("invalid_index", "index_col must be a valid non-negative column position.")
    text = _read_text(source, encoding)
    rows = csv.reader(StringIO(text), strict=True) if format == "csv" else (line.split() for line in text.splitlines())
    count = 0
    try:
        for row in rows:
            if not row:
                continue
            count += 1
            if len(row) != width:
                raise DatasetValidationError(
                    "column_count_mismatch", "Every record must match the supplied column count.",
                    record=count, expected=width, actual=len(row),
                )
    except csv.Error as exc:
        raise DatasetValidationError("parse_error", str(exc)) from exc
    if count <= int(has_header):
        raise DatasetValidationError("empty_data", "The file contains no data rows.")
    try:
        # Do not pass names until after parsing: no implicit index or silent truncation.
        data = pd.read_csv(
            StringIO(text), sep="," if format == "csv" else r"\s+",
            header=0 if has_header else None, index_col=False,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise DatasetValidationError("parse_error", str(exc)) from exc
    if data.shape != (count - int(has_header), width):
        raise DatasetValidationError("parse_error", "Parsed shape does not match the validated records.")
    if index_col is not None:
        index = data.iloc[:, index_col].copy()
        data = data.iloc[:, [i for i in range(width) if i != index_col]].copy()
        data.index = pd.Index(index, name=None)
    data.columns = names
    return Dataset(data=data, column_map=mapping.copy())
