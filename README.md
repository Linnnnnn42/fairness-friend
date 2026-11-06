# fairness-friend

English | [简体中文](README.zh-CN.md)

A small fairness assessment library powered by **Fairlearn 0.15**, version **0.0.1**. It reads data containing ground-truth labels and model decisions, evaluates Independence, Separation, and Sufficiency for each sensitive attribute, and returns structured results for use in a Streamlit frontend.

Requires Python 3.11+. The library does not depend on Streamlit, train models, or generate predictions.

## Using uv

Run from the repository root:

```sh
uv sync --locked
uv run pytest -q -W error
uv build
```

A separate frontend project using uv can add this local library:

```sh
uv add --editable /path/to/fairness-friend
```

Runtime dependencies are Fairlearn, Pandas, and NumPy; Fairlearn brings in Scikit-learn. `uv.lock` pins the development environment, while published packages use the compatibility ranges in `pyproject.toml`.

## Quick start

Paths below are relative to the repository root. The sample CSV has an exported row-number column, so use `index_col=0`.

```python
from fairness_friend import read_dataset, analyze_attribute, analyze_dataset

dataset = read_dataset(
    "test_data/simple_data/model_outcome.csv",
    column_names=["Sex", "Age", "Ethnicity", "Outcome", "Decision"],
    format="csv",
    has_header=True,
    index_col=0,
    column_map={
        "A1": "Sex",
        "A2": "Age",
        "A3": "Ethnicity",
        "Y": "Outcome",
        "R": "Decision",
    },
)

sex = analyze_attribute(dataset, "A1", epsilon=0.05)
report = analyze_dataset(dataset, epsilon=0.05)

assert sex.independence.status == "PASS"
assert sex.separation.status == "FAIL"
assert report.n_rows == 100

# Each group includes numerator, denominator, value, and reason.
tpr_groups = sex.separation.components["tpr"].groups

# Plain dictionaries; undefined values become JSON null rather than NaN.
import json
payload = json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False)
```

The DataFrame retains your column names. `column_map` always maps **logical roles → actual column names**. Sensitive attributes use `A1`, `A2`, etc.; `Y` is ground truth and `R` is the predicted decision. Each role must refer to a distinct column. Other feature columns may remain in the table without participating in the assessment. Batch analysis processes attributes in numeric A-role order, independently rather than as intersecting groups.

## Loading data

```python
read_dataset(
    source,
    *,
    column_names,
    format,
    has_header,
    column_map=None,
    index_col=None,
    encoding="utf-8-sig",
)
```

| Parameter | Contract |
| --- | --- |
| `source` | A local path, text stream, binary stream, or Streamlit `UploadedFile` |
| `column_names` | Names for all data columns in file order, after excluding the explicit index column; nonempty and unique |
| `format` | `"csv"` for comma-separated data; `"whitespace"` for consecutive spaces/tabs, regardless of extension |
| `has_header` | Required explicitly: True replaces the existing header; False preserves the first row as data |
| `column_map` | May be omitted or incomplete during loading; completeness is checked before analysis |
| `index_col` | Zero-based position of the index column in the original file, such as 0; no index column by default |
| `encoding` | Supports a UTF-8 BOM by default; other encodings can be specified explicitly |

Returns `Dataset(data: pd.DataFrame, column_map: dict[str, str])`. The reader wraps `pd.read_csv` and checks the number of fields in every record to prevent implicit indexing, lost columns, or silently skipped malformed rows. CSV supports quoted fields and quoted newlines. Whitespace input is intended for simple fields, not complex quoted fields containing spaces. Missing-value detection follows Pandas defaults.

Seekable streams are read from the beginning and restored to their original position. The library never closes caller-owned streams. Nonseekable streams are consumed from their current position. Files are read fully into memory; chunked processing is not provided in this release.

If you already have a DataFrame, construct a dataset directly:

```python
from fairness_friend import Dataset

dataset = Dataset(
    data=prepared_dataframe,
    column_map={"A1": "gender", "Y": "actual", "R": "predicted"},
)
```

Callers may modify the Dataset's data and mapping before analysis. Analysis functions do not modify the input DataFrame. Y/R accept only numeric 0/1 values, including equivalent boolean or floating-point values; 1 denotes the caller's chosen positive class. Explicitly convert string labels, 1/2 labels, or prediction probabilities first. The library does not infer the positive class or decision threshold.

Sensitive attributes support string, boolean, and finite numeric categories. Distinct numeric values form separate groups. Callers should bin continuous features such as age using explicit rules. Only groups observed in the file are compared; entirely absent categories are not inferred.

## The three criteria and epsilon

Let Δ denote the maximum group probability minus the minimum. With two groups, this is their absolute difference.

| Criterion | `components` | Criterion `gap` |
| --- | --- | --- |
| Independence | `selection_rate`: P(R=1 \| A) | Δ(selection_rate) |
| Separation | `tpr`: P(R=1 \| Y=1,A); `fpr`: P(R=1 \| Y=0,A) | max(Δ(TPR), Δ(FPR)) |
| Sufficiency | `y_positive_given_r0`: P(Y=1 \| R=0,A); `y_positive_given_r1`: P(Y=1 \| R=1,A) | Maximum of the two component gaps |

Sufficiency checks both possible decisions. The R=0 component is FOR, not NPV; the R=1 component is PPV.

`epsilon` defaults to 0.05 and must be in [0,1]. The default means an **absolute gap of 5 percentage points**, not a relative percentage or significance level. Values are not rounded before comparison. Equality passes, with `abs_tol=1e-12, rel_tol=0` used to compensate for floating-point error at the boundary.

- `PASS`: All required probabilities are defined and gap ≤ epsilon.
- `FAIL`: At least one defined required component exceeds epsilon.
- `UNDEFINED`: No component has failed, but a required component cannot be compared because of a zero denominator or fewer than two groups.

If any group's conditional denominator is zero, the entire base component is UNDEFINED; that group is not dropped. When combining Separation/Sufficiency components, FAIL takes precedence over UNDEFINED. If any required component is undefined, the full `gap=None`. Consequently, `status=FAIL, gap=None` means a component has established failure, but the complete maximum gap remains unknown.

A frontend may display PASS/FAIL as FAIR(PASS)/UNFAIR(FAIL), limited to the current data, criterion, and epsilon. The library does not combine all three criteria into an overall fairness verdict.

Computation uses Fairlearn's `MetricFrame`, group-level base metrics, and `difference(method="between_groups")`. Sufficiency uses two custom base metrics. See [Performing a Fairness Assessment](https://fairlearn.org/v0.15/user_guide/assessment/perform_fairness_assessment.html) and [Common fairness metrics](https://fairlearn.org/v0.15/user_guide/assessment/common_fairness_metrics.html).

## Result objects

```text
FairnessReport(epsilon, n_rows, attributes)
  attributes["A1"] → AttributeAssessment(attribute, column_name, epsilon,
                                         independence, separation, sufficiency)
    Each criterion → CriterionResult(gap, status, components)
      Each component → ComponentResult(gap, status, groups, reason)
        Each group → GroupRate(group, numerator, denominator, value, reason)
```

Every result object provides `to_dict()`. Groups are stored as lists, preserving label types and first-observed order; numeric `0` and string `"0"` can remain distinct. Numerically equivalent categories, such as `0` and `0.0`, form the same group under Pandas grouping semantics.

Undefined reason codes are `zero_denominator` or `insufficient_groups`. Group values may still be displayed: a single group can have a valid probability while its between-group gap is undefined.

## Input errors and missing R

Every analysis entry point validates the selected sensitive attributes, Y, and R. Missing R is an input error, **not FAIL or a statistically UNDEFINED result**.

```python
from fairness_friend import DatasetValidationError

try:
    report = analyze_dataset(dataset)
except DatasetValidationError as error:
    error.code             # For example, "missing_roles"
    error.missing_roles    # For example, ["R"]
    error.missing_columns  # Mapped column names absent from the table
    error.to_dict()        # {"code": ..., "message": ..., "details": ...}
```

Main analysis error codes: `missing_roles`, `missing_columns`, `missing_values`, `non_binary_labels`, `duplicate_roles`, `invalid_epsilon`, and `invalid_group`. If no sensitive attribute is mapped, `missing_roles=["A"]`. Loading errors also include `empty_data`, `column_count_mismatch`, `parse_error`, and `read_error`. For column-count errors, `details.record` is the one-based logical record number, including the header.

Single-attribute analysis requires only the selected A and Y/R to be valid. Batch analysis validates all declared A roles before computing results. Missing samples are not automatically dropped and labels are not imputed.

German Credit example:

```python
german = read_dataset(
    "test_data/german_credit/german.data",
    column_names=[f"feature_{i}" for i in range(1, 21)] + ["label"],
    format="whitespace",
    has_header=False,
    column_map={"A1": "feature_9", "Y": "label"},
)
assert german.data.shape == (1000, 21)
# analyze_dataset(german) raises missing_roles with missing_roles=["R"].
```

Both German Credit files lack predictions and use 1/2 labels in their final column. Callers must supply actual model predictions and explicitly map labels to 0/1. In the original data, `feature_9` combines sex and marital status; it is not a sex-only column. The library does not default to R=Y, generate random R, or train a model automatically.

## Streamlit integration example

Install Streamlit in the frontend project. This snippet demonstrates loading the sample CSV, selecting a prediction column, and handling errors. Other datasets require their own column names and loading options.

```python
import streamlit as st
from fairness_friend import DatasetValidationError, analyze_dataset, read_dataset

uploaded = st.file_uploader("Upload sample data", type=["csv", "txt"])
if uploaded is not None:
    try:
        dataset = read_dataset(
            uploaded,
            column_names=["Sex", "Age", "Ethnicity", "Outcome", "Decision"],
            format="csv", has_header=True, index_col=0,
        )
        st.dataframe(dataset.data)
        prediction_column = st.selectbox(
            "Model predictions R", [None, *dataset.data.columns],
            format_func=lambda value: "Select a prediction column" if value is None else value,
        )
        dataset.column_map = {
            "A1": "Sex", "A2": "Age", "A3": "Ethnicity", "Y": "Outcome",
        }
        if prediction_column is not None:
            dataset.column_map["R"] = prediction_column
        else:
            st.info("Model predictions are missing. Add or select a prediction column.")

        epsilon = st.slider("Allowed absolute probability gap", 0.0, 1.0, 0.05, 0.01)
        if st.button("Analyze", disabled=prediction_column is None):
            report = analyze_dataset(dataset, epsilon=epsilon)
            st.dataframe([
                {
                    "Attribute": result.column_name,
                    "Independence": result.independence.status.value,
                    "Separation": result.separation.status.value,
                    "Sufficiency": result.sufficiency.status.value,
                }
                for result in report.attributes.values()
            ])
            st.json(report.to_dict())
    except DatasetValidationError as error:
        if "R" in error.missing_roles:
            st.error("Model predictions are missing. Add a prediction column.")
        else:
            st.error(str(error))
```

The frontend provides early feedback, while the library validates again so other callers cannot bypass missing-column checks. The [UploadedFile documentation](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader) describes its file-like interface.

## Verification and scope

Tests use the three original files under `test_data`, independently calculated confusion matrices, and boundary cases. With the default epsilon, the sample CSV has these expected results:

| Attribute | Independence | Separation | Sufficiency |
| --- | --- | --- | --- |
| Sex | PASS | FAIL | FAIL |
| Age | FAIL | FAIL | PASS |
| Ethnicity | FAIL | FAIL | FAIL |

This release supports binary Y/R, discrete sensitive attributes, and absolute gaps. It does not include ratios, ROC curves, intersecting groups, automatic binning, confidence intervals, sample weights, model training, or fairness mitigation.

For design background, see the [0.0.1 design proposal (Chinese)](docs/v0.0.1-plan.md).
