# fairness-friend

[English](README.md) | 简体中文

一个基于 **Fairlearn 0.15** 的小型公平性分析库，版本 **0.0.1**。读取已有真实标签与模型决策的数据，逐个敏感属性计算 Independence、Separation 和 Sufficiency，返回可供 Streamlit 使用的结构化对象。

需要 Python 3.11+。库不依赖 Streamlit，不训练模型、不生成预测。

## 使用 uv

在仓库根目录运行：

```sh
uv sync --locked
uv run pytest -q -W error
uv build
```

另一个使用 uv 的前端项目可以添加这个本地库：

```sh
uv add --editable /path/to/fairness-friend
```

运行时依赖为 Fairlearn、Pandas 和 NumPy；Scikit-learn 由 Fairlearn 引入。`uv.lock` 固定开发环境的实际版本，发布包则使用 `pyproject.toml` 中的兼容范围。

## 快速开始

下面的路径相对于仓库根目录；示例 CSV 第一列是导出的行号，所以需要 `index_col=0`。

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

# 每个分组包含 numerator、denominator、value 和 reason。
tpr_groups = sex.separation.components["tpr"].groups

# 返回普通字典；未定义值变成 JSON null，不输出 NaN。
import json
payload = json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False)
```

DataFrame 保留用户指定的列名；`column_map` 的方向始终是 **角色 → 实际列名**。敏感属性角色为 `A1`、`A2` 等，`Y` 为真实标签，`R` 为预测决策。各角色使用不同列；普通特征列可以保留、不参与计算。批量分析按 A 编号排序，独立分析每个属性，不计算属性交叉群体。

## 数据读取

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

| 参数 | 约定 |
| --- | --- |
| `source` | 本地路径、文本流、二进制流或 Streamlit `UploadedFile` |
| `column_names` | 排除显式索引列后，按文件顺序指定全部数据列名；非空、不重复 |
| `format` | `"csv"` 为逗号分隔；`"whitespace"` 为连续空格/tab 分隔，不依赖扩展名 |
| `has_header` | 必须显式指定；True 替换已有表头，False 保留第一行数据 |
| `column_map` | 可以先省略或只指定一部分，分析时再校验完整性 |
| `index_col` | 原始文件中索引列的零基位置，例如 0；默认没有索引列 |
| `encoding` | 默认兼容 UTF-8 BOM；可以显式指定其他编码 |

返回 `Dataset(data: pd.DataFrame, column_map: dict[str, str])`。读入器包装 `pd.read_csv`，额外检查每条记录的列数，防止隐式索引、丢列或忽略坏行。CSV 支持带引号字段和带引号的换行；空白文本面向普通字段，不支持带空格的复杂引号字段。缺失值识别采用 Pandas 默认规则。

可定位文件流会从头读取并恢复原位置；库不关闭调用方传入的流。不可定位的流从当前位置读取并消耗内容。文件先在内存中完整读取，首版不提供分块处理。

如果已有 DataFrame，可以直接构造对象：

```python
from fairness_friend import Dataset

dataset = Dataset(
    data=prepared_dataframe,
    column_map={"A1": "gender", "Y": "actual", "R": "predicted"},
)
```

Dataset 允许调用方在分析前修改数据和映射。分析函数不修改原始 DataFrame。Y/R 只接受数值 0/1（含等价布尔或浮点值），1 表示调用方选定的正类。字符串标签、1/2 标签、预测概率等应先显式转换；库不会猜测正类或决策阈值。

敏感属性支持字符串、布尔值和有限数值类别；数值按其不同取值分组。年龄等连续特征应由调用方按明确规则分箱。只比较文件中实际出现的群体，不推断完全未出现的类别。

## 三类指标与 epsilon

令 Δ 表示所有群体中最大概率减去最小概率。二元群体时等于两组绝对差。

| 标准 | `components` | 标准的 `gap` |
| --- | --- | --- |
| Independence | `selection_rate`: P(R=1 \| A) | Δ(selection_rate) |
| Separation | `tpr`: P(R=1 \| Y=1,A)；`fpr`: P(R=1 \| Y=0,A) | max(Δ(TPR), Δ(FPR)) |
| Sufficiency | `y_positive_given_r0`: P(Y=1 \| R=0,A)；`y_positive_given_r1`: P(Y=1 \| R=1,A) | 两个分量差距的最大值 |

Sufficiency 同时检查两种预测结果。R=0 分量是 FOR，而不是 NPV；R=1 分量是 PPV。

`epsilon` 默认 0.05，范围 [0,1]，含义是 **5 个百分点的绝对差距**，不是相对百分比或显著性水平。判定前不舍入；等于阈值也通过，边界使用 `abs_tol=1e-12, rel_tol=0` 补偿浮点误差。

- `PASS`：必要概率都有定义，gap ≤ epsilon。
- `FAIL`：已定义的必要分量中至少一个超过 epsilon。
- `UNDEFINED`：未发现失败分量，但有必要分量因零分母或不足两个群体无法比较。

基础分量中任何群体的条件分母为零，整个分量为 UNDEFINED，不忽略该群体。Separation/Sufficiency 汇总时 FAIL 优先于 UNDEFINED。任何必要分量未定义时，完整 `gap=None`；因此可能出现 `status=FAIL, gap=None`，表示已证实某一分量失败，但完整最大差距未知。

前端可以把 PASS/FAIL 显示为 FAIR(PASS)/UNFAIR(FAIL)，含义限定于当前数据、指标和 epsilon。库不生成三类标准合并后的整体公平结论。

计算使用 Fairlearn 的 `MetricFrame`、分组基础指标和 `difference(method="between_groups")`；Sufficiency 使用两个自定义基础指标。参见 [Fairlearn 评估流程](https://fairlearn.org/v0.15/user_guide/assessment/perform_fairness_assessment.html) 和 [常见指标](https://fairlearn.org/v0.15/user_guide/assessment/common_fairness_metrics.html)。

## 返回对象

```text
FairnessReport(epsilon, n_rows, attributes)
  attributes["A1"] → AttributeAssessment(attribute, column_name, epsilon,
                                         independence, separation, sufficiency)
    每个标准 → CriterionResult(gap, status, components)
      每个分量 → ComponentResult(gap, status, groups, reason)
        每个群体 → GroupRate(group, numerator, denominator, value, reason)
```

所有结果对象都有 `to_dict()`。群体以列表记录，保留类别类型和首次出现顺序；例如数字 `0` 与字符串 `"0"` 可以区别。数值等价类别（如 `0` 与 `0.0`）按 Pandas 的分组语义视为同一组。

未定义原因码为 `zero_denominator` 或 `insufficient_groups`。分组值仍可用于展示：单群体情况下可能有合法概率，但组间差距未定义。

## 输入错误与缺少 R

所有分析入口都会校验选中的敏感属性、Y 和 R。缺少 R 属于输入错误，**不是 FAIL，也不是统计量 UNDEFINED**。

```python
from fairness_friend import DatasetValidationError

try:
    report = analyze_dataset(dataset)
except DatasetValidationError as error:
    error.code             # 例如 "missing_roles"
    error.missing_roles    # 例如 ["R"]
    error.missing_columns  # 已映射但文件中不存在的列名
    error.to_dict()        # {"code": ..., "message": ..., "details": ...}
```

主要分析错误码：`missing_roles`、`missing_columns`、`missing_values`、`non_binary_labels`、`duplicate_roles`、`invalid_epsilon`、`invalid_group`。未指定任何敏感属性时 `missing_roles=["A"]`。读取错误还包括 `empty_data`、`column_count_mismatch`、`parse_error` 和 `read_error`；列数错误的 `details.record` 为从 1 开始的逻辑记录编号，包含表头。

单属性分析只要求选中的 A 与 Y/R 有效；批量分析会在开始计算之前校验所有声明的 A。不会自动删除缺失样本或填补标签。

German Credit 示例：

```python
german = read_dataset(
    "test_data/german_credit/german.data",
    column_names=[f"feature_{i}" for i in range(1, 21)] + ["label"],
    format="whitespace",
    has_header=False,
    column_map={"A1": "feature_9", "Y": "label"},
)
assert german.data.shape == (1000, 21)
# analyze_dataset(german) 会抛出 missing_roles，missing_roles 为 ["R"]。
```

两份 German Credit 文件都缺少预测，末列标签为 1/2。需要调用方补充真实模型预测，并明确映射为 0/1。原始数据的 `feature_9` 是性别/婚姻状态的组合类别，不等同于纯性别列。库不会默认 R=Y、随机生成 R 或自动训练模型。

## Streamlit 接入示例

Streamlit 安装在前端项目中。以下片段展示 CSV 样例的读取、可选预测列和异常捕获；其他数据需要前端提供对应的列名和读取配置。

```python
import streamlit as st
from fairness_friend import DatasetValidationError, analyze_dataset, read_dataset

uploaded = st.file_uploader("上传样例数据", type=["csv", "txt"])
if uploaded is not None:
    try:
        dataset = read_dataset(
            uploaded,
            column_names=["Sex", "Age", "Ethnicity", "Outcome", "Decision"],
            format="csv", has_header=True, index_col=0,
        )
        st.dataframe(dataset.data)
        prediction_column = st.selectbox(
            "模型预测 R", [None, *dataset.data.columns],
            format_func=lambda value: "请选择预测列" if value is None else value,
        )
        dataset.column_map = {
            "A1": "Sex", "A2": "Age", "A3": "Ethnicity", "Y": "Outcome",
        }
        if prediction_column is not None:
            dataset.column_map["R"] = prediction_column
        else:
            st.info("缺少模型预测结果，请补充或选择预测列。")

        epsilon = st.slider("允许的绝对概率差", 0.0, 1.0, 0.05, 0.01)
        if st.button("分析", disabled=prediction_column is None):
            report = analyze_dataset(dataset, epsilon=epsilon)
            st.dataframe([
                {
                    "属性": result.column_name,
                    "Independence": result.independence.status.value,
                    "Separation": result.separation.status.value,
                    "Sufficiency": result.sufficiency.status.value,
                }
                for result in report.attributes.values()
            ])
            st.json(report.to_dict())
    except DatasetValidationError as error:
        if "R" in error.missing_roles:
            st.error("缺少模型预测结果，请补充预测列。")
        else:
            st.error(str(error))
```

前端提前提示，库再次校验；其他调用方也无法绕过缺列检查。[UploadedFile 文档](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader) 说明它支持通用文件流接口。

## 验收与范围

测试使用 `test_data` 中三个原始文件，以及独立手算的混淆矩阵和边界样例。默认 epsilon 下 CSV 的预期汇总为：

| 属性 | Independence | Separation | Sufficiency |
| --- | --- | --- | --- |
| Sex | PASS | FAIL | FAIL |
| Age | FAIL | FAIL | PASS |
| Ethnicity | FAIL | FAIL | FAIL |

首版只处理二元 Y/R、离散敏感属性和绝对差距，不包含 ratio、ROC、交叉群体、自动分箱、置信区间、样本权重、模型训练或公平性缓解。

设计背景见 [0.0.1 规划文档](docs/v0.0.1-plan.md)。
