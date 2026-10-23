"""TASK-002 工具契约测试。

覆盖 TASK-002 §3 判据 1 的 ⑩ 项（对每个工具）：
① 合法输入返回正确结构化结果
② 空表/零匹配返回 ``{"rows": []}`` 不报错
③ 缺字段/未知列/未知 op/非法类型返回 error 且 code 在枚举内
④ 重复记录正确处理
⑤ 边界数值（0、负数、大数）正确
⑥ 同输入两次调用结果逐字节一致
⑦ 结果可 ``json.dumps``（无 float、无 tuple、日期 ISO）
⑧ stats 的 avg 用 ROUND_HALF_UP（有半值用例）
⑨ organize 的并列按 tiebreak 次级键稳定
⑩ format 的 CSV 空字段为 ""、含引号/逗号/中文正确解析

外加本单专属的铁律断言：
- 工具不 import ``tasks``（换数据生成策略不碰工具）
- 工具不返回拼好的答案（不泄漏 ground truth）
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from zh_agent_bench.tools import (
    ORDERS,
    STATS_OPS,
    TOOL_CONTRACT_VERSION,
    TOOL_REGISTRY,
    ErrorCode,
    check_contract,
    csv_to_json,
    get_tool,
    is_error,
    table_organize,
    table_query,
    table_stats,
)

#: A 场景的四个表/格式工具。B 场景的 ``knowledge.search`` 由 test_retrieval.py 独立验收，
#: 注册表里多它一个；但本档其余用例钉的是「A 场景 agent 拿到哪几个工具」，显式列出而不是
#: 抄注册表 —— 抄了就等于把「恰好是哪几个」这条判据自己给注销了。
A_SCENE_TOOLS = ("table.query", "table.stats", "table.organize", "format.csv_to_json")
B_SCENE_TOOLS = ("knowledge.search",)
TOOLS = A_SCENE_TOOLS + B_SCENE_TOOLS

#: 中文办公样例表（与 TASK-001 build.py 的 initial_data 结构对齐）。
EMP_TABLE = [
    {"姓名": "王小明", "部门": "研发部", "城市": "北京", "入职日期": "2026-01-15"},
    {"姓名": "李静", "部门": "市场部", "城市": "上海", "入职日期": "2026-02-03"},
    {"姓名": "张伟", "部门": "销售部", "城市": "广州", "入职日期": "2026-03-08"},
]
SALES_TABLE = [
    {"姓名": "王小明", "销售额": 100_00},
    {"姓名": "李静", "销售额": 200_00},
    {"姓名": "张伟", "销售额": 200_00},
]


def _err_code(result: dict) -> str:
    assert set(result) == {"error"}, f"错误结构必须只有 error 键，实际: {sorted(result)}"
    err = result["error"]
    assert set(err) == {"code", "message"}, f"error 必须有 code 与 message，实际: {sorted(err)}"
    assert isinstance(err["message"], str) and err["message"]
    return err["code"]


def _assert_legal_code(result: dict) -> str:
    """断言是合法错误结构，返回 code 供用例继续比对具体取值。"""
    legal = {c.value for c in ErrorCode}
    code = _err_code(result)
    assert code in legal, f"error code 必须在枚举内: {code}"
    return code


def _json_roundtrip(result: dict) -> str:
    """能 json.dumps，且序列化后再解析等价 —— 契约⑦。"""
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def _assert_json_clean(result: dict) -> None:
    """契约⑦：无 float、无 tuple，日期保持 ISO 字符串。"""
    def walk(node: object) -> None:
        if isinstance(node, float):
            pytest.fail(f"结果里出现原生 float: {node}")
        if isinstance(node, tuple):
            pytest.fail(f"结果里出现 tuple（应序列化为 list）: {node}")
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(result)


# ======================================================================
# 注册表 / 契约冻结
# ======================================================================


def test_registry_has_exactly_four_tools():
    """A 场景四个工具一个不多一个不少（QA 工具是 B 场景的，另有一档断言）。"""
    assert sorted(TOOL_REGISTRY) == sorted(TOOLS) == sorted(A_SCENE_TOOLS + B_SCENE_TOOLS)


def test_get_tool_returns_spec():
    for tool_id in TOOLS:
        spec = get_tool(tool_id)
        assert spec.id == tool_id
        assert spec.version == TOOL_CONTRACT_VERSION == "1.0"
        assert callable(spec.fn)
        assert spec.input_schema["type"] == "object"
        assert spec.output_schema["type"] == "object"


def test_get_tool_unknown_id_raises():
    with pytest.raises(ValueError):
        get_tool("table.nope")


def test_tool_spec_is_frozen():
    spec = get_tool("table.query")
    with pytest.raises(Exception):
        spec.id = "hacked"  # type: ignore[misc]


def test_tools_do_not_import_tasks():
    """铁律：工具不依赖 build.py 的内部函数/词表，只依赖公开 schema。"""
    tools_dir = Path(__file__).resolve().parents[1] / "src" / "zh_agent_bench" / "tools"
    for py in sorted(tools_dir.glob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "tasks" not in mod, f"{py.name} 不许 import {mod}"
                assert "build" not in mod, f"{py.name} 不许 import {mod}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "tasks" not in alias.name, f"{py.name} 不许 import {alias.name}"


def test_tools_have_no_third_party_imports():
    """运行时零新依赖：tools/ 只能 import stdlib 与本包。"""
    stdlib_ok = {
        "ast", "collections", "collections.abc", "csv", "dataclasses", "decimal",
        "enum", "io", "json", "math", "re", "sys", "typing", "__future__",
    }
    tools_dir = Path(__file__).resolve().parents[1] / "src" / "zh_agent_bench" / "tools"
    for py in sorted(tools_dir.glob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in stdlib_ok, f"{py.name} 引入非 stdlib: {alias.name}"
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top = node.module.split(".")[0]
                assert top in stdlib_ok, f"{py.name} 引入非 stdlib: {node.module}"


def test_check_contract_rejects_bad_input():
    with pytest.raises(ValueError):
        check_contract("table.query", {"table": EMP_TABLE})  # 缺 filters
    with pytest.raises(ValueError):
        check_contract("table.stats", {"table": "not-a-list", "column": "x", "op": "sum"})


def test_check_contract_accepts_good_input():
    out = check_contract("table.query", {"table": EMP_TABLE, "filters": {"姓名": "李静"}})
    assert len(out["rows"]) == 1
    out = check_contract("table.stats", {"table": SALES_TABLE, "column": "销售额", "op": "sum"})
    assert out == {"result": 500_00}


def test_check_contract_passes_the_tool_error_through_verbatim():
    """error 分支是**合法输出**，不是「输出不合契约」—— 原样返回，一个字都不加工。

    TASK-007 之前不分叉：工具吐的 ``{"error": {"code": "unknown_column", ...}}`` 不含 ``result``，
    被 output 校验拦下抛 ``ValueError("缺少必填字段 'result'")``，**真正的错误码当场被吞掉**。
    LLM 拿到的是一句「你的输出不对」，于是拿同样错的列名重试到耗尽步数
    （RESULT-006 §4.2 的 ``stats-0004`` 现场）。契约校验要抓的是「工具行为与声明不符」，
    而 error 恰恰是工具**声明过**的行为。
    """
    out = check_contract("table.stats", {"table": SALES_TABLE, "column": "不存在", "op": "sum"})
    assert out == table_stats({"table": SALES_TABLE, "column": "不存在", "op": "sum"})
    assert out["error"]["code"] == "unknown_column"
    assert "不存在" in out["error"]["message"], "message 里的具体信息也不该被吃掉"


def test_check_contract_still_rejects_genuinely_malformed_output():
    """放行 error **不等于**放行一切：输入不合约照样抛（那是调用方的 bug）。"""
    with pytest.raises(ValueError, match="输入不符合契约"):
        check_contract("table.stats", {"table": SALES_TABLE, "column": "销售额"})  # 缺 op
    with pytest.raises(ValueError, match="输入不符合契约"):
        check_contract("table.stats", {"table": "not-a-list", "column": "销售额", "op": "sum"})


# ======================================================================
# table.query
# ======================================================================


def test_query_returns_matching_rows():
    out = table_query({"table": EMP_TABLE, "filters": {"姓名": "李静"}})
    assert out == {"rows": [{"姓名": "李静", "部门": "市场部", "城市": "上海", "入职日期": "2026-02-03"}]}


def test_query_multiple_filters_are_and():
    out = table_query({"table": EMP_TABLE, "filters": {"部门": "研发部", "城市": "北京"}})
    assert len(out["rows"]) == 1
    out = table_query({"table": EMP_TABLE, "filters": {"部门": "研发部", "城市": "上海"}})
    assert out == {"rows": []}


def test_query_zero_match_is_not_error():
    """零匹配返回 rows: []，**不报错** —— 没查到是合法答案。"""
    out = table_query({"table": EMP_TABLE, "filters": {"姓名": "查无此人"}})
    assert out == {"rows": []}
    assert not is_error(out)


def test_query_empty_table_returns_empty_rows():
    assert table_query({"table": [], "filters": {"姓名": "王小明"}}) == {"rows": []}


def test_query_duplicate_rows_all_returned():
    """重复记录要全部返回，不能去重 —— agent 需要看到全部匹配。"""
    dupes = [
        {"姓名": "王小明", "部门": "研发部"},
        {"姓名": "王小明", "部门": "市场部"},
        {"姓名": "王小明", "部门": "研发部"},
    ]
    out = table_query({"table": dupes, "filters": {"姓名": "王小明"}})
    assert len(out["rows"]) == 3
    assert [r["部门"] for r in out["rows"]] == ["研发部", "市场部", "研发部"]


def test_query_unknown_column_errors():
    out = table_query({"table": EMP_TABLE, "filters": {"不存在": "x"}})
    assert _assert_legal_code(out) == ErrorCode.UNKNOWN_COLUMN.value


def test_query_invalid_types_error():
    assert _assert_legal_code(table_query({"table": "not-list", "filters": {}})) \
        == ErrorCode.INVALID_INPUT.value
    assert _assert_legal_code(table_query({"table": [1, 2], "filters": {}})) \
        == ErrorCode.INVALID_INPUT.value
    assert _assert_legal_code(table_query({"table": EMP_TABLE, "filters": "x"})) \
        == ErrorCode.INVALID_INPUT.value
    assert _assert_legal_code(table_query({"filters": {}})) == ErrorCode.INVALID_INPUT.value


def test_query_does_not_leak_answer():
    """铁律：query 返回行，不返回「部门是 XX」这种拼好的答案。"""
    out = table_query({"table": EMP_TABLE, "filters": {"姓名": "李静"}})
    assert set(out) == {"rows"}
    assert isinstance(out["rows"][0], dict)


# ======================================================================
# table.stats
# ======================================================================


def test_stats_ops():
    def s(op: str) -> object:
        return table_stats({"table": SALES_TABLE, "column": "销售额", "op": op})["result"]

    assert s("sum") == 500_00
    assert s("max") == 200_00
    assert s("min") == 100_00
    assert s("count") == 3


def test_stats_avg_is_two_decimal_string():
    """契约：avg 返回十进制字符串（与 TASK-001 AnswerKind.NUMBER 的规范值对齐）。"""
    out = table_stats({"table": SALES_TABLE, "column": "销售额", "op": "avg"})
    assert isinstance(out["result"], str)
    assert out["result"] == "166.67"


def test_stats_avg_uses_round_half_up():
    """判据⑧：半值用例必须走 HALF_UP，不能是银行家舍入。

    100+200+200=500 分，500/3/100 = 1.6666... 元，量化到两位是 1.67。
    换成 HALF_EVEN 结果相同，故另造一个真正落在 .xx5 的用例：
    1 分 + 2 分 + 0 分 = 3 分，平均 0.01 元 -> "0.01"；
    构造 cents 总量使得 /N/100 恰好为 0.125 元 -> HALF_UP 得 0.13、HALF_EVEN 得 0.12。
    """
    # 3 行，每行 125 分 => 平均 125 分 = 1.25 元，两位小数就是 1.25（无分歧）
    # 改用 2 行：125 + 125 = 250/2 = 125 分 = 1.25 元 —— 仍无分歧。
    # 真正的 .xx5 分歧点：平均值 = x.xx5 元，即 cents 总和 / N / 100 的第三位小数为 5。
    # 取 N=4 行、每行 1 分 -> 平均 0.01 元（无分歧）；
    # 取 N=8 行、每行 1 分 -> 0.0125 元 -> HALF_UP = 0.01，HALF_EVEN = 0.01（仍同）
    # 直接用能算出的最小分歧：N=8，cents=10 -> 10/8/100 = 0.0125 元 -> 量到 0.01，同。
    # 换到分位：让平均分带 .5 分即 元位 .005 -> 两位小数量化时正好在 .xx5。
    # N=2，cents=1 与 cents=2 -> 平均 1.5 分 = 0.015 元 -> HALF_UP 0.02 / HALF_EVEN 0.02
    # N=2，cents=1 与 cents=0 -> 0.5 分 = 0.005 元 -> HALF_UP 0.01 / HALF_EVEN 0.00  <- 就是它
    table = [{"销售额": 1}, {"销售额": 0}]
    out = table_stats({"table": table, "column": "销售额", "op": "avg"})["result"]
    assert out == "0.01", f"应为 ROUND_HALF_UP 的 0.01，实际 {out}"
    # 独立复算，钉死舍入方向
    avg = (Decimal(1) / Decimal(2)) / Decimal(100)
    assert avg == Decimal("0.005")
    assert str(avg.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)) == "0.01"
    assert str(avg.quantize(Decimal("0.01"))) == "0.00", "若默认舍入也是 0.01，本用例失去区分力"


def test_stats_avg_zero_rows_is_empty_table():
    out = table_stats({"table": [], "column": "销售额", "op": "avg"})
    assert _assert_legal_code(out) == ErrorCode.EMPTY_TABLE.value


def test_stats_count_on_empty_table_is_zero():
    """count 数行数，空表是合法的 0，不是错误。"""
    assert table_stats({"table": [], "column": "销售额", "op": "count"}) == {"result": 0}


def test_stats_count_ignores_column_existence():
    """count 与列无关，不该报 unknown_column。"""
    assert table_stats({"table": EMP_TABLE, "column": "根本没有这列", "op": "count"}) == {"result": 3}


def test_stats_boundary_values():
    """判据⑤：0、负数、大数。"""
    def s(op: str, table: list[dict]) -> object:
        return table_stats({"table": table, "column": "v", "op": op})["result"]

    assert s("sum", [{"v": 0}, {"v": 0}]) == 0
    assert s("min", [{"v": -500}, {"v": 100}]) == -500
    assert s("max", [{"v": -500}, {"v": 100}]) == 100
    assert s("sum", [{"v": 10**18}]) == 10**18
    assert s("avg", [{"v": 0}, {"v": 0}]) == "0.00"
    assert s("avg", [{"v": -100}, {"v": -200}]) == "-1.50"


def test_stats_unknown_column():
    out = table_stats({"table": SALES_TABLE, "column": "不存在", "op": "sum"})
    assert _assert_legal_code(out) == ErrorCode.UNKNOWN_COLUMN.value


def test_stats_unknown_op():
    out = table_stats({"table": SALES_TABLE, "column": "销售额", "op": "median"})
    assert _assert_legal_code(out) == ErrorCode.UNKNOWN_OP.value


def test_stats_invalid_input():
    assert _assert_legal_code(table_stats({"table": SALES_TABLE, "column": "销售额"})) \
        == ErrorCode.INVALID_INPUT.value
    assert _assert_legal_code(table_stats({"table": SALES_TABLE, "column": 1, "op": "sum"})) \
        == ErrorCode.INVALID_INPUT.value
    assert _assert_legal_code(
        table_stats({"table": [{"v": "abc"}], "column": "v", "op": "sum"})
    ) == ErrorCode.INVALID_INPUT.value


def test_stats_op_set_is_exactly_specified():
    assert STATS_OPS == ("avg", "sum", "count", "max", "min")


def test_stats_does_not_leak_answer():
    """铁律：stats 返回数值，不拼「平均数是 XX 元」。"""
    out = table_stats({"table": SALES_TABLE, "column": "销售额", "op": "avg"})
    assert set(out) == {"result"}


# ======================================================================
# table.organize
# ======================================================================

TIE_TABLE = [
    {"姓名": "王小明", "销售额": 200_00},
    {"姓名": "李静", "销售额": 200_00},
    {"姓名": "张伟", "销售额": 300_00},
    {"姓名": "赵敏", "销售额": 100_00},
]


def test_organize_desc_with_tiebreak():
    """判据⑨：并列按 tiebreak 次级键（姓名升序）稳定。"""
    out = table_organize({
        "table": TIE_TABLE, "sort_by": "销售额", "order": "desc",
        "limit": 3, "tiebreak": {"姓名": "asc"},
    })
    assert [r["姓名"] for r in out["rows"]] == ["张伟", "李静", "王小明"]


def test_organize_asc():
    out = table_organize({"table": TIE_TABLE, "sort_by": "销售额", "order": "asc", "limit": 2})
    assert [r["姓名"] for r in out["rows"]] == ["赵敏", "王小明"]


def test_organize_limit_larger_than_rows_takes_all():
    out = table_organize({"table": TIE_TABLE, "sort_by": "销售额", "order": "desc", "limit": 99})
    assert len(out["rows"]) == len(TIE_TABLE)


def test_organize_limit_zero():
    assert table_organize({"table": TIE_TABLE, "sort_by": "销售额", "limit": 0}) == {"rows": []}


def test_organize_no_limit_takes_all():
    out = table_organize({"table": TIE_TABLE, "sort_by": "销售额"})
    assert len(out["rows"]) == 4
    assert out["rows"][0]["姓名"] == "张伟", "默认 order 应为 desc"


def test_organize_empty_table():
    assert table_organize({"table": [], "sort_by": "销售额"}) == {"rows": []}


def test_organize_stability_is_deterministic():
    """判据⑥ + ⑨：同输入两次逐字节一致，且并列项顺序不漂移。"""
    payload = {"table": TIE_TABLE, "sort_by": "销售额", "order": "desc",
               "limit": 3, "tiebreak": {"姓名": "asc"}}
    assert _json_roundtrip(table_organize(payload)) == _json_roundtrip(table_organize(payload))


def test_organize_duplicate_rows_preserved():
    dupes = [{"姓名": "同名", "销售额": 1}, {"姓名": "同名", "销售额": 1}]
    out = table_organize({"table": dupes, "sort_by": "销售额", "tiebreak": {"姓名": "asc"}})
    assert len(out["rows"]) == 2


def test_organize_mixed_types_do_not_crash():
    """混合类型列不能抛 TypeError —— 工具必须给结构化错误或排序结果，不能炸。"""
    mixed = [{"姓名": "a", "v": 1}, {"姓名": "b", "v": "x"}, {"姓名": "c", "v": None}]
    out = table_organize({"table": mixed, "sort_by": "v", "order": "desc"})
    assert "rows" in out
    assert len(out["rows"]) == 3


def test_organize_errors():
    assert _assert_legal_code(
        table_organize({"table": TIE_TABLE, "sort_by": "不存在"})
    ) == ErrorCode.UNKNOWN_COLUMN.value
    assert _assert_legal_code(
        table_organize({"table": TIE_TABLE, "sort_by": "销售额", "order": "sideways"})
    ) == ErrorCode.UNKNOWN_OP.value
    assert _assert_legal_code(
        table_organize({"table": TIE_TABLE, "sort_by": "销售额", "tiebreak": {"姓名": "up"}})
    ) == ErrorCode.UNKNOWN_OP.value
    assert _assert_legal_code(
        table_organize({"table": TIE_TABLE, "sort_by": "销售额", "tiebreak": {"不存在": "asc"}})
    ) == ErrorCode.UNKNOWN_COLUMN.value
    assert _assert_legal_code(table_organize({"table": TIE_TABLE})) \
        == ErrorCode.INVALID_INPUT.value
    assert _assert_legal_code(
        table_organize({"table": TIE_TABLE, "sort_by": "销售额", "limit": -1})
    ) == ErrorCode.INVALID_INPUT.value
    assert _assert_legal_code(
        table_organize({"table": TIE_TABLE, "sort_by": "销售额", "limit": "三"})
    ) == ErrorCode.INVALID_INPUT.value


def test_organize_orders_set_is_exactly_specified():
    assert ORDERS == ("asc", "desc")


def test_organize_does_not_leak_answer():
    """铁律：organize 返回排序后的行，不返回「前 3 名姓名列表」。"""
    out = table_organize({"table": TIE_TABLE, "sort_by": "销售额", "order": "desc",
                          "limit": 3, "tiebreak": {"姓名": "asc"}})
    assert set(out) == {"rows"}
    assert all(isinstance(r, dict) for r in out["rows"])


# ======================================================================
# format.csv_to_json
# ======================================================================

CSV_SIMPLE = "姓名,部门,城市\n王小明,研发部,北京\n李静,市场部,上海\n"


def test_csv_to_json_simple():
    out = csv_to_json({"csv": CSV_SIMPLE, "table_name": "员工信息表"})
    assert out["json"] == {
        "表名": "员工信息表",
        "rows": [
            {"姓名": "王小明", "部门": "研发部", "城市": "北京"},
            {"姓名": "李静", "部门": "市场部", "城市": "上海"},
        ],
    }


def test_csv_to_json_without_table_name():
    out = csv_to_json({"csv": CSV_SIMPLE})
    assert set(out["json"]) == {"rows"}


def test_csv_empty_field_is_empty_string_not_null():
    """判据⑩ + TASK-001 §2.5：空字段为 ""，不是 null。"""
    out = csv_to_json({"csv": "姓名,城市\n王小明,\n"})
    assert out["json"]["rows"][0]["城市"] == ""
    assert "None" not in _json_roundtrip(out)


def test_csv_handles_quotes_and_commas():
    csv_text = '姓名,备注\n王小明,"含,逗号"\n李静,"含""引号"""\n'
    out = csv_to_json({"csv": csv_text})
    rows = out["json"]["rows"]
    assert rows[0]["备注"] == "含,逗号"
    assert rows[1]["备注"] == '含"引号"'


def test_csv_handles_chinese_and_bomless_utf8():
    out = csv_to_json({"csv": "商品,金额\n智能会议平板,1234\n激光打印机,5678\n"})
    rows = out["json"]["rows"]
    assert rows[0]["商品"] == "智能会议平板"
    assert rows[1]["金额"] == "5678"


def test_csv_crlf_input_parsed_identically():
    """CRLF 与 LF 是同一份数据 —— 解析结果不得随换行符漂移。"""
    lf = csv_to_json({"csv": CSV_SIMPLE})
    crlf = csv_to_json({"csv": CSV_SIMPLE.replace("\n", "\r\n")})
    assert lf == crlf


def test_csv_empty_input():
    assert csv_to_json({"csv": ""}) == {"json": {"rows": []}}
    assert csv_to_json({"csv": "\n"}) == {"json": {"rows": []}}


def test_csv_short_row_gets_empty_string():
    out = csv_to_json({"csv": "a,b,c\n1,2\n"})
    assert out["json"]["rows"] == [{"a": "1", "b": "2", "c": ""}]


def test_csv_errors():
    assert _assert_legal_code(csv_to_json({"csv": 123})) == ErrorCode.INVALID_INPUT.value
    assert _assert_legal_code(csv_to_json({})) == ErrorCode.INVALID_INPUT.value
    assert _assert_legal_code(csv_to_json({"csv": "a", "table_name": 5})) \
        == ErrorCode.INVALID_INPUT.value


def test_csv_does_not_leak_answer():
    """铁律：format 返回 JSON 对象，不返回「转好了，结果是……」这种话。"""
    out = csv_to_json({"csv": CSV_SIMPLE, "table_name": "员工信息表"})
    assert set(out) == {"json"}
    assert isinstance(out["json"], dict)


# ======================================================================
# 通用：契约⑥⑦⑩ 对全部 4 个工具
# ======================================================================


ALL_CALLS: list[tuple[str, dict]] = [
    ("table.query", {"table": EMP_TABLE, "filters": {"姓名": "王小明"}}),
    ("table.query", {"table": [], "filters": {"姓名": "x"}}),
    ("table.stats", {"table": SALES_TABLE, "column": "销售额", "op": "avg"}),
    ("table.stats", {"table": SALES_TABLE, "column": "销售额", "op": "sum"}),
    ("table.stats", {"table": [], "column": "销售额", "op": "count"}),
    ("table.organize", {"table": TIE_TABLE, "sort_by": "销售额", "order": "desc",
                        "limit": 3, "tiebreak": {"姓名": "asc"}}),
    ("format.csv_to_json", {"csv": CSV_SIMPLE, "table_name": "员工信息表"}),
]


@pytest.mark.parametrize("tool_id,payload", ALL_CALLS)
def test_deterministic_byte_identical(tool_id, payload):
    """判据⑥：同输入两次调用结果逐字节一致。"""
    spec = get_tool(tool_id)
    assert _json_roundtrip(spec.fn(payload)) == _json_roundtrip(spec.fn(payload))


@pytest.mark.parametrize("tool_id,payload", ALL_CALLS)
def test_result_is_json_serializable(tool_id, payload):
    """判据⑦：结果可 json.dumps，无 float、无 tuple。"""
    result = get_tool(tool_id).fn(payload)
    _assert_json_clean(result)
    assert json.loads(json.dumps(result, ensure_ascii=False)) == result


@pytest.mark.parametrize("tool_id,payload", ALL_CALLS)
def test_input_not_mutated(tool_id, payload):
    """纯函数：不得改入参。"""
    before = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    get_tool(tool_id).fn(payload)
    assert json.dumps(payload, ensure_ascii=False, sort_keys=True) == before


def test_dates_stay_iso_strings():
    """判据⑦：日期保持 ISO 8601 字符串原样透传。"""
    out = table_query({"table": EMP_TABLE, "filters": {"姓名": "王小明"}})
    assert out["rows"][0]["入职日期"] == "2026-01-15"
    assert re_fullmatch_iso(out["rows"][0]["入职日期"])


def re_fullmatch_iso(value: str) -> bool:
    import re

    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))


def test_all_error_codes_are_from_enum():
    """判据③：任何错误路径的 code 都必须在枚举内。"""
    bad_calls = [
        ("table.query", {"table": "x", "filters": {}}),
        ("table.query", {"table": EMP_TABLE, "filters": {"不存在": 1}}),
        ("table.stats", {"table": SALES_TABLE, "column": "销售额", "op": "median"}),
        ("table.stats", {"table": SALES_TABLE, "column": "销售额", "op": "avg"}),
        ("table.organize", {"table": TIE_TABLE, "sort_by": "不存在"}),
        ("format.csv_to_json", {"csv": 1}),
    ]
    for tool_id, payload in bad_calls:
        if tool_id == "table.stats" and payload.get("op") == "avg":
            continue  # 空表分支在别处测
        result = get_tool(tool_id).fn(payload)
        _assert_legal_code(result)
        assert is_error(result)


def test_tools_import_standalone_without_uv_run():
    """判据 3：import 成功（不依赖 pytest 环境）。"""
    proc = subprocess.run(
        [sys.executable, "-c", "import zh_agent_bench.tools; print('imported')"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "imported" in proc.stdout