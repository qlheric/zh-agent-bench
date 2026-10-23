"""工具注册表 —— 冻结工具契约的唯一真相源。

设计要点：
- ``ToolSpec`` 把「稳定 ID + 版本 + 实现 + 输入 schema + 输出 schema」绑成不可变整体，
  harness（TASK-003）按 ``TOOL_REGISTRY`` 给 agent 分发工具，judge 消费工具输出。
- 错误结构与 ``ErrorCode`` 定义在 ``_errors.py``（独立成模块以打断导入环），这里 re-export。
- 本模块只依赖 stdlib，**不 import ``tasks``**（TASK-002 铁律：换数据生成策略不碰工具）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._errors import ERROR_MESSAGES, ErrorCode, is_error, make_error

#: 全部工具的契约版本。改动任一工具的输入/输出语义都必须 bump。
TOOL_CONTRACT_VERSION = "1.0"


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的完整契约。frozen：注册后不可篡改，避免 harness 与 judge 各拿一份改过的。"""

    id: str
    version: str
    fn: Callable[[dict], dict]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


# --------------------------------------------------------------------------------------
# 极简 JSON-schema 校验器（只覆盖本项目用到的子集；不引第三方依赖）
#
# 支持：type（object/string/integer/array/any）、required、properties、items。
# 够不够由下面 4 个工具的 input_schema 自己保证 —— 判据要求「尽量简」。
# --------------------------------------------------------------------------------------

_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "any": lambda v: True,
}


def validate_payload(schema: dict[str, Any], payload: Any) -> list[str]:
    """按 schema 校验 payload，返回错误描述列表（空列表 = 通过）。"""
    errors: list[str] = []
    expected_type = schema.get("type", "any")
    check = _TYPE_CHECKS.get(expected_type)
    if check is not None and not check(payload):
        return [f"类型应为 {expected_type}，实际为 {type(payload).__name__}"]

    if expected_type == "object":
        for field in schema.get("required", []):
            if field not in payload:
                errors.append(f"缺少必填字段 {field!r}")
        for field, sub in schema.get("properties", {}).items():
            if field in payload:
                errors.extend(f"{field}: {e}" for e in validate_payload(sub, payload[field]))
    elif expected_type == "array":
        item_schema = schema.get("items")
        if item_schema is not None:
            for i, item in enumerate(payload):
                errors.extend(f"[{i}]: {e}" for e in validate_payload(item_schema, item))
    return errors


def check_contract(tool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """按注册表校验输入 → 调用工具 → 校验输出。返回工具的实际输出。

    这条路径就是 TASK-002 判据 3 的机械校验方式：契约不是写在文档里的承诺，
    而是「注册表说的」与「代码做的」必须每一条对得上。

    **error 分支单独放行**（findings §七.3）：``output_schema`` 描述的是**成功**输出
    （``{"result": ...}`` / ``{"rows": ...}``），而工具失败时按 ``_errors.make_error``
    返回 ``{"error": {"code": ..., "message": ...}}`` —— 那是另一个分支，不该拿成功 schema 去判。
    早期这里不分叉，工具返回 error 时先被 output 校验拦下抛
    「缺少必填字段 'result'」，**把真正的错误码（``unknown_column`` 等）吞掉了**：
    agent 拿到的只有一个「你的输出不对」，于是拿同样错的参数重试到耗尽步数
    （RESULT-006 §4.2 的 ``stats-0004`` 现场）。契约校验的价值是抓「工具行为与声明不符」，
    而 error 形状恰恰是工具**声明过**的行为 —— 拿 schema 去判它只会产出误导性的错误信息。
    """
    spec = get_tool(tool_id)
    problems = validate_payload(spec.input_schema, payload)
    if problems:
        raise ValueError(f"{tool_id} 输入不符合契约: {problems}")
    result = spec.fn(payload)
    if is_error(result):
        # 原样透传：code 与 message 是 agent 自我纠正的唯一线索，包装任何一层都会把它弄丢。
        return result
    out_problems = validate_payload(spec.output_schema, result)
    if out_problems:
        raise ValueError(f"{tool_id} 输出不符合契约: {out_problems}")
    return result


def get_tool(tool_id: str) -> ToolSpec:
    """按 ID 取工具；未知 ID 抛 ValueError（这是调用方的 bug，不是工具的错误）。"""
    try:
        return TOOL_REGISTRY[tool_id]
    except KeyError:
        raise ValueError(
            f"未知工具 ID: {tool_id!r}；已注册: {sorted(TOOL_REGISTRY)}"
        ) from None


# --------------------------------------------------------------------------------------
# 构造注册表
# 依赖方向：table/format -> _errors（单向）。因此这里 import 它们不会成环。
# --------------------------------------------------------------------------------------

from . import format as _format
from . import retrieval as _retrieval
from . import table as _table

TOOL_REGISTRY: dict[str, ToolSpec] = {
    "table.query": ToolSpec(
        id="table.query",
        version=TOOL_CONTRACT_VERSION,
        fn=_table.table_query,
        input_schema={
            "type": "object",
            "required": ["table", "filters"],
            "properties": {
                "table": {"type": "array", "items": {"type": "object"}},
                "filters": {"type": "object"},  # {列名: 精确等值}
            },
        },
        output_schema={
            "type": "object",
            "required": ["rows"],
            "properties": {"rows": {"type": "array", "items": {"type": "object"}}},
        },
    ),
    "table.stats": ToolSpec(
        id="table.stats",
        version=TOOL_CONTRACT_VERSION,
        fn=_table.table_stats,
        input_schema={
            "type": "object",
            "required": ["table", "column", "op"],
            "properties": {
                "table": {"type": "array", "items": {"type": "object"}},
                "column": {"type": "string"},
                "op": {"type": "string"},  # avg|sum|count|max|min
            },
        },
        output_schema={
            "type": "object",
            "required": ["result"],
            "properties": {"result": {"type": "any"}},
        },
    ),
    "table.organize": ToolSpec(
        id="table.organize",
        version=TOOL_CONTRACT_VERSION,
        fn=_table.table_organize,
        input_schema={
            "type": "object",
            "required": ["table", "sort_by"],
            "properties": {
                "table": {"type": "array", "items": {"type": "object"}},
                "sort_by": {"type": "string"},
                "order": {"type": "string"},           # desc|asc，默认 desc
                "limit": {"type": "integer"},         # 大于行数则全取
                "tiebreak": {"type": "object"},       # {次级列: asc|desc}
            },
        },
        output_schema={
            "type": "object",
            "required": ["rows"],
            "properties": {"rows": {"type": "array", "items": {"type": "object"}}},
        },
    ),
    "format.csv_to_json": ToolSpec(
        id="format.csv_to_json",
        version=TOOL_CONTRACT_VERSION,
        fn=_format.csv_to_json,
        input_schema={
            "type": "object",
            "required": ["csv"],
            "properties": {
                "csv": {"type": "string"},
                # 可选：给 JSON 顶层补一个「表名」键，与 TASK-001 format 任务的
                # ground_truth 结构（{"表名": ..., "rows": [...]}）对齐。
                "table_name": {"type": "string"},
            },
        },
        output_schema={
            "type": "object",
            "required": ["json"],
            "properties": {"json": {"type": "object"}},
        },
    ),
    # B 场景（TASK-008）：知识检索。与表工具有两处**刻意**不同 ——
    # 零命中返回空 hits（合法结果，「证据不足」是 B 场景要考的能力），
    # 参数用错才走 error；hits 带 score 让 agent 看得出哪几篇最相关。
    "knowledge.search": ToolSpec(
        id="knowledge.search",
        version=TOOL_CONTRACT_VERSION,
        fn=_retrieval.knowledge_search,
        input_schema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "knowledge": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["doc_id", "title", "text"],
                        "properties": {
                            "doc_id": {"type": "string"},
                            "title": {"type": "string"},
                            "text": {"type": "string"},
                        },
                    },
                },
                "top_k": {"type": "integer"},  # 默认 5，上限 10（见 retrieval.py）
            },
        },
        output_schema={
            "type": "object",
            "required": ["hits"],
            "properties": {
                "hits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["doc_id", "title", "text", "score"],
                        "properties": {
                            "doc_id": {"type": "any"},   # 原样透传，允许 None（脏文档）
                            "title": {"type": "any"},
                            "text": {"type": "any"},
                            "score": {"type": "integer"},
                        },
                    },
                }
            },
        },
    ),
}