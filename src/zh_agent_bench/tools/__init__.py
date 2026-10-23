"""工具集：4 个确定性中文办公工具 + 冻结契约。

harness（TASK-003）从 :data:`TOOL_REGISTRY` 分发工具，judge 消费工具的结构化输出。
"""

from ._errors import ErrorCode, is_error, make_error
from .format import csv_to_json
from .registry import (
    TOOL_CONTRACT_VERSION,
    TOOL_REGISTRY,
    ToolSpec,
    check_contract,
    get_tool,
    validate_payload,
)
from .table import ORDERS, STATS_OPS, table_organize, table_query, table_stats

__all__ = [
    "TOOL_CONTRACT_VERSION",
    "TOOL_REGISTRY",
    "STATS_OPS",
    "ORDERS",
    "ErrorCode",
    "ToolSpec",
    "check_contract",
    "get_tool",
    "is_error",
    "make_error",
    "validate_payload",
    "table_query",
    "table_stats",
    "table_organize",
    "csv_to_json",
]