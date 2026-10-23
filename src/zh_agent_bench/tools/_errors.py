"""统一错误结构 —— 独立成模块是为了打断导入环。

``registry.py`` 需要 ``table``/``format`` 来填 ``TOOL_REGISTRY``，而这两个工具模块
需要 ``ErrorCode`` 来造错误。若三者互相 import 就会形成环：
``registry -> table -> registry``（表还没定义完就被读）。

把「错误」这一层单独放在这里，依赖方向就变成单向的
``table/format -> _errors``，``registry -> table/format``，永不成环。
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """工具错误 code 枚举。**固定取值，不得自创**，harness 靠它做机器可判定的分支。"""

    INVALID_INPUT = "invalid_input"      # 输入缺必填字段 / 类型错误
    UNKNOWN_COLUMN = "unknown_column"    # 指定的列不存在
    UNKNOWN_OP = "unknown_op"            # 统计/排序算子不合法
    EMPTY_TABLE = "empty_table"          # 表为空且该 op 无定义结果（如空表求平均）


#: 各 code 的默认中文说明。工具可覆盖 message，但 code 必须来自本枚举。
ERROR_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.INVALID_INPUT: "输入缺少必填字段或类型不正确",
    ErrorCode.UNKNOWN_COLUMN: "指定的列在表中不存在",
    ErrorCode.UNKNOWN_OP: "不支持的算子",
    ErrorCode.EMPTY_TABLE: "表为空，该统计无定义结果",
}


def make_error(code: ErrorCode, message: str | None = None) -> dict[str, Any]:
    """构造统一错误结构。``code`` 必须是 ErrorCode，杜绝自由文本。"""
    return {"error": {"code": code.value, "message": message or ERROR_MESSAGES[code]}}


def is_error(result: Any) -> bool:
    """判断工具返回值是否为错误结构（harness 的统一分支）。"""
    return isinstance(result, dict) and "error" in result