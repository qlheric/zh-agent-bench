"""三个表工具：``table.query`` / ``table.stats`` / ``table.organize`。

共享契约（TASK-002 §2.5）：
- 金额一律整数分（``int``）；统计运算走 ``Decimal``，``avg`` 结果序列化为两位小数字符串。
  **禁止原生 float 进入结果** —— 浮点会毁掉 ``decimal_exact`` 判分。
- 日期是 ISO 8601 字符串，缺失值是 ``None``（→ JSON ``null``）。
- 行是 ``list[dict]``，不发明 row_id；表里已有的 ``id`` 字段原样保留。
- 错误统一 ``{"error": {"code": ..., "message": ...}}``，code 取自 :class:`ErrorCode`。
- **只返回结构化结果，绝不返回拼好的答案。** 例如 organize 返回排序后的「行」，
  不返回「前 3 名的姓名列表」——后者就是 ground truth，泄漏了评测就废了。
- 纯函数：同输入 → 同输出，逐字节一致；无随机、不读时钟、不联网、不改入参。
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from ._errors import ErrorCode, make_error

#: table.stats 支持的算子。
STATS_OPS: tuple[str, ...] = ("avg", "sum", "count", "max", "min")

#: table.organize 支持的排序方向。
ORDERS: tuple[str, ...] = ("asc", "desc")

#: 统计「列」必须是什么类型。bool 是 int 的子类，要显式排除。
_NUMERIC = (int,)


def _is_number(value: Any) -> bool:
    """数值判定：只接受 ``int``（整数分）。float 一律不认 —— 契约禁止 float 进入金额。"""
    return isinstance(value, _NUMERIC) and not isinstance(value, bool)


def _columns(table: list[dict]) -> set[str]:
    """表的列名集合 = 所有行键的并集。表为空时返回空集。"""
    return {key for row in table for key in row}


def _require_table(payload: Any) -> list[dict] | None:
    """校验 table 是 list[dict]；不合法返回 None（调用方转成 invalid_input）。"""
    if not isinstance(payload, list):
        return None
    return payload if all(isinstance(row, dict) for row in payload) else None


def _has_column(table: list[dict], column: str) -> bool:
    """列是否存在。**只按已出现的行判断**：空表时任何列都算存在，
    让它走到 empty_table / count=0 分支，而不是误报 unknown_column。"""
    if not table:
        return True
    return column in _columns(table)


def _sortable(value: Any) -> tuple[int, Any]:
    """排序键。数字与字符串不可直接比较，同表里混类型会炸。

    返回 ``(类型秩, 值)``：数字统一降为字符串比较会丢顺序（"9" > "10"），
    所以数字与数字比、数字秩 0；非数字秩 1 按字符串比（``None`` 排在最前）。
    """
    if value is None:
        return (2, "")
    if isinstance(value, bool):
        return (1, str(value))
    if isinstance(value, (int, float, Decimal)):
        return (0, value)
    return (1, str(value))


# --------------------------------------------------------------------------------------
# table.query
# --------------------------------------------------------------------------------------


def table_query(payload: dict[str, Any]) -> dict[str, Any]:
    """精确等值查询。

    输入 ``{"table": list[dict], "filters": {列: 值}}``
    输出 ``{"rows": [匹配行...]}``

    **零匹配返回 ``{"rows": []}``，不报错** —— 「没查到」是合法答案，不是异常。
    """
    table = _require_table(payload.get("table"))
    if table is None:
        return make_error(ErrorCode.INVALID_INPUT, "table 必须是对象数组")

    filters = payload.get("filters")
    if not isinstance(filters, dict):
        return make_error(ErrorCode.INVALID_INPUT, "filters 必须是对象 {列名: 值}")

    # 先确认 filters 里提到的列真的存在，否则 agent 会以为「查不到」其实是「列名写错」。
    existing = _columns(table)
    for key in filters:
        if table and key not in existing:
            return make_error(ErrorCode.UNKNOWN_COLUMN, f"列 {key!r} 不存在")

    rows = [
        row for row in table
        if all(row.get(key) == value for key, value in filters.items())
    ]
    return {"rows": rows}


# --------------------------------------------------------------------------------------
# table.stats
# --------------------------------------------------------------------------------------


def table_stats(payload: dict[str, Any]) -> dict[str, Any]:
    """统计聚合。

    输入 ``{"table": list[dict], "column": str, "op": "avg|sum|count|max|min"}``
    输出 ``{"result": ...}``

    - ``count`` 返回匹配行数（``int``），**不看 column 是否存在** —— 数行数与列无关。
    - ``avg`` 返回**十进制字符串**（两位小数，``Decimal`` + ``ROUND_HALF_UP``），
      与 TASK-001 ``AnswerKind.NUMBER`` 的规范值形式一致。
    - ``sum`` / ``max`` / ``min`` 返回整数分 ``int``。
    - 空表：``count`` → 0；``avg`` / ``sum`` / ``max`` / ``min`` → ``empty_table`` 错误。
    """
    table = _require_table(payload.get("table"))
    if table is None:
        return make_error(ErrorCode.INVALID_INPUT, "table 必须是对象数组")

    column = payload.get("column")
    op = payload.get("op")
    if not isinstance(column, str) or not isinstance(op, str):
        return make_error(ErrorCode.INVALID_INPUT, "column 与 op 必须是字符串")
    if op not in STATS_OPS:
        return make_error(ErrorCode.UNKNOWN_OP, f"op 必须是 {STATS_OPS} 之一，收到 {op!r}")

    # count 数的是行数，与列是否存在无关 —— 先于列校验短路掉。
    if op == "count":
        return {"result": len(table)}

    if not _has_column(table, column):
        return make_error(ErrorCode.UNKNOWN_COLUMN, f"列 {column!r} 不存在")
    if not table:
        return make_error(ErrorCode.EMPTY_TABLE, f"表为空，无法计算 {op}")

    values = [row[column] for row in table if column in row]
    if not values:
        return make_error(ErrorCode.EMPTY_TABLE, f"列 {column!r} 在所有行中都缺失")

    if any(not _is_number(v) for v in values):
        return make_error(ErrorCode.INVALID_INPUT, f"列 {column!r} 含非整数分值，无法统计")

    total = sum(values)
    if op == "sum":
        return {"result": total}
    if op == "max":
        return {"result": max(values)}
    if op == "min":
        return {"result": min(values)}

    # avg：先在「分」上求和，再折元，最后 HALF_UP 到两位小数。
    avg_yuan = (Decimal(total) / Decimal(len(values))) / Decimal(100)
    return {"result": str(avg_yuan.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))}


# --------------------------------------------------------------------------------------
# table.organize
# --------------------------------------------------------------------------------------


def table_organize(payload: dict[str, Any]) -> dict[str, Any]:
    """稳定排序。

    输入 ``{"table": list[dict], "sort_by": str, "order": "asc|desc",
           "limit": int, "tiebreak": {次级列: asc|desc}}``
    输出 ``{"rows": [排序后前 N 行...]}``

    - 主键 ``sort_by``；并列时按 ``tiebreak`` 次级键（Python ``sorted`` 的 key 列表天然稳定）。
    - ``order`` / ``limit`` / ``tiebreak`` 可选；默认 ``desc``、全取。
    - 返回的是**行集合**，不是「前 N 名的姓名列表」——拼答案是 agent 的活。
    """
    table = _require_table(payload.get("table"))
    if table is None:
        return make_error(ErrorCode.INVALID_INPUT, "table 必须是对象数组")

    sort_by = payload.get("sort_by")
    if not isinstance(sort_by, str):
        return make_error(ErrorCode.INVALID_INPUT, "sort_by 必须是字符串")

    order = payload.get("order", "desc")
    if not isinstance(order, str):
        return make_error(ErrorCode.INVALID_INPUT, "order 必须是字符串")
    if order not in ORDERS:
        return make_error(ErrorCode.UNKNOWN_OP, f"order 必须是 {ORDERS} 之一，收到 {order!r}")

    limit = payload.get("limit")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool)):
        return make_error(ErrorCode.INVALID_INPUT, "limit 必须是整数")
    if limit is not None and limit < 0:
        return make_error(ErrorCode.INVALID_INPUT, "limit 不能为负数")

    tiebreak = payload.get("tiebreak", {})
    if tiebreak is None:
        tiebreak = {}
    if not isinstance(tiebreak, dict):
        return make_error(ErrorCode.INVALID_INPUT, "tiebreak 必须是对象 {列名: asc|desc}")
    for key, direction in tiebreak.items():
        if direction not in ORDERS:
            return make_error(ErrorCode.UNKNOWN_OP, f"tiebreak[{key!r}] 必须是 asc 或 desc")

    if not _has_column(table, sort_by):
        return make_error(ErrorCode.UNKNOWN_COLUMN, f"列 {sort_by!r} 不存在")
    if table:
        existing = _columns(table)
        for key in tiebreak:
            if key not in existing:
                return make_error(ErrorCode.UNKNOWN_COLUMN, f"并列次级键列 {key!r} 不存在")

    # 一律升序排（reverse=False），方向编码进每个 key 自己的包装里。
    # 用 sorted(..., reverse=True) 是错的：它会把整个 key 列表统一反转，
    # 于是「主键 desc + 次级键 asc」会被悄悄变成「主键 desc + 次级键 desc」。
    def _key(row: dict[str, Any]) -> list[Any]:
        primary = _sortable(row.get(sort_by))
        keys: list[Any] = [primary if order == "asc" else _Reversed(primary)]
        for key, direction in tiebreak.items():
            part = _sortable(row.get(key))
            keys.append(part if direction == "asc" else _Reversed(part))
        return keys

    rows = sorted(table, key=_key)
    if limit is not None:
        rows = rows[:limit]
    return {"rows": rows}


class _Reversed:
    """包装一个 key 元素，让它的比较方向反转。

    之所以不直接对数字取负：``_sortable`` 返回 ``(类型秩, 值)``，取负会破坏类型秩
    （秩为 0 的数字变负后会排到秩 2 的空值前面），跨类型也就没法安全比较了。
    """

    __slots__ = ("wrapped",)

    def __init__(self, wrapped: tuple[int, Any]) -> None:
        self.wrapped = wrapped

    def __lt__(self, other: "_Reversed") -> bool:
        return other.wrapped < self.wrapped

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Reversed) and other.wrapped == self.wrapped