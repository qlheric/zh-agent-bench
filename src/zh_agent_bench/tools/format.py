"""格式转换工具：``format.csv_to_json``。

CSV 方言固定（与 TASK-001 §2.5 一致）：``delimiter=","``、``quotechar='"'``、UTF-8、
首行表头、**空字段为 ``""``（不是 null）**。

本模块是 ``tools/`` 里唯一叫 ``format`` 的 —— 它遮蔽了内置 ``format`` 函数，
所以本文件内一律用 f-string，不要用 ``format(...)`` 或 ``str.format``。
"""

from __future__ import annotations

import csv
import io
from typing import Any

from ._errors import ErrorCode, make_error

#: 固定 CSV 方言。整个项目只有这一处定义，工具与 TASK-001 构造器共用同一套约定。
CSV_DIALECT: dict[str, str] = {
    "delimiter": ",",
    "quotechar": '"',
    "encoding": "utf-8",
    "空字段": "",
    "首行": "表头",
}


def csv_to_json(payload: dict[str, Any]) -> dict[str, Any]:
    """CSV 文本 → JSON 对象。

    输入 ``{"csv": "...", "table_name": "可选"}``
    输出 ``{"json": {"表名": <table_name>, "rows": [dict, ...]}}``
    （不传 ``table_name`` 时输出 ``{"rows": [...]}``）

    ``json`` 的结构刻意与 TASK-001 ``format`` 类任务的 ``ground_truth.answer``
    对齐 —— judge 的 ``json_canonical`` 直接吃它，agent 不需要再拼形状。
    """
    text = payload.get("csv")
    if not isinstance(text, str):
        return make_error(ErrorCode.INVALID_INPUT, "csv 必须是字符串")

    table_name = payload.get("table_name")
    if table_name is not None and not isinstance(table_name, str):
        return make_error(ErrorCode.INVALID_INPUT, "table_name 必须是字符串")

    # lineterminator="\n"：不让 csv 模块跟随平台（Windows 会写 CRLF），
    # 否则解析结果随操作系统漂移，破坏「逐字节一致」。
    reader = csv.DictReader(
        io.StringIO(text),
        delimiter=CSV_DIALECT["delimiter"],
        quotechar=CSV_DIALECT["quotechar"],
        lineterminator="\n",
        restval=CSV_DIALECT["空字段"],  # 短行缺列 -> ""
        restkey=None,                     # 长行多出的键不要（保持 key 集合稳定）
    )
    fieldnames = reader.fieldnames
    if not fieldnames:
        # 只有空行 / 空字符串：没有表头就没有行
        return {"json": {"rows": []}}

    rows = [dict(row) for row in reader]

    payload_obj: dict[str, Any] = {}
    if table_name is not None:
        payload_obj["表名"] = table_name
    payload_obj["rows"] = rows
    return {"json": payload_obj}