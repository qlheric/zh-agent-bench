"""参考 agent —— 确定性求解器，榜单基线。

它存在的意义有三个：
1. **证明任务可解**：如果连确定性求解器都答不对，说明任务本身有问题（查询键不唯一、
   排序没有次级键、规范答案算不出来），而不是 agent 不行。
2. **证明工具与 judge 自洽**：参考答案走的是「工具 → 结构化结果 → 拼答案」这条路，
   和真 agent 一模一样；它 100% 说明这条链路通。
3. **给榜单一个基线**：M2 接真 LLM 时，参考 agent 的分数就是下限。

**硬约束：只读 ``task`` 与工具，绝不读 ``task.ground_truth``。**
读了 ground_truth 再吐出来，那就是自欺欺人 —— 判据 2 的 100% 也就毫无意义了。
下面每处用到 ``initial_data`` 的地方都能核对：没有一处碰 ``ground_truth``。
"""

from __future__ import annotations

import re
from typing import Any

from .tasks.schema import Category, Task

#: organize：取前几名的姓名字段。排序交给工具，这里只负责「从行里取哪个字段」。
_NAME_FIELD = "姓名"

#: query：指令形如「查一下『张三』的部门是什么」，目标值就是姓名本身，
#: 返回字段从指令里的「的X是什么」反解。为稳妥起见只认固定的一套字段名。
_QUERY_TARGET_FIELDS = ("部门", "城市", "商品", "销售额")

#: stats：算哪一列。指令里带「」的列名优先，否则用默认的销售额。
_DEFAULT_STATS_COLUMN = "销售额"

#: stats 的算子：按指令关键词推断。参考求解器只支持本数据集的这四种问法。
_STATS_OP_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"平均值|平均"), "avg"),
    (re.compile(r"总和|合计|求和"), "sum"),
    (re.compile(r"最大|最高|最多"), "max"),
    (re.compile(r"最小|最低|最少"), "min"),
    (re.compile(r"数量|几条|多少条"), "count"),
)


def _call(tool_id: str, tools: dict, payload: dict) -> dict:
    """经 harness 提供的工具代理调用（代理内部走 check_contract）。

    这里故意不直接 import TOOL_REGISTRY：这样 harness 可以注入不同快照做测试。
    """
    return tools[tool_id](payload)


def _rows(initial_data: dict) -> list[dict]:
    """从 initial_data 取表。TASK-001 的四类任务里，只有 format 没有 rows。"""
    rows = initial_data.get("rows")
    return rows if isinstance(rows, list) else []


def solve_query(task: Task, tools: dict) -> Any:
    """query：按指令里『姓名』查到匹配行，再取目标字段的值。"""
    match = re.search(r"『(.+?)』", task.instruction)
    if not match:
        return None
    name = match.group(1)
    result = _call("table.query", tools, {"table": _rows(task.initial_data), "filters": {"姓名": name}})
    if "error" in result or not result["rows"]:
        return None
    row = result["rows"][0]
    for field in _QUERY_TARGET_FIELDS:
        if field in row:
            return row[field]
    return None


def solve_stats(task: Task, tools: dict) -> Any:
    """stats：求平均值/合计/最值。

    指令形如「计算『销售明细表』中『销售额』的平均值」——**有两个**直角引号对：
    第一个是表名，第二个才是列名。取第一个会拿表名当列名查，工具直接报 unknown_column。
    所以取**最后一个**（``findall(...)[-1]``）。
    """
    column_match = re.findall(r"「(.+?)」", task.instruction)
    column = column_match[-1] if column_match else _DEFAULT_STATS_COLUMN
    op = "avg"
    for pattern, candidate in _STATS_OP_PATTERNS:
        if pattern.search(task.instruction):
            op = candidate
            break
    result = _call("table.stats", tools, {"table": _rows(task.initial_data), "column": column, "op": op})
    if "error" in result:
        return None
    return result["result"]


def solve_organize(task: Task, tools: dict) -> Any:
    """organize：排序取前 N 名的姓名。**N 与次级键都从指令里读**，
    不写死 3 —— 写死就等于把数据集的巧合当契约。"""
    top_n = 3
    limit_match = re.search(r"前\s*(\d+)\s*名", task.instruction)
    if limit_match:
        top_n = int(limit_match.group(1))

    sort_by = _DEFAULT_STATS_COLUMN
    column_match = re.search(r"「(.+?)」", task.instruction)
    if column_match:
        sort_by = column_match.group(1)

    # 「并列按姓名升序」→ 次级键姓名 asc。这是数据集钉死的口径。
    tiebreak = {_NAME_FIELD: "asc"}
    result = _call("table.organize", tools, {
        "table": _rows(task.initial_data),
        "sort_by": sort_by,
        "order": "desc",
        "limit": top_n,
        "tiebreak": tiebreak,
    })
    if "error" in result:
        return None
    return [row.get(_NAME_FIELD) for row in result["rows"]]


def solve_format(task: Task, tools: dict) -> Any:
    """format：CSV → JSON。表名取 ``initial_data["表名"]``。

    TASK-006 之前这里硬编码了一个与生成器共享的表名常量，而真值里的表名在指令与 initial_data 中
    都无处可寻 —— 参考求解器能 100% 只因它与生成器共享同一个暗号，任何不作弊的 agent 都 0 分。
    表名进了 ``initial_data`` 之后，这条路与真 LLM agent 走的是同一条（findings §七.1）。
    """
    csv_text = task.initial_data.get("csv_text")
    if not isinstance(csv_text, str):
        return None
    table_name = task.initial_data.get("表名")
    if not isinstance(table_name, str):
        return None
    result = _call("format.csv_to_json", tools, {
        "csv": csv_text, "table_name": table_name,
    })
    if "error" in result:
        return None
    return result["json"]


_SOLVERS = {
    Category.QUERY: solve_query,
    Category.STATS: solve_stats,
    Category.ORGANIZE: solve_organize,
    Category.FORMAT: solve_format,
}


def solve(task: Task, tools: dict) -> Any:
    """参考求解器入口。签名与 harness 的 :data:`Agent` 协议一致。"""
    solver = _SOLVERS.get(task.category)
    if solver is None:
        return None
    return solver(task, tools)


#: harness 用这个名字标注「本榜由参考 agent 产生」
REFERENCE_AGENT_NAME = "reference-solver"