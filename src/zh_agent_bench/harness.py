"""harness —— 「轨迹 → 判分 → 榜单」闭环。

M1 的评测流程只有三步，harness 就是把这三步固定下来：

1. **分发**：按任务把 :data:`TOOL_REGISTRY` 里的工具交给 agent；
2. **跑**：agent 自己决定调哪些工具、怎么拼答案（本单是确定性参考求解器）；
3. **判**：把最终答案交给 :func:`judge.judge`，按任务的 ``answer_spec`` 比。

**工具调用一律走** ``check_contract``（不直接 ``spec.fn``，findings.md §四.5）：
每一条调用路径都校验一遍 input/output schema，契约漂移当场暴露，
而不是等判分阶段才发现「工具返回的东西 judge 接不住」。

**榜单输出必须确定性**：同一批 results 进、同一串文本出，不含时间戳、不含内存地址、
不含集合迭代序（字典按插入序，分类按 :class:`Category` 枚举序）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
import json

from .judge import judge, judge_task
from .sandbox import TOOL_SPECS, Sandbox, snapshot, tool_descriptions, tool_functions
from .tasks.schema import Category, Task
from .tools.registry import TOOL_CONTRACT_VERSION, TOOL_REGISTRY, check_contract
from .tools._errors import is_error

#: agent 签名：``(task, tools) -> final_answer``。``tools`` 是 TOOL_REGISTRY 的受控视图。
Agent = Callable[[Task, dict], Any]


class _ToolBox:
    """交给 agent 的工具视图。

    - ``box[tool_id](payload)`` → 经 ``check_contract`` 调用工具，并把这次调用记进轨迹；
    - 只暴露任务 :attr:`Task.tool_names` 声明的工具 —— agent 不该有全量工具的权限，
      否则「这条任务该用哪个工具」这道考题就废了；
    - 未知 tool_id 抛 ``KeyError``，是调用方的 bug，不是工具错误。
    """

    def __init__(self, tool_names: Iterable[str]) -> None:
        self._names = tuple(tool_names)
        self.calls: list[dict[str, Any]] = []

    def __contains__(self, tool_id: object) -> bool:
        return tool_id in self._names

    def __getitem__(self, tool_id: str) -> Callable[[dict], dict]:
        if tool_id not in self._names:
            raise KeyError(f"任务未声明工具 {tool_id!r}，可用: {list(self._names)}")
        record = self.calls

        def _invoke(payload: dict) -> dict:
            result = check_contract(tool_id, payload)
            # 轨迹**全量**记录（findings/RESULT-007）：只留键名的话，事后回看「到底传了什么参、
            # 收到了什么错」就什么都查不出来 —— 而那正是轨迹落盘存在的唯一理由。
            # 失败时把 error 单独提出来给分析用，result 仍保留原始返回值（与工具吐出的完全一致）。
            entry: dict[str, Any] = {"tool": tool_id, "arguments": payload, "result": result}
            if is_error(result):
                entry["error"] = result["error"]
            record.append(entry)
            return result

        return _invoke


def run_trajectory(task: Task, agent: Agent) -> dict[str, Any]:
    """跑一条轨迹并判分。

    返回 ``{"task_id", "category", "tool_calls", "final_answer", "correct", "reason"}``。

    agent 自己抛异常时不吞成「答错」——那会把「工具用错了」和「agent 崩了」混成同一件事，
    前者该扣分、后者该修 bug。异常原样上抛，让调用方看到栈。

    **轨迹进 judge**：走 :func:`judge_task` 一次性把 task + 轨迹交给判分器，
    B 场景的 ``T``（工具协议合规）与证据覆盖率才拿得到 ``tool_calls``。
    A 场景 ``judge_task`` 内部仍走原 :func:`judge` 的四个客观比较算子、**不读 trajectory**，
    所以 A 场景的输出逐字节不变。
    """
    box = _ToolBox(task.tool_names)
    final_answer = agent(task, box)
    verdict = judge_task(task, final_answer, trajectory=box.calls)
    return {
        "task_id": task.id,
        "category": task.category.value,
        "tool_calls": box.calls,
        "final_answer": final_answer,
        "correct": verdict["correct"],
        "reason": verdict["reason"],
    }


def evaluate(tasks: Iterable[Task], agent: Agent) -> list[dict[str, Any]]:
    """逐任务跑轨迹判分。返回结果列表，顺序与输入一致。"""
    return [run_trajectory(task, agent) for task in tasks]


# --------------------------------------------------------------------------------------
# C 场景：沙箱 → agent → 快照 → 判分
# --------------------------------------------------------------------------------------


class _SandboxToolBox:
    """C 场景的工具视图（**不是** :class:`_ToolBox`）。

    三个和 A/B 不同的点，都是被逼出来的而不是设计洁癖：

    1. 工具来自 :func:`sandbox.tool_functions` 而非 ``TOOL_REGISTRY`` —— C 场景的
       ``task.tool_names`` 是 ``order.get`` 这类沙箱工具名，A/B 那套 registry 里没有；
    2. **不做 ``check_contract``**：沙箱工具的契约是 ``sandbox.py`` 自己的同形状错误
       （``unknown_id`` / ``invalid_state``，见 RESULT-019 §5.4），不在 ``tools/_errors``
       的枚举里，硬套 registry 的 schema 校验只会把它们全判成非法调用；
    3. 只暴露 ``task.tool_names`` 声明的工具，与 A/B 同口径 —— 否则「这条该用哪个工具」
       这道考题就废了。

    agent 拿到的仍然只是**函数**，拿不到 :class:`sandbox.Sandbox` 对象本身（闭包隔离）。
    """

    def __init__(self, tool_names: Iterable[str], tools: dict[str, Callable[[dict], dict]]) -> None:
        self._names = tuple(tool_names)
        self._tools = tools
        self.calls: list[dict[str, Any]] = []

    def __contains__(self, tool_id: object) -> bool:
        return tool_id in self._names

    def specs(self) -> list[dict[str, Any]]:
        """本任务声明工具的 OpenAI function schema（**只列 ``tool_names`` 里的**）。

        agents 层优先走这里拿 schema：沙箱工具不在全局 ``TOOL_REGISTRY`` 里，
        硬查那张表只会得到「未知工具 ID」（TASK-020 打回的根因，TASK-021 修）。
        给全量工具等于泄题 —— 与 :class:`_ToolBox` 同口径。
        """
        descriptions = tool_descriptions()
        return [
            {
                "name": tool_id,
                "description": descriptions.get(tool_id, f"沙箱工具 {tool_id}"),
                "parameters": TOOL_SPECS[tool_id],
            }
            for tool_id in self._names
        ]

    def __getitem__(self, tool_id: str) -> Callable[[dict], dict]:
        if tool_id not in self._names:
            raise KeyError(f"任务未声明工具 {tool_id!r}，可用: {list(self._names)}")
        fn = self._tools[tool_id]
        record = self.calls

        def _invoke(payload: dict) -> dict:
            result = fn(payload)
            # 轨迹全量记录，含失败时的 error —— 与 A/B 同口径，判分与事后复盘都靠它。
            entry: dict[str, Any] = {"tool": tool_id, "arguments": payload, "result": result}
            if isinstance(result, dict) and "error" in result:
                entry["error"] = result["error"]
            record.append(entry)
            return result

        return _invoke


def sandbox_from_snapshot(snap: dict) -> Sandbox:
    """从 :func:`sandbox.snapshot` 的输出重建一个 :class:`sandbox.Sandbox`。

    ``sandbox.py`` 是禁区（不能给它加 ``sandbox_from_snapshot``），所以在这里重建。
    **必须深拷贝**：``snap`` 是 ``task.initial_data["sandbox"]``，直接 ``Sandbox(**snap)``
    会让 agent 的写操作污染任务的初始数据 —— 同一个 Task 对象判第二次时，
    环境已经不是出题时那个了。
    """
    if not isinstance(snap, dict):
        raise ValueError("initial_data['sandbox'] 必须是沙箱快照字典")
    missing = {"orders", "inventory", "coupons"} - set(snap)
    if missing:
        raise ValueError(f"沙箱快照缺字段: {sorted(missing)}")
    # json 往返做深拷贝：沙箱快照是 snapshot() 产出的纯 JSON，与 deepcopy 等价，
    # 且 copy 不在本模块的 stdlib 白名单里（tests/test_harness.py 有钉）。
    return Sandbox(**json.loads(json.dumps(snap, ensure_ascii=False)))


def run_orchestration_trajectory(
    task: Task, agent: Agent, judge_fn: Callable[..., dict] = judge_task
) -> dict[str, Any]:
    """跑一条 C 场景轨迹：建沙箱 → 给 agent 工具 → 跑 → 取快照 → 判。

    返回 ``{"task_id", "category", "task_type", "tool_calls", "final_answer",
    "final_snapshot", "correct", "reason"}``（C 场景的三段明细由 eval_run 从 verdict 取）。

    **每条任务一份全新的沙箱**（由 ``initial_data`` 重建）而不是复用一份：
    上一条任务退掉的单会让下一条的陷阱物理上失效，分数随执行顺序变（RESULT-019 §2.3）。

    agent 抛异常时不吞成「答错」，与 :func:`run_trajectory` 同一条铁律：
    工具用错了该扣分、agent 崩了该修 bug。
    """
    initial = getattr(task, "initial_data", None)
    sb = sandbox_from_snapshot(initial["sandbox"] if isinstance(initial, dict) else None)
    box = _SandboxToolBox(task.tool_names, tool_functions(sb))
    final_answer = agent(task, box)
    final = snapshot(sb)
    verdict = judge_fn(task, final_answer, trajectory=box.calls, final_snapshot=final)
    return {
        "task_id": task.id,
        "category": task.category.value,
        "task_type": (initial or {}).get("任务类型", "unknown") if isinstance(initial, dict) else "unknown",
        "tool_calls": box.calls,
        "final_answer": final_answer,
        "final_snapshot": final,
        "correct": bool(verdict.get("correct")),
        "reason": verdict.get("reason", ""),
        "verdict": verdict,
    }


def evaluate_orchestration(
    tasks: Iterable[Task], agent: Agent, judge_fn: Callable[..., dict] = judge_task
) -> list[dict[str, Any]]:
    """C 场景逐任务跑分。返回结果列表，顺序与输入一致。"""
    return [run_orchestration_trajectory(task, agent, judge_fn) for task in tasks]


#: 落盘时对非 JSON 可序列化值兜底。轨迹里可能出现 ``Decimal`` / 集合 / 自定义对象
#: （工具没碰过的路径，但 agent 的 final_answer 是任意的），而 jsonl 必须能写。
#: 用 ``repr`` 而不是 str：集合会打乱、Decimal 会丢精度，两者在事后核对时都是坑。
def _json_safe(value: Any) -> Any:
    """递归转成可 JSON 序列化的值；不认识的结构退化成 repr 字符串。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


def write_trace(results: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """把轨迹写成 jsonl（UTF-8 + LF，一行一条任务）。

    :param results: :func:`evaluate` 的返回值。
    :param path: 输出文件；父目录不存在会自动建。

    **为什么落盘要全量而内存里可以只留键名**：内存里的轨迹只是榜单渲染的副产品，
    而落盘的轨迹是**事后唯一的证据** —— 「模型为什么答错」全靠回看它当时传了什么参、
    收到了什么错。留键名的轨迹回答不了这个问题（RESULT-007 起 `tool_calls` 全量记录）。

    **确定性**：行序 = 输入序，无时间戳、无内存地址、同输入两次逐字节一致
    （findings §二：榜单与产物都必须是可复现的，否则「同 seed 同结果」这句话是假的）。
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in results:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")
    return out


def summarize(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """把结果列表汇成结构化汇总（榜单文本由 :func:`leaderboard` 渲染）。"""
    rows = list(results)
    per_category: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = per_category.setdefault(row["category"], {"n_correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["n_correct"] += 1 if row["correct"] else 0
    n_correct = sum(1 for row in rows if row["correct"])
    return {
        "n_correct": n_correct,
        "total": len(rows),
        "score": (n_correct / len(rows)) if rows else 0.0,
        "per_category": per_category,
    }


#: 榜单 ``agent:`` 行的默认值 —— M1 唯一的 agent 就是参考求解器，所以默认值等于它。
#: 这里写死字符串而**不** import ``reference.REFERENCE_AGENT_NAME``：harness 是通用评测层，
#: 不该反向依赖某一个具体 agent 的实现（harness → reference 是错误的方向）。两侧各写一份
#: 字面量、由测试 ``test_reference_agent_name_matches_hardcoded_board`` 钉住一致，
#: 比多一条模块依赖便宜也干净。
DEFAULT_AGENT_NAME = "reference-solver"


def leaderboard(results: Iterable[dict[str, Any]], agent_name: str = DEFAULT_AGENT_NAME) -> str:
    """渲染榜单文本。

    :param agent_name: 榜单该署谁的名。默认值等于 M1 的 ``reference-solver``，
        所以现有调用点一个字不改、输出逐字节不变。
        TASK-006 之前这里写死 ``reference-solver``，``--agent llm`` 时榜单正文报的是别人的名字 ——
        榜单第一行就是「这榜是谁跑的」，报错名字等于这张榜不可信（findings §七.2）。
        保持 ASCII：本榜单正文承诺纯 ASCII（见下），别往这里传中文。

    **表头纯 ASCII**：本机控制台代码页非 UTF-8，中文表头会渲染成乱码（findings.md §二.2）。
    中文内容（指令、原因）只出现在可选的明细行里，且一律写到 stderr 之外的正文里 ——
    正文保持纯 ASCII，任何代码页下都不会花。
    """
    rows = list(results)
    stats = summarize(rows)
    lines: list[str] = []
    lines.append("=" * 52)
    lines.append("zh-agent-bench  M1 leaderboard")
    lines.append("=" * 52)
    lines.append(f"agent            : {agent_name}")
    lines.append(f"tool contract    : {TOOL_CONTRACT_VERSION}")
    lines.append(f"score            : {stats['score'] * 100:.2f}%")
    lines.append(f"correct          : {stats['n_correct']}/{stats['total']}")
    lines.append("-" * 52)
    lines.append("per-category:")
    # 按 Category 枚举序而非字典插入序：前者是稳定契约，后者取决于任务顺序。
    known = [c.value for c in Category if c.value in stats["per_category"]]
    unknown = sorted(k for k in stats["per_category"] if k not in known)
    for name in known + unknown:
        bucket = stats["per_category"][name]
        pct = (bucket["n_correct"] / bucket["total"] * 100) if bucket["total"] else 0.0
        lines.append(f"  {name:<12} {bucket['n_correct']}/{bucket['total']}  {pct:6.2f}%")
    lines.append("=" * 52)
    wrong = [row for row in rows if not row["correct"]]
    if wrong:
        lines.append(f"missed ({len(wrong)}):")
        for row in wrong:
            lines.append(f"  [{row['category']}] {row['task_id']}: {row['reason']}")
        lines.append("=" * 52)
    return "\n".join(lines)
