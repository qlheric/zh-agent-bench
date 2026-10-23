"""C 场景参考求解器（oracle）—— 判分器上界校准（TASK-022）。

它回答一个问题：**判分器能给出的最高分是多少？**
现在只有「真 LLM agent 8 条里对 7 条」，这句话没法解读 ——
对的那 7 条是判分器判对了，还是 agent 恰好蒙对？错的那 1 条是 agent 差一点，
还是判分器本身严了一点？两种情况都会产出同一个分数。
所以需要一个**知道正确答案**的 agent：它拿满分 ⇒ 分数的差距归 agent；
它拿不满 ⇒ 判分器（rubric / delta / 回复点）自己有 bug。

**与 A 场景 ``reference.py`` 的关键差别**：A 场景的硬约束是「绝不读 ground_truth」。
C 场景**必须**读 —— ``expected_final_state`` 就是「该改成的目标态」这一事实的
机器可读表述，而被评 agent 能从沙箱里推出等价信息（订单状态、券门槛、金额都在
工具返回里）。所以这里不存在 A 场景那种「读了真值就作弊」的作弊面，
读 delta 是**合法的提问方式**，不是偷看。

但有一条是硬约束，和 A 场景一样：**必须经工具改沙箱，不许直接改 Sandbox 对象**。
直接改对象的话，测到的就是「判分器认可任意写法」而不是「判分器认可正确工具链」——
而后者才是本单要验的东西。

**对 ``must_not_change`` 的对象一律不碰**：陷阱任务（退款判断 / 券抵扣）的正确行为
就是「什么都没改」。碰到非 paid 单还去调 ``order.refund`` 是 agent 的错，不是 oracle 的。

**运行时只碰 stdlib**（与 judge / harness 同一条铁律）。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Sequence

from .tasks.orchestration import TASK_TYPES, TYPE_LABELS, build_orchestration_dataset
from .tasks.schema import Task

#: 任务 id → 求解器。**没有 fallback**：C 场景每条任务都必须显式登记处理方式，
#: 没登记的 id 直接报错 —— 兜一个 ``else: pass`` 只会让新任务静默拿 0 分，
#: 而 oracle 的分数正是用来当上界的，静默 0 分会让上界校准失去意义。
_SOLVERS: dict[str, Any] = {}

#: ``order.change_address`` 的 ``reason`` 占位。C 场景数据集里改地址任务必用地址工具，
#: 退款任务必用退款工具；这个常量在这两条路上都合法。
_CHANGE_REASON = "客户要求修改"


def solver(task_id_prefix: str) -> Any:
    """按任务 id 前缀登记求解器。id 形如 ``orch-0007``，前 4 位是类型槽位。"""

    def _register(fn: Any) -> Any:
        _SOLVERS[task_id_prefix] = fn
        return fn

    return _register


def _rows(task: Task, section: str) -> list[dict[str, Any]]:
    value = task.ground_truth.get("expected_final_state", {}).get(section)
    return [row for row in (value or []) if isinstance(row, dict)]


def _initial_orders(task: Task) -> dict[str, dict[str, Any]]:
    snap = (task.initial_data or {}).get("sandbox") or {}
    return {o["order_id"]: o for o in snap.get("orders", []) if isinstance(o, dict)}


def _example_answer(task: Task) -> str:
    answer = task.ground_truth.get("example_answer")
    return answer if isinstance(answer, str) else ""


def _must_not_touch(task: Task, order_id: str) -> bool:
    """该订单是否被 delta 列为「必须保持原样」。"""
    return any(row.get("order_id") == order_id for row in _rows(task, "must_not_change"))


def _refund(tools: Any, order_id: str) -> None:
    """退一笔款。工具报错就炸 —— oracle 的工具链是确定性的，报错即代码错。"""
    result = tools["order.refund"]({"order_id": order_id, "reason": "客户申请退款"})
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"oracle 退款 {order_id} 失败：{result['error']}")


@solver("0001")
def _solve_address(task: Task, tools: Any) -> str:
    """改地址：``order.get`` 看清现状 → ``order.change_address`` 改。"""
    for row in _rows(task, "orders"):
        order_id, new_address = row["order_id"], row["address"]
        current = tools["order.get"]({"order_id": order_id})
        if isinstance(current, dict) and "error" in current:
            raise RuntimeError(f"oracle 查单 {order_id} 失败：{current['error']}")
        result = tools["order.change_address"](
            {"order_id": order_id, "new_address": new_address, "reason": _CHANGE_REASON}
        )
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"oracle 改地址 {order_id} 失败：{result['error']}")
    return _example_answer(task)


@solver("0003")
def _solve_refund_judge(task: Task, tools: Any) -> str:
    """退款判断：**先查后判**。非 paid 单在 ``must_not_change`` 里，退款会被沙箱拒绝，
    所以正确行为是查完就停手 —— 这里刻意不调 ``order.refund``。"""
    for row in _rows(task, "must_not_change"):
        tools["order.get"]({"order_id": row["order_id"]})
    return _example_answer(task)


@solver("0005")
def _solve_coupon(task: Task, tools: Any) -> str:
    """券抵扣：查金额、比对门槛，不达标就**不用券**。

    ``must_not_change`` 里躺的是**券**（``{"code": ..., "used": False}``），
    不含 ``order_id`` —— 要比门槛得先有订单金额，而订单号只在指令里。
    """
    match = re.search(r"SO-\d+", task.instruction)
    if match is None:
        raise ValueError(f"任务 {task.id} 的指令里找不到订单号：{task.instruction!r}")
    order_id = match.group(0)
    order = tools["order.get"]({"order_id": order_id})
    if isinstance(order, dict) and "error" in order:
        raise RuntimeError(f"oracle 查单 {order_id} 失败：{order['error']}")
    return _example_answer(task)


@solver("0007")
def _solve_batch_refund(task: Task, tools: Any) -> str:
    """多单批量：``order.list`` 按客户筛出名下全部订单 → 只退 ``paid`` 的。
    不逐单盲查，是 rubric 第一条明确要求的行为。"""
    initial = _initial_orders(task)
    customer = initial[_rows(task, "orders")[0]["order_id"]]["customer"]
    listed = tools["order.list"]({"customer": customer})
    if isinstance(listed, dict) and "error" in listed:
        raise RuntimeError(f"oracle 列单失败：{listed['error']}")
    for order in listed["orders"]:
        if order["status"] == "paid" and not _must_not_touch(task, order["order_id"]):
            _refund(tools, order["order_id"])
    return _example_answer(task)


#: 任务类型 → id 槽位。槽位由 ``tasks.orchestration.TASK_TYPES`` 的位置推出
#: （``build_orchestration_dataset`` 按该顺序发号，``per_type=2`` ⇒ 每类占 2 个槽位），
#: **不手抄 id**：抄一份的话，改了 ``TASK_TYPES`` 的顺序或 ``per_type`` 就会静默错配。
_ID_SLOTS: dict[str, str] = {t: f"{i * 2 + 1:04d}" for i, t in enumerate(TASK_TYPES)}


def _assert_every_type_has_a_solver() -> None:
    """每个任务类型都必须登记求解器。少一个就直接炸 —— 上界校准最怕「默默少判一类」。"""
    missing = sorted(set(TASK_TYPES) - set(_ID_SLOTS))
    missing += [f"{t}（槽位 {_ID_SLOTS[t]}）" for t in TASK_TYPES
                if _ID_SLOTS.get(t) not in _SOLVERS]
    if missing:
        raise ValueError(f"以下任务类型没有登记 oracle 求解器: {missing}")


def solver_for(task: Task) -> Any:
    """按任务类型挑求解器。类型不认识就报错，不静默返回 0 分。"""
    _assert_every_type_has_a_solver()
    task_type = (task.initial_data or {}).get("任务类型")
    prefix = _ID_SLOTS.get(task_type)
    if prefix is None or prefix not in _SOLVERS:
        raise ValueError(f"任务 {task.id} 的类型 {task_type!r} 没有登记 oracle 求解器")
    return _SOLVERS[prefix]


def solve(task: Task, tools: Any) -> Any:
    """oracle 入口。签名与 :data:`harness.Agent` 一致 —— 走的正是被测 agent 的那条路。"""
    return solver_for(task)(task, tools)


#: harness / eval_run 用这个名字标注「本榜由参考 agent 产生」
REFERENCE_ORCHESTRATION_AGENT_NAME = "reference-orchestration-solver"


# --------------------------------------------------------------------------------------
# 报告渲染（复用 eval_run 的现成件，不另写一套表头/统计口径）
# --------------------------------------------------------------------------------------


def render_reference_report(records: Sequence[dict[str, Any]]) -> str:
    """把 oracle 的记录渲染成 ``report.md``（无时间戳）。

    前半段直接复用 :func:`eval_run.render_orchestration_report`（同一个
    ``agent_name`` / 同一份统计口径），再追加一段 **delta 三段逐条** ——
    上界校准要能一眼看出「是哪一段掉的」，汇总表不够定位。
    """
    from .eval_run import render_orchestration_report

    body = render_orchestration_report(
        records,
        agent_name=REFERENCE_ORCHESTRATION_AGENT_NAME,
        seed=_DEFAULT_SEED, per_type=_DEFAULT_PER_TYPE,
    )
    # 标题里那句「真实 agent」对 oracle 是假话（oracle 读 ground_truth），
    # 换成中性说法，表格与统计口径一字不改。
    body = body.replace(
        "# C 场景真实 agent 跑分报告（多工具编排）",
        "# C 场景 oracle 跑分报告（判分器上界校准）",
        1,
    )
    lines = [body.rstrip("\n"), "", "## oracle 逐条 delta 明细", "",
             "| task_id | 类型 | 工具调用 | orders | coupons | must_not_change | 回复 | 结论 |",
             "|---|---|---|---|---|---|---|---|"]
    mark = lambda v: "✓" if v else "✗"  # noqa: E731 —— 表格里就地一行写完更清楚
    for record in records:
        conclusion = "满分" if record["correct"] else record["reason"]
        lines.append(
            f"| {record['task_id']} | {record['task_type']} | {record['n_tool_calls']} "
            f"| {mark(record['orders_ok'])} | {mark(record['coupons_ok'])} "
            f"| {mark(record['must_not_change_ok'])} | {mark(record['reply_ok'])} "
            f"| {conclusion} |"
        )
    lines.append("")
    lines.append(
        "> oracle 读 `ground_truth.expected_final_state` 是**合法**的：C 场景里该信息"
        "（订单状态 / 券门槛 / 金额）被测 agent 也能从工具返回里推出来，"
        "所以这里不存在 A 场景那种「读真值即作弊」的作弊面。"
        "**真正不许作弊的是直接改 Sandbox 对象** —— oracle 一律经工具改。"
    )
    lines.append("")
    return "\n".join(lines)


_DEFAULT_SEED = 42
_DEFAULT_PER_TYPE = 2
_DEFAULT_OUT = Path("benchmarks/orchestration/reference")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m zh_agent_bench.reference_orchestration",
        description="跑 C 场景 oracle：校准判分器上界（满分 = 判分器认得正确工具链）",
    )
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED, help="沙箱种子（同 eval_run）")
    parser.add_argument("--per-type", type=int, default=_DEFAULT_PER_TYPE,
                        help="每种编排类型多少条（>=1）")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                        help="输出目录（写 report.md）")
    args = parser.parse_args(argv)
    if args.per_type < 1:
        parser.error(f"--per-type 必须 >= 1，收到 {args.per_type}")
        return 2  # pragma: no cover —— parser.error 不返回

    try:
        tasks = build_orchestration_dataset(per_type=args.per_type, seed=args.seed)
    except ValueError as exc:
        print(f"C 场景任务集构造失败：{exc}")
        return 2

    # judge_fn 在这里 import：纯逻辑的 delta 三段由真 judge_task 跑，LLM 那一步会失败并
    # 降级成 0 分（judge.py 的铁律是不抛），失败明细原样进 reason —— 报告里看得见。
    from .eval_run import run_split_orchestration, summarize_orchestration

    records = run_split_orchestration(tasks, solve)
    total = summarize_orchestration(records)["overall"]

    args.out.mkdir(parents=True, exist_ok=True)
    report = args.out / "report.md"
    report.write_text(
        render_reference_report(records), encoding="utf-8", newline="\n"
    )

    print(
        f"agent={REFERENCE_ORCHESTRATION_AGENT_NAME} tasks={total['n']} "
        f"correct={total['n_correct']}/{total['n']} "
        f"delta_ok={total['n_delta_ok']}/{total['n']} "
        f"reply_ok={total['n_reply_ok']}/{total['n']} "
        f"orders={total['segments']['orders']} coupons={total['segments']['coupons']} "
        f"must_not_change={total['segments']['must_not_change']} "
        f"agent_failed={total['n_agent_failed']}"
    )
    for record in records:
        print(f"  {record['task_id']} {record['task_type']} "
              f"tools={record['n_tool_calls']} "
              f"delta={'ok' if record['delta_ok'] else 'NG'} "
              f"reply={'ok' if record['reply_ok'] else 'NG'} "
              f"{'PASS' if record['correct'] else record['reason']}")
    print(f"report -> {report}")
    return 0 if total["n_correct"] == total["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
