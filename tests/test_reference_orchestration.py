"""C 场景 oracle（TASK-022）的离线测试。

本单要证明的是**判分器的上界**。所以这里最关键的不是「oracle 得了多少分」，
而是：**三段 delta 全是纯逻辑，一条 LLM 都不调** —— 真 `judge_task` 的 delta 段必须
在 8 条上全通。回复点要 LLM，离线只能 mock，这一层在回执里如实标注。

回复点的测法：用一个「假设 judge 每点都给满分」的包装器，**只覆盖 reply 段**，
delta 段仍是真 `judge_task` 算出来的。这样「oracle 满分」这句话里，
可判的部分全是真跑的，不可判的部分被显式隔离出来了。
"""

from __future__ import annotations

import copy
import json

import pytest

from zh_agent_bench.harness import run_orchestration_trajectory
from zh_agent_bench.judge import judge_task
from zh_agent_bench.reference_orchestration import (
    REFERENCE_ORCHESTRATION_AGENT_NAME,
    _SOLVERS,
    main,
    render_reference_report,
    solve,
    solver_for,
)
from zh_agent_bench.tasks.orchestration import (
    TASK_TYPES,
    build_orchestration_dataset,
)

WRITE_TOOLS = {"order.change_address", "order.refund", "coupon.use"}


@pytest.fixture(scope="module")
def tasks():
    return build_orchestration_dataset(per_type=2, seed=42)


@pytest.fixture(autouse=True)
def offline_judge(monkeypatch):
    """把 LLM 请求顶掉：返回错误包，让真 judge_task 的回复段**降级成 0 分而不是抛**。

    钉的是 judge.py 的既有铁律「判分层不抛」—— 若哪天改成抛，本单所有测试直接 traceback，
    不会静默变成「测试通过但其实跑的是网络」。
    """
    from zh_agent_bench import llm_client

    monkeypatch.setattr(
        llm_client, "chat", lambda *a, **k: {"error": {"message": "offline"}}, raising=True
    )


def _perfect_reply_judge(task, answer, trajectory=None, *, final_snapshot=None):
    """真 judge_task + 「回复点全满分」的假设。只覆盖 reply 段，delta 段逐字来自真判分。"""
    verdict = judge_task(
        task, answer, trajectory=trajectory, final_snapshot=final_snapshot
    )
    verdict["scores"] = {cid: 1.0 for cid in verdict["scores"]}
    verdict["G"] = 1.0
    verdict["capped"] = False
    verdict["reply_ok"] = True
    verdict["correct"] = bool(verdict["delta_ok"])
    return verdict


def _rows(task, section: str) -> list[dict]:
    return list((task.ground_truth.get("expected_final_state") or {}).get(section) or [])


# --------------------------------------------------------------------------------------
# 判据 1 ①：8 条的 delta 三段全通（真 judge_task，纯逻辑）
# --------------------------------------------------------------------------------------


def test_the_oracle_hits_every_delta_segment_on_all_eight_tasks(tasks):
    """**本单最核心的一条**：delta 三段 8/8 全中，一条 LLM 都不调。

    这一条过了才谈得上「判分器认可正确工具链」。它在 reply_ok 全 0 的情况下断言 ——
    因为 delta 三段本就不该受 LLM 影响。
    """
    assert len(tasks) == 8
    rows = [run_orchestration_trajectory(task, solve) for task in tasks]
    for row in rows:
        verdict = row["verdict"]
        assert verdict["orders_ok"], (row["task_id"], verdict["orders_detail"])
        assert verdict["coupons_ok"], (row["task_id"], verdict["coupons_detail"])
        assert verdict["must_not_change_ok"], (
            row["task_id"], verdict["must_not_change_detail"]
        )
        assert verdict["delta_ok"] is True, row["task_id"]
    assert all(r["final_answer"] is not None for r in rows)      # 没有 agent_failed


def test_the_oracle_is_full_marks_when_the_reply_judge_is_perfect(tasks):
    """reply 段假设满分时，8/8 correct —— 即**判分器的上界是 100%**。"""
    rows = [run_orchestration_trajectory(task, solve, _perfect_reply_judge) for task in tasks]
    wrong = [(r["task_id"], r["reason"]) for r in rows if not r["correct"]]
    assert not wrong, wrong
    assert all(r["verdict"]["reply_ok"] for r in rows)


def test_each_of_the_three_segments_is_asserted_on_its_own(tasks):
    """三段逐段单独断言 —— 合在一起断言的话，段与段之间互相掩盖的错看不见。

    空段本身不算失败（券抵扣任务没有该改的券），所以断言的是「**非空的段必须命中、
    任何一段的失败明细都必须为空**」。
    """
    for task in tasks:
        verdict = run_orchestration_trajectory(task, solve)["verdict"]
        if _rows(task, "orders"):
            assert verdict["orders_ok"], (task.id, verdict["orders_detail"])
        if _rows(task, "coupons"):
            assert verdict["coupons_ok"], (task.id, verdict["coupons_detail"])
        if _rows(task, "must_not_change"):
            assert verdict["must_not_change_ok"], (
                task.id, verdict["must_not_change_detail"]
            )
        assert verdict["orders_detail"] == []
        assert verdict["coupons_detail"] == []
        assert verdict["must_not_change_detail"] == []


# --------------------------------------------------------------------------------------
# 判据 1 ②：oracle 走的是工具路，不是直接改沙箱
# --------------------------------------------------------------------------------------


def test_the_oracle_really_calls_tools_instead_of_touching_the_sandbox(tasks):
    """工具调用计数 > 0 —— 判据 1 ② 的直接验收。

    与此同时钉住 oracle **只拿到函数、拿不到 Sandbox 对象**：真把沙箱递到手上就能
    直接赋值绕过工具，而那测到的就不是「判分器认可正确工具链」了。
    """
    total = 0
    for task in tasks:
        row = run_orchestration_trajectory(task, solve)
        calls = row["tool_calls"]
        total += len(calls)
        assert not hasattr(row, "sandbox")
    assert total > 0
    # 逐条轨迹都记了工具 id 与参数（_SandboxToolBox 全量记录，是复盘的唯一证据）
    for task in tasks:
        row = run_orchestration_trajectory(task, solve)
        for call in row["tool_calls"]:
            assert call["tool"] in task.tool_names
            assert "arguments" in call


def test_the_write_tasks_actually_reach_a_write_tool(tasks):
    """改地址 / 多单批量这两类**必须**调到写工具，否则 delta 是空跑出来的满分。"""
    by_type = {}
    for task in tasks:
        row = run_orchestration_trajectory(task, solve)
        by_type.setdefault(task.initial_data["任务类型"], []).append(
            [c["tool"] for c in row["tool_calls"]]
        )
    for tool_ids in by_type["改地址"]:
        assert "order.change_address" in tool_ids
        assert "order.get" in tool_ids
    for tool_ids in by_type["多单批量"]:
        assert "order.refund" in tool_ids
        assert "order.list" in tool_ids


# --------------------------------------------------------------------------------------
# 判据 1 ③：must_not_change 对象逐字节不变
# --------------------------------------------------------------------------------------


def test_every_must_not_change_object_is_byte_identical_after_the_run(tasks):
    """判据 1 ③：不许动的不许动 —— 逐字节比全量字段，不是比 delta 声明的那一两个。

    ``must_not_change`` 行的键随对象而变（订单是 ``order_id``、券是 ``code``），
    这里按实际存在的键取。
    """
    for task in tasks:
        initial = {
            o["order_id"]: json.dumps(o, ensure_ascii=False, sort_keys=True)
            for o in task.initial_data["sandbox"]["orders"]
        }
        final = {
            o["order_id"]: json.dumps(o, ensure_ascii=False, sort_keys=True)
            for o in run_orchestration_trajectory(task, solve)["final_snapshot"]["orders"]
        }
        for row in _rows(task, "must_not_change"):
            token = row.get("order_id") or row.get("code")
            if token is None:
                continue
            if token in initial:                      # 订单
                assert initial[token] == final[token], (
                    f"{task.id} 的 {token} 是 must_not_change，却变了"
                )


def test_coupons_are_untouched_by_the_oracle(tasks):
    """券抵扣任务里券是 must_not_change；批次退款里没碰券 —— 全部保持初始字节。"""
    for task in tasks:
        before = json.dumps(task.initial_data["sandbox"]["coupons"],
                            ensure_ascii=False, sort_keys=True)
        after = json.dumps(run_orchestration_trajectory(task, solve)["final_snapshot"]["coupons"],
                           ensure_ascii=False, sort_keys=True)
        assert before == after, task.id


# --------------------------------------------------------------------------------------
# 判据 1 ④：non-paid 单不被强行退款
# --------------------------------------------------------------------------------------


def test_the_oracle_never_refunds_an_order_that_is_not_paid(tasks):
    """判据 1 ④。两条证据：轨迹里没有对非 paid 单的 refund；最终状态里它们也没变 refunded。"""
    for task in tasks:
        status = {o["order_id"]: o["status"] for o in task.initial_data["sandbox"]["orders"]}
        row = run_orchestration_trajectory(task, solve)
        for call in row["tool_calls"]:
            if call["tool"] != "order.refund":
                continue
            order_id = call["arguments"]["order_id"]
            assert status[order_id] == "paid", (
                f"{task.id}：oracle 对非 paid 单（{status[order_id]}）{order_id} 调了退款"
            )
            assert "error" not in call["result"]
        final_status = {o["order_id"]: o["status"] for o in row["final_snapshot"]["orders"]}
        for order_id, was in status.items():
            if was != "paid":
                assert final_status[order_id] == was, f"{task.id} 的 {order_id} 状态被改了"


def test_the_trap_tasks_do_not_write_at_all(tasks):
    """陷阱任务（退款判断 / 券抵扣）的正确行为就是「查完停手」，一次写都不许有。"""
    for task in tasks:
        if task.initial_data["任务类型"] not in ("退款判断", "券抵扣"):
            continue
        used = [c["tool"] for c in run_orchestration_trajectory(task, solve)["tool_calls"]]
        assert not set(used) & WRITE_TOOLS, f"{task.id} 调了写工具：{used}"
        assert used, f"{task.id} 一个工具都没调 —— 正确答案也得先查再判"


# --------------------------------------------------------------------------------------
# oracle 的回复
# --------------------------------------------------------------------------------------


def test_the_oracle_returns_the_example_answer_verbatim(tasks):
    for task in tasks:
        row = run_orchestration_trajectory(task, solve)
        assert row["final_answer"] == task.ground_truth["example_answer"], task.id


# --------------------------------------------------------------------------------------
# 求解器登记完整性
# --------------------------------------------------------------------------------------


def test_every_task_type_has_a_solver(tasks):
    """四类任务都登记了求解器，且四类拿到的是**四个不同的函数**。

    断言的是「分派正确」而不是「跑得通」——跑得通由上面几条负责。这样
    ``solve`` 不必真的被执行，``tasks`` fixture 也不会被写脏（它是 module 级共享的，
    在里面就地改 ``initial_data`` 会把同文件里后面几条测试的输入也改了）。
    """
    picked = {t.initial_data["任务类型"]: solver_for(copy.deepcopy(t)) for t in tasks}
    assert set(picked) == set(TASK_TYPES)
    assert len(set(picked.values())) == len(TASK_TYPES)
    assert len(_SOLVERS) == len(TASK_TYPES)


def test_an_unregistered_task_type_raises_instead_of_scoring_zero(tasks):
    """未知类型直接报错。上界校准最怕「默默给 0 分」—— 那会让上界变成一个假数字。"""
    task = copy.deepcopy(tasks[0])
    task.initial_data["任务类型"] = "没见过的类型"
    with pytest.raises(ValueError, match="没有登记"):
        solve(task, {})


# --------------------------------------------------------------------------------------
# CLI 与报告
# --------------------------------------------------------------------------------------


def test_the_cli_writes_a_report(tmp_path):
    """离线跑（LLM 请求被顶成错误包）时 judge 铁律是不抛、回复段降级 0 分 ⇒ 退出码 1。

    **退出码 1 不是 bug**：oracle 没拿满分就不能算上界校准通过，而回复段拿不满
    恰恰是「真 key 缺席」的直接证据。用 ``return 1`` 硬凑 exit 0 等于把假分洗成真分。
    """
    out = tmp_path / "reference"
    assert main(["--out", str(out)]) == 1
    report = out / "report.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert REFERENCE_ORCHESTRATION_AGENT_NAME in text
    for task_id in (t.id for t in build_orchestration_dataset(per_type=2, seed=42)):
        assert task_id in text
    # delta 三段全通（有 key 时 correct 才会全中）⇒ 本单唯一要求「必须真值全通」的部分
    assert "delta 三段全中：8/8" in text
    assert "judge 调用失败" in text
    assert "回复质量达标：0/8" in text


def test_the_report_has_no_timestamp_and_is_byte_reproducible(tasks):
    """同输入两次跑出逐字节相同的报告 —— 否则「可复现」这句话就是假的。"""
    first = [run_orchestration_trajectory(task, solve) for task in tasks]
    second = [run_orchestration_trajectory(task, solve) for task in tasks]
    rows = [_record(row) for row in first]
    assert render_reference_report(rows) == render_reference_report(
        [_record(row) for row in second]
    )
    assert render_reference_report(rows) == render_reference_report(rows)


def _record(row: dict) -> dict:
    """把轨迹压成报告需要的字段（n_tool_calls 等），顺序与 eval_run 的记录一致。"""
    verdict = row["verdict"]
    return {
        "task_id": row["task_id"],
        "task_type": row["task_type"],
        "correct": row["correct"],
        "delta_ok": verdict["delta_ok"],
        "reply_ok": verdict["reply_ok"],
        "orders_ok": verdict["orders_ok"],
        "coupons_ok": verdict["coupons_ok"],
        "must_not_change_ok": verdict["must_not_change_ok"],
        "G": verdict["G"],
        "capped": verdict["capped"],
        "n_tool_calls": len(row["tool_calls"]),
        "agent_failed": row["final_answer"] is None,
        "reason": row["reason"],
    }


def test_an_illegal_per_type_exits_two_without_touching_disk(tmp_path):
    """``--per-type 0`` 在写任何东西之前就被 argparse 挡掉。

    ``parser.error`` 自己 ``SystemExit(2)``，所以这里断言 SystemExit 而不是返回值；
    关键是不 traceback（CLI 层给的是一句人话）。
    """
    out = tmp_path / "reference"
    with pytest.raises(SystemExit) as exc:
        main(["--per-type", "0", "--out", str(out)])
    assert exc.value.code == 2
    assert not out.exists()      # 边界校验发生在写盘之前（非法输入不留半份产物）
