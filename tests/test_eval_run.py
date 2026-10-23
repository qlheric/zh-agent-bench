"""eval_run 的离线测试：**全程不联网**（fake agent + fake judge）。

钉五件事（判据 1）：

1. ``load_formal_split`` 三个 split 的条数 60/120/150，task_id 与入库产物一致；
2. mock 跑分闭环：fake agent + fake judge 跑完，报告字段与统计正确；
3. 降级计数：agent 失败与 judge 降级**分两栏记**，且不混进「答错」；
4. 报告**无时间戳**、同输入两次逐字节一致（LLM 随机性只允许影响分数、不允许影响结构）；
5. 边界：空 split 不除零、单条 agent 异常不掀掉整批、F 为 ``None`` 不被当成 0 进均值。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from zh_agent_bench.calibrate import load_formal_split
from zh_agent_bench.eval_run import (
    _TASK_TYPE_ORDER,
    build_blind_manifest,
    check_blind_allowed,
    evaluate_one,
    render_orchestration_report,
    render_report,
    run_split,
    run_split_orchestration,
    safe_agent_dirname,
    summarize,
    summarize_orchestration,
    verify_blind_manifest,
    write_outputs,
    write_outputs_orchestration,
)
from zh_agent_bench.judge import PASS_THRESHOLD_G
from zh_agent_bench.tasks.orchestration import TYPE_LABELS
from zh_agent_bench.tasks.qa import TASK_TYPES, build_qa_dataset
from zh_agent_bench.tasks.schema import (
    AnswerKind,
    AnswerSpec,
    Category,
    Comparison,
    Normalization,
    Split,
    Task,
)


# --------------------------------------------------------------------------------------
# 测试替身
# --------------------------------------------------------------------------------------


def _fake_agent(answers: dict[str, Any]):
    """按 task_id 给定答案的假 agent。没给到的任务回 ``None``（模拟 agent 没答出来）。"""
    def _run(task: Task, tools: dict) -> Any:
        tools["knowledge.search"]({"query": task.id})  # 走一次真实工具，轨迹才有内容
        return answers.get(task.id)
    return _run


def _fake_judge(g_by_task: dict[str, float], *, f: float | None = 0.5, t: float | None = 1.0,
                degraded_for: frozenset[str] = frozenset()):
    """按 task_id 给定 G 的假 judge。``degraded_for`` 里的任务带降级文案。

    轨迹照收（fake judge 也校验 ``T`` 从轨迹算出来了），好证明接线是通的。
    """
    def _run(task: Task, answer: Any, trajectory: Any = None) -> dict:
        g = g_by_task.get(task.id, 0.0)
        degraded = task.id in degraded_for
        return {
            "correct": g >= 0.8 and not degraded,
            "G": g,
            "F": None if answer is None else f,
            "T": None if not trajectory else t,
            "capped": g < 0.4,
            "scores": {p["criterion_id"]: 0.0 for p in task.rubric},
            "reason": "judge 调用失败：假降级" if degraded else "逐条 rubric 判定完成",
        }
    return _run


@pytest.fixture
def tasks() -> list[Task]:
    """4 条真任务（场景池现生成，每类 1 条）。不用入库产物 —— 后者有专门的加载测试。"""
    return build_qa_dataset(per_type=1, seed=7)


# --------------------------------------------------------------------------------------
# 判据 ①：加载器
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("split", "expected"),
    [("trial", 60), ("dev", 120), ("blind", 150)],
)
def test_load_formal_split_sizes_are_sixty_one_twenty_one_fifty(split: str, expected: int) -> None:
    assert len(load_formal_split(split)) == expected


@pytest.mark.parametrize("split", ["trial", "dev", "blind"])
def test_load_formal_split_ids_match_the_committed_file(split: str) -> None:
    """加载出来的 task_id 必须与产物文件一字不差、且唯一。

    比「条数」更强：条数对但读错了行（比如某条解析成默认值）条数照样对得上。
    """
    root = Path(__file__).resolve().parents[1] / "benchmarks" / "qa"
    on_disk = [
        json.loads(line)["id"]
        for line in (root / split / "qa.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    loaded = [t.id for t in load_formal_split(split)]
    assert loaded == on_disk
    assert len(set(loaded)) == len(loaded)


def test_load_formal_split_rejects_an_unknown_split_name() -> None:
    with pytest.raises(ValueError, match="split 必须是"):
        load_formal_split("holdout")


def test_load_formal_split_honours_an_explicit_data_root(tmp_path: Path) -> None:
    """``--data-root`` 覆盖：把产物复制到临时目录，读出来的必须一模一样。"""
    src = Path(__file__).resolve().parents[1] / "benchmarks" / "qa" / "trial" / "qa.jsonl"
    (tmp_path / "trial").mkdir()
    (tmp_path / "trial" / "qa.jsonl").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    assert [t.id for t in load_formal_split("trial", tmp_path)] == [
        t.id for t in load_formal_split("trial")
    ]


def test_load_formal_split_missing_file_names_the_regeneration_command(tmp_path: Path) -> None:
    """找不到产物时报错要**给出重建命令**。只说「文件不存在」的报错让人无从下手，
    而重建这一步不是每个调用方都记得住。"""
    with pytest.raises(FileNotFoundError, match="--preset formal"):
        load_formal_split("trial", tmp_path)


# --------------------------------------------------------------------------------------
# 判据 ②：mock 跑分闭环
# --------------------------------------------------------------------------------------


def test_run_split_records_g_f_t_and_the_verdict_fields(tasks: list[Task]) -> None:
    answers = {t.id: f"答案 {t.id}" for t in tasks}
    records = run_split(tasks, _fake_agent(answers), _fake_judge({t.id: 1.0 for t in tasks}))
    assert [r["task_id"] for r in records] == [t.id for t in tasks]
    for r in records:
        assert r["correct"] is True
        assert r["G"] == 1.0
        assert r["F"] == 0.5
        assert r["T"] == 1.0
        assert r["capped"] is False
        assert r["n_tool_calls"] == 1          # 轨迹进了 judge，不是空壳
        assert r["agent_failed"] is False
        assert r["judge_degradation"] is None
        assert set(r) == {
            "task_id", "task_type", "correct", "G", "F", "T", "capped",
            "n_tool_calls", "agent_failed", "judge_degradation", "reason",
        }


def test_summarize_slices_by_all_four_task_types(tasks: list[Task]) -> None:
    records = run_split(tasks, _fake_agent({t.id: "x" for t in tasks}),
                        _fake_judge({t.id: 1.0 for t in tasks}))
    stats = summarize(records)
    assert list(stats["per_type"]) == [t for t in TASK_TYPES if t in stats["per_type"]]
    assert sum(s["n"] for s in stats["per_type"].values()) == len(tasks)
    assert stats["overall"]["score"] == 1.0


def test_report_has_totals_type_slices_and_the_failure_table(tasks: list[Task]) -> None:
    """一半判对一半判错：三类必备内容（总分 / 类型切片 / 失败样本）都得在。"""
    ids = [t.id for t in tasks]
    records = run_split(
        tasks,
        _fake_agent({task_id: "x" for task_id in ids}),
        _fake_judge({ids[0]: 1.0, ids[1]: 0.9, ids[2]: 0.2, ids[3]: 0.1}),
    )
    text = render_report(records, split="trial", agent_name="llm-agent:fake")
    assert "## 总分" in text and "## 按任务类型切片" in text and "## 失败样本" in text
    assert "llm-agent:fake" in text
    assert f"主榜门槛：G ≥ {PASS_THRESHOLD_G}" in text
    wrong_ids = {ids[2], ids[3]}
    assert {r["task_id"] for r in records if not r["correct"]} == wrong_ids
    for task_id in wrong_ids:
        assert task_id in text          # 两条失败都进了表


def test_records_jsonl_round_trips_and_carries_every_field(tmp_path: Path, tasks: list[Task]) -> None:
    records = run_split(tasks, _fake_agent({t.id: "x" for t in tasks}),
                        _fake_judge({t.id: 0.5 for t in tasks}))
    write_outputs(records, tmp_path, split="trial", agent_name="llm-agent:fake")
    lines = (tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
    for line, record in zip(lines, records, strict=True):
        # 落盘行 = 内存记录 + 序列化那一刻补上的 agent/judge 署名
        data = json.loads(line)
        assert {k: data[k] for k in record} == record
        assert data["agent"] == "llm-agent:fake"
        assert data["judge"] == "deepseek-chat"


def test_fake_agent_traversal_goes_through_the_real_tool_contract(tasks: list[Task]) -> None:
    """轨迹里的 ``tool_calls`` 是 harness ``_ToolBox`` 记的（过 ``check_contract``）。

    接线错了就悄悄变成空轨迹，而 ``T`` 恰好是「从轨迹算出来的」那个指标 ——
    空轨迹会让 ``T`` 恒为 ``n/a``，报告上根本看不出是接线断了。
    """
    records = run_split(tasks, _fake_agent({t.id: "x" for t in tasks}),
                        _fake_judge({t.id: 1.0 for t in tasks}))
    assert all(r["n_tool_calls"] == 1 for r in records)
    assert all(r["T"] == 1.0 for r in records)


# --------------------------------------------------------------------------------------
# 判据 ③④：降级与异常分栏
# --------------------------------------------------------------------------------------


def test_agent_failure_is_counted_separately_from_a_wrong_answer(tasks: list[Task]) -> None:
    """agent 回 ``None``（LLM 挂了）**不是**「答错了」——两栏必须能分开数出来。

    混在一起的后果很具体：judge 挂掉会把 G 兜底成 0，报告上就长得像
    「agent 全都不会」，而真因是网络/key。
    """
    ids = [t.id for t in tasks]
    records = run_split(
        tasks,
        _fake_agent({ids[0]: "x"}),          # 其余三条 agent 没答出来 → None
        _fake_judge({t.id: 0.0 for t in tasks}),
    )
    stats = summarize(records)
    assert stats["n_agent_failed"] == 3
    assert stats["overall"]["n"] == 4
    assert stats["n_degraded"] == 0        # judge 正常，不是 judge 的锅
    assert all(r["correct"] is False for r in records)


def test_judge_degradation_is_detected_and_flagged_in_the_report(tasks: list[Task]) -> None:
    """judge 降级按 :mod:`calibrate` 的口径认（复用它的判定，不抄第二份）。"""
    ids = [t.id for t in tasks]
    records = run_split(
        tasks,
        _fake_agent({t.id: "x" for t in tasks}),
        _fake_judge({t.id: 1.0 for t in tasks}, degraded_for=frozenset({ids[0]})),
    )
    stats = summarize(records)
    assert stats["n_degraded"] == 1
    assert stats["degradation"] == {"llm_failure": 1}
    assert stats["n_agent_failed"] == 0
    text = render_report(records, split="trial", agent_name="a:b")
    assert "judge 降级" in text and "不可直接引用" in text


def test_one_task_raising_does_not_take_down_the_whole_batch(tasks: list[Task]) -> None:
    """单条 agent 崩了记一条、批继续 —— 60 条里挂 1 条不该让报告整份作废。"""
    def _boom(task: Task, tools: dict) -> Any:
        if task.id == tasks[0].id:
            raise RuntimeError("炸了")
        return "答案"
    records = run_split(tasks, _boom, _fake_judge({t.id: 1.0 for t in tasks}))
    assert len(records) == len(tasks)
    first = records[0]
    assert first["agent_failed"] is True
    assert first["correct"] is False
    assert "agent 抛异常" in first["reason"]
    assert all(r["agent_failed"] is False for r in records[1:])


# --------------------------------------------------------------------------------------
# 判据 ④：无时间戳、结构稳定
# --------------------------------------------------------------------------------------


def test_report_has_no_timestamp_and_is_byte_identical_across_runs(
    tmp_path: Path, tasks: list[Task]
) -> None:
    """同输入两次跑逐字节一致 —— **输入相同就得输出相同**（LLM 随机性是另一层，
    由 fake agent 消掉，见任务契约「不要求逐字节」的例外说明）。"""
    def _build() -> list[dict[str, Any]]:
        return run_split(tasks, _fake_agent({t.id: "x" for t in tasks}),
                         _fake_judge({t.id: 0.6 for t in tasks}))

    reports = []
    for name in ("a", "b"):
        path = write_outputs(_build(), tmp_path / name, split="trial", agent_name="llm-agent:fake")
        reports.append(path.read_text(encoding="utf-8"))
    assert reports[0] == reports[1]


def test_report_carries_no_clock_or_absolute_path(tmp_path: Path, tasks: list[Task]) -> None:
    """把「跑分时刻」和「机器相关的绝对路径」钉死。

    榜单与产物只要有一处随环境变，「同 seed 同结果」这句话就是假的，而事后没人
    说得清两次跑的分数差到底来自模型还是来自落盘环境。
    """
    records = run_split(tasks, _fake_agent({t.id: "x" for t in tasks}),
                        _fake_judge({t.id: 0.6 for t in tasks}))
    path = write_outputs(records, tmp_path, split="trial", agent_name="llm-agent:fake")
    text = path.read_text(encoding="utf-8")
    for banned in ("20:", "T00:", "Z\n", str(tmp_path), "0x"):
        assert banned not in text


# --------------------------------------------------------------------------------------
# 边界
# --------------------------------------------------------------------------------------


def test_summary_of_zero_tasks_does_not_divide_by_zero() -> None:
    stats = summarize([])
    assert stats["overall"] == {
        "n": 0, "n_correct": 0, "score": 0.0, "G": None, "F": None, "T": None,
        "n_capped": 0, "n_agent_failed": 0,
    }
    assert stats["per_type"] == {}
    assert stats["per_type_other"] is None
    assert "无。" in render_report([], split="trial", agent_name="a:b")


def test_metrics_that_are_not_applicable_are_excluded_from_the_mean(tasks: list[Task]) -> None:
    """``None`` = 不适用（空答案的 F），不能当 0 进均值。

    当 0 算的话，一批空答案的 F 均值是 0.000，看着像「完全不忠实」，
    实际是「压根没判」—— 这两件事在归因时要走的路完全不同。
    """
    ids = [t.id for t in tasks]
    records = run_split(
        tasks,
        _fake_agent({ids[0]: "x"}),          # 其余三条 → None → F 不适用
        _fake_judge({t.id: 1.0 for t in tasks}, f=0.8),
    )
    stats = summarize(records)
    assert stats["overall"]["F"] == pytest.approx(0.8)   # 只由那一条有值的答案决定
    assert "n/a" in render_report(records, split="trial", agent_name="a:b")


def test_evaluate_one_leaves_an_explicit_reason_for_a_missing_answer(tasks: list[Task]) -> None:
    """agent 没答出来时仍要留下 judge 的判定记录，不是空白。"""
    record = evaluate_one(tasks[0], lambda task, tools: None, _fake_judge({tasks[0].id: 0.0}))
    assert record["agent_failed"] is True
    assert record["G"] == 0.0
    assert record["reason"]


# --------------------------------------------------------------------------------------
# --save-answers（判据 1 的新增部分：供 annotate 双标用）
# --------------------------------------------------------------------------------------


def test_save_answer_adds_the_answer_and_changes_nothing_else(tasks: list[Task]) -> None:
    """开关只**加** ``answer`` 键，判分相关的字段一字不动。

    这是双标的输入契约：annotate 要答案，但它读的是同一批判分结果 ——
    两个字段集一旦分叉，「judge vs 共识」比的就是两次不同的判分了。
    """
    ids = [t.id for t in tasks]
    agent = _fake_agent({t.id: f"答案-{i}" for i, t in enumerate(tasks)})
    judge = _fake_judge({t.id: 0.9 for t in tasks})

    plain = run_split(tasks, agent, judge)
    saved = run_split(tasks, agent, judge, save_answer=True)

    for index, task in enumerate(tasks):
        assert "answer" not in plain[index]
        assert saved[index]["answer"] == f"答案-{index}"
        assert {k: v for k, v in saved[index].items() if k != "answer"} == plain[index]


def test_the_answer_key_never_leaks_as_an_internal_placeholder(tasks: list[Task]) -> None:
    """``_answer`` 是内部占位，两条返回路径都必须在收尾时清掉。"""
    for save in (False, True):
        record = run_split(tasks[:1], lambda t, tools: "答案",
                           _fake_judge({tasks[0].id: 0.9}), save_answer=save)[0]
        assert "_answer" not in record
        assert ("answer" in record) is save


def test_save_answer_keeps_the_answer_when_the_agent_raises(tasks: list[Task]) -> None:
    """agent 崩了也是一条记录：``answer`` 记 ``None``，不静默丢字段。"""

    def _boom(task: Task, _tools: dict) -> Any:
        raise RuntimeError("连接被重置")

    record = run_split(tasks[:1], _boom, _fake_judge({}), save_answer=True)[0]
    assert record["agent_failed"] is True
    assert record["answer"] is None
    assert "连接被重置" in record["reason"]


def test_records_file_carries_the_answer_when_asked(tmp_path: Path, tasks: list[Task]) -> None:
    """落盘验证：records.jsonl 每行一个 answer，report.md 不变。"""
    records = run_split(
        tasks, _fake_agent({t.id: "甲" for t in tasks}),
        _fake_judge({t.id: 1.0 for t in tasks}), save_answer=True,
    )
    report = write_outputs(records, tmp_path, split="trial", agent_name="fake:model")
    rows = [json.loads(line) for line in
            (tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(tasks)
    assert all(row["answer"] == "甲" for row in rows)
    # answer 是给 annotate 的中间产物，不是报告主体
    assert "answer" not in report.read_text(encoding="utf-8")


# ======================================================================================
# C 场景：--split orchestration 的跑分与报告（全程离线）
# ======================================================================================


def _c_task(task_id: str = "orch-0001") -> Task:
    from zh_agent_bench import sandbox as sb

    return Task(
        id=task_id, category=Category.ORCHESTRATION, schema_version="1.1",
        instruction="请处理这笔订单",
        initial_data={"任务类型": "改地址", "sandbox": sb.snapshot(sb.make_sandbox(42))},
        tool_names=("order.get", "order.change_address", "order.list", "order.refund"),
        ground_truth={"expected_final_state": {"orders": [], "coupons": [],
                                               "must_not_change": []}},
        answer_spec=AnswerSpec(kind=AnswerKind.JSON, comparison=Comparison.ORCHESTRATION,
                               normalization=Normalization.NONE),
        split=Split.TRAIN, rubric=(),
    )


def _c_record(task_id: str, task_type: str, **flags: Any) -> dict[str, Any]:
    """造一条 C 场景记录。七字段是契约 §2.5 点名的，不给默认值。"""
    record: dict[str, Any] = {
        "task_id": task_id,
        "task_type": task_type,
        "correct": False,
        "delta_ok": False,
        "reply_ok": False,
        "orders_ok": False,
        "coupons_ok": False,
        "must_not_change_ok": False,
        "G": 0.0,
        "capped": False,
        "n_tool_calls": 3,
        "agent_failed": False,
        "reason": "空记录",
    }
    record.update(flags)
    return record


def _c_judge(task: Task, answer: Any, trajectory: Any = None,
             final_snapshot: Any = None) -> dict[str, Any]:
    """只做 delta 段的假 judge（回复点一律判 0 分）。

    ``must_not_change`` 是拿 initial 与 final 逐段实比出来的，不是写死的 True ——
    写死的话这条断言就只验了「跑分器会填这个字段」，没验「跑分器算得对」。
    """
    import json as _json

    initial = task.initial_data["sandbox"]
    unchanged = _json.dumps(initial, sort_keys=True, ensure_ascii=False) == _json.dumps(
        final_snapshot, sort_keys=True, ensure_ascii=False
    )
    must_not_change_ok = unchanged
    orders_ok = coupons_ok = False  # 什么都没调，目标态必然没达成
    delta_ok = orders_ok and coupons_ok and must_not_change_ok
    return {
        "correct": delta_ok, "reason": "delta 比对完成",
        "delta_ok": delta_ok, "reply_ok": False,
        "orders_ok": orders_ok, "coupons_ok": coupons_ok,
        "must_not_change_ok": must_not_change_ok, "G": 0.0, "capped": False,
    }


def test_the_seven_contract_fields_are_present_even_when_the_task_crashes() -> None:
    """契约 §2.5 要求的七个字段每条都得有，崩掉的那条也一样 —— 少字段报告就崩。"""
    def _boom(task: Task, _tools: dict) -> Any:
        raise RuntimeError("连接被重置")

    records = run_split_orchestration(
        [_c_task()], _boom, lambda *a, **k: {"correct": False, "reason": "不该走到这里"},
    )
    required = ("correct", "delta_ok", "reply_ok", "orders_ok", "coupons_ok",
                "must_not_change_ok", "reason")
    assert all(field in records[0] for field in required)
    assert records[0]["agent_failed"] is True
    assert "连接被重置" in records[0]["reason"]


def test_a_crashing_task_does_not_take_down_the_batch() -> None:
    """8 条里挂 1 条是数据问题，掀掉 8 条报告才是。"""
    seen: list[str] = []

    def _maybe_boom(task: Task, _tools: dict) -> Any:
        seen.append(task.id)
        if task.id == "orch-0002":
            raise RuntimeError("单条崩了")
        return "没动"

    tasks = [_c_task(f"orch-000{i}") for i in (1, 2, 3)]
    records = run_split_orchestration(
        tasks, _maybe_boom, lambda *a, **k: {"correct": True, "reason": "fake"},
    )
    assert len(records) == 3
    assert seen == ["orch-0001", "orch-0002", "orch-0003"]
    assert [r["agent_failed"] for r in records] == [False, True, False]


def test_the_delta_segments_are_counted_separately() -> None:
    """「把该改的改错了」和「不该动的动了」是两种失败，合成一个布尔就说不出是哪一种。"""
    records = [
        _c_record("a", "改地址", correct=True, delta_ok=True, reply_ok=True,
                  orders_ok=True, coupons_ok=True, must_not_change_ok=True),
        _c_record("b", "退款判断", orders_ok=True, must_not_change_ok=False,
                  reason="must_not_change 被破坏"),
        _c_record("c", "券抵扣", coupons_ok=True, must_not_change_ok=True, reply_ok=False,
                  reason="回复质量不达标"),
    ]
    total = summarize_orchestration(records)["overall"]
    assert total["n"] == 3
    assert total["n_correct"] == 1
    assert total["n_delta_ok"] == 1
    assert total["n_reply_ok"] == 1
    assert total["segments"] == {"orders": 2, "coupons": 2, "must_not_change": 2}
    assert total["score"] == pytest.approx(1 / 3)


def test_the_per_type_slice_uses_the_c_scene_type_labels() -> None:
    """分组按 C 场景自己的类型表（B 的四类在这里一条都不该出现）。"""
    records = [_c_record("a", "退款判断"), _c_record("b", "单跳")]
    stats = summarize_orchestration(records)
    assert "退款判断" in stats["per_type"]
    assert "单跳" not in stats["per_type"]
    assert stats["per_type_other"]["n"] == 1


def test_the_report_lists_all_three_delta_segments() -> None:
    """三段明细是报告主体，缺一段就看不出 agent 到底错在哪。"""
    report = render_orchestration_report(
        [_c_record("orch-0001", "改地址", orders_ok=True)],
        agent_name="fake:model", seed=42, per_type=2,
    )
    for seg in ("orders", "coupons", "must_not_change"):
        assert seg in report


def test_the_report_has_no_timestamp_and_is_byte_identical_across_runs() -> None:
    """无时间戳 + 同输入两次逐字节一致（LLM 随机性只允许影响分数、不允许影响结构）。"""
    records = [_c_record("orch-0001", "改地址", orders_ok=True)]
    first = render_orchestration_report(records, agent_name="fake:model", seed=42, per_type=2)
    second = render_orchestration_report(records, agent_name="fake:model", seed=42, per_type=2)
    assert first.encode("utf-8") == second.encode("utf-8")


def test_the_report_carries_a_statistical_caveat() -> None:
    """8 条只验通路，任何准确率数字都不具统计意义 —— 这句话必须留在报告里。"""
    report = render_orchestration_report(
        [_c_record("orch-0001", "改地址")], agent_name="fake:model", seed=42, per_type=2,
    )
    assert "统计意义" in report


def test_a_failed_sample_shows_which_segment_broke() -> None:
    """失败样本表逐段打勾/打叉：只写「答错了」等于让人回看轨迹重读一遍。"""
    report = render_orchestration_report(
        [_c_record("orch-0001", "退款判断", orders_ok=True,
                   must_not_change_ok=False, reason="must_not_change 被破坏")],
        agent_name="fake:model", seed=42, per_type=2,
    )
    assert "orch-0001" in report
    assert "must_not_change 被破坏" in report


def test_write_outputs_orchestration_writes_both_artifacts(tmp_path: Path) -> None:
    """``records.jsonl`` + ``report.md`` 两份产物，且每条一行可解析。"""
    records = [_c_record("a", "改地址", correct=True, delta_ok=True, reply_ok=True),
               _c_record("b", "退款判断")]
    report_path = write_outputs_orchestration(
        records, tmp_path, agent_name="fake:model", seed=42, per_type=2,
    )
    assert report_path.name == "report.md"
    rows = [json.loads(line) for line in
            (tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["task_id"] for r in rows] == ["a", "b"]
    assert report_path.read_text(encoding="utf-8").startswith("# C 场景")


def test_an_empty_orchestration_split_does_not_divide_by_zero() -> None:
    """空 split 与「全错」在报告上是两件事，不能除零炸掉。"""
    total = summarize_orchestration([])["overall"]
    assert total["n"] == 0
    assert total["score"] == 0.0
    assert total["G"] is None


def test_the_c_and_b_reports_do_not_share_a_shape() -> None:
    """B 的字段是 G/F/T/capped、C 的是 delta 三段 + reply，混在一张表里
    「不适用」与「零分」就再也分不开了。"""
    assert set(_TASK_TYPE_ORDER) != set(TYPE_LABELS)
    b_report = render_report([], split="trial", agent_name="fake:model")
    c_report = render_orchestration_report([], agent_name="fake:model", seed=42, per_type=2)
    assert "B 场景" in b_report and "C 场景" not in b_report
    assert "C 场景" in c_report and "B 场景" not in c_report


def test_the_cli_wires_the_orchestration_split_end_to_end(monkeypatch, tmp_path: Path) -> None:
    """``--split orchestration`` 走 C 场景全套：现算任务集 → 跑分 → 写两份产物 → exit 0。"""
    from zh_agent_bench import eval_run

    monkeypatch.setattr(eval_run, "build_orchestration_dataset",
                        lambda per_type, seed: [_c_task("orch-0001")])

    class _FakeAgent:
        name = "fake:model"

    monkeypatch.setattr("zh_agent_bench.agents.llm.make_llm_agent",
                        lambda model=None: _FakeAgent())
    assert eval_run.main(["--split", "orchestration", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "records.jsonl").exists()


def test_the_orchestration_cli_reports_a_bad_dataset_instead_of_crashing(monkeypatch) -> None:
    """构造不出任务集（环境规模不够等）⇒ exit 2 + 原因，不是 traceback。"""
    from zh_agent_bench import eval_run

    def _boom(per_type, seed):
        raise ValueError("沙箱订单数不够造陷阱")

    monkeypatch.setattr(eval_run, "build_orchestration_dataset", _boom)
    assert eval_run.main(["--split", "orchestration"]) == 2


def test_the_real_c_dataset_runs_through_the_runner() -> None:
    """用**真实构造器**的任务集过一遍跑分器：字段形状对得上，不是只对假任务成立。

    顺带钉住三段的极性：一个工具都不调 ⇒ 世界没被动过 ⇒ 所有 must_not_change 通过；
    而「该改的没改」的任务 orders/coupons 必须失败。三段的极性反了这两条立刻红。
    """
    from zh_agent_bench.tasks.orchestration import build_orchestration_dataset

    tasks = build_orchestration_dataset(per_type=2, seed=42)
    assert len(tasks) == 8

    records = run_split_orchestration(
        tasks, lambda task, tools: "我什么都没动", _c_judge,
    )
    assert len(records) == 8
    assert all(r["must_not_change_ok"] for r in records)
    assert sum(1 for r in records if r["orders_ok"]) == 0
    assert sum(1 for r in records if r["coupons_ok"]) == 0


# --------------------------------------------------------------------------------------
# TASK-024：C 场景接正式集（--preset dev|blind）
# --------------------------------------------------------------------------------------


def _fake_llm_agent(monkeypatch) -> None:
    """把真 agent 换成一个只会说话的假 agent。

    本组测试只验「**读哪一档任务集**」，判分一律走 :func:`_c_judge`（纯本地），
    所以绝不能真调 LLM。
    """

    class _FakeAgent:
        name = "fake:preset-probe"

        def __call__(self, task, tools):
            return "我什么都没动"

    monkeypatch.setattr(
        "zh_agent_bench.agents.llm.make_llm_agent", lambda model=None: _FakeAgent()
    )


def _patch_c_judge(monkeypatch) -> None:
    """C 场景 CLI 内部 import 的 judge 换成纯本地的 :func:`_c_judge`。
    ``run_split_orchestration`` 的 ``judge_fn`` 有默认值，monkeypatch 它会一并改掉。"""
    from zh_agent_bench import eval_run

    monkeypatch.setattr(
        eval_run,
        "run_split_orchestration",
        lambda tasks, agent, judge_fn=None: run_split_orchestration(tasks, agent, _c_judge),
    )


def _read_records(out_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (out_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_the_c_preset_defaults_to_trial_and_still_builds_eight(monkeypatch, tmp_path) -> None:
    """**缺省 ``--preset trial`` = 原行为逐字不变**：按 ``(seed, per_type)`` 现算 8 条。

    这是向后兼容的锚点 —— 没给 ``--preset`` 的老命令行必须还跑现算那条路，
    而不是去读一个可能没入库的 trial 档文件。
    """
    from zh_agent_bench import eval_run

    _patch_c_judge(monkeypatch)
    _fake_llm_agent(monkeypatch)
    assert eval_run.main(["--split", "orchestration", "--out", str(tmp_path)]) == 0

    records = _read_records(tmp_path)
    assert len(records) == 8
    assert {r["task_id"] for r in records} == {f"orch-{i:04d}" for i in range(1, 9)}


@pytest.mark.parametrize("preset,n", [("dev", 16), ("blind", 16)])
def test_each_c_preset_reads_its_own_sixteen_task_set(monkeypatch, tmp_path, preset, n) -> None:
    """``--preset dev|blind`` 读**该档入库的正式集**，各 16 条。

    断言两件事：条数对得上，且 task_id 带该档前缀 —— 只查条数的话，
    ``--preset dev`` 悄悄读了 blind 那份也照样绿。
    """
    from zh_agent_bench import eval_run

    _patch_c_judge(monkeypatch)
    _fake_llm_agent(monkeypatch)
    # blind 档要显式放行才能读（TASK-029 封存闸）；dev 档**不给**这个旗子 ——
    # 顺手钉住「加了也不影响 dev」不是本单要管的事。
    assert eval_run.main(
        ["--split", "orchestration", "--preset", preset, "--out", str(tmp_path),
         *(["--allow-blind"] if preset == "blind" else [])]
    ) == 0

    records = _read_records(tmp_path)
    assert len(records) == n
    assert all(r["task_id"].startswith(f"orch-{preset}-") for r in records)


def test_the_c_presets_never_read_each_other() -> None:
    """三档任务集两两不相交，且合起来是入库的 40 条。

    ``load_orchestration_preset`` 是「按目录名读」而不是「按档位字段筛」，
    唯一的保证就是三份文件本身不相交 —— 这条钉住那个前提。
    """
    from zh_agent_bench.eval_run import load_orchestration_preset

    sets = {
        preset: {t.id for t in load_orchestration_preset(preset)}
        for preset in ("trial", "dev", "blind")
    }
    assert len(sets["trial"]) == 8 and len(sets["dev"]) == 16 and len(sets["blind"]) == 16
    assert sets["trial"] & sets["dev"] == set()
    assert sets["dev"] & sets["blind"] == set()
    assert sets["trial"] & sets["blind"] == set()
    assert len(sets["trial"] | sets["dev"] | sets["blind"]) == 40


def test_a_missing_c_preset_is_reported_not_crashed(tmp_path) -> None:
    """档位产物没入库 ⇒ exit 2 + 人话原因，不是 traceback。"""
    from zh_agent_bench import eval_run

    # 必须放行才能走到「档位文件缺失」那一步，否则 exit 2 是封存闸给的，
    # 这条测试就变成在测封存闸、却顶着「缺档位」的标题。
    assert eval_run.main([
        "--split", "orchestration", "--preset", "blind", "--allow-blind",
        "--data-root", str(tmp_path / "空的"),
    ]) == 2


def test_the_c_preset_rejects_an_unknown_grade() -> None:
    """档位名不认识 ⇒ argparse choices 直接拒（B 场景 ``--split`` 同一纪律）。"""
    from zh_agent_bench import eval_run

    with pytest.raises(SystemExit):
        eval_run.main(["--split", "orchestration", "--preset", "final"])


def test_each_c_preset_writes_its_own_results_directory(monkeypatch, tmp_path) -> None:
    """档位与产物目录一一对应（``--out`` 没给时）。

    三档混在一个目录里，榜单和人工核对都分不清「这张表跑的是哪一档」。
    """
    from zh_agent_bench import eval_run

    _patch_c_judge(monkeypatch)
    _fake_llm_agent(monkeypatch)
    # chdir 到 tmp_path 让产物落在这儿（不往仓库里写），正式集路径显式指回仓库。
    # 相对仓库根解析，**不能靠 cwd**：同一测试里第二次调本函数时 cwd 已被 chdir 到 tmp_path，
    # 再 resolve() 会解析成 tmp 下的假路径，读不到入库任务集。
    data_root = Path(__file__).resolve().parents[1] / eval_run._C_TASKS_ROOT
    monkeypatch.chdir(tmp_path)
    assert eval_run.main(["--split", "orchestration", "--preset", "dev",
                          "--data-root", str(data_root)]) == 0
    # agent 目录名 = agent 自报名「fake:preset-probe」的小写安全化
    assert (tmp_path / "benchmarks/orchestration/dev/fake-preset-probe/records.jsonl").is_file()
    assert not (tmp_path / "benchmarks/orchestration/trial").exists()


# --------------------------------------------------------------------------------------
# TASK-028：产物路径带 agent 名 + records 自证署名
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("llm:deepseek-chat", "llm-deepseek-chat"),
    ("llm-agent:deepseek-chat", "llm-agent-deepseek-chat"),
    ("reference-solver", "reference-solver"),
    ("gpt-6.1", "gpt-6-1"),
    ("A/B:C", "a-b-c"),                     # 斜杠：Windows 下就是目录分隔符
    ('weird<>:"\\|?*name', "weird-name"),   # Windows 非法字符全换掉
    ("多个   连续  空格", "多个-连续-空格"),  # 折叠连续分隔符
    ("  裁掉首尾  ", "裁掉首尾"),
    ("", "unknown"),                        # 空名不许产出 "." 这种目录
    ("---", "unknown"),                     # 全是分隔符 ⇒ 同上
])
def test_safe_agent_dirname(raw: str, expected: str) -> None:
    """agent 名 → 目录名。**只做安全化，不改名** —— 目录名要能与 report.md 的署名对上。"""
    assert safe_agent_dirname(raw) == expected


def _run_c_as(monkeypatch, tmp_path, agent_factory, *extra: str) -> Path:
    """用 ``agent_factory`` 造一个假 agent 跑一遍 C 场景 dev 档，返回产物目录。"""
    from zh_agent_bench import eval_run

    _patch_c_judge(monkeypatch)
    monkeypatch.setattr("zh_agent_bench.agents.llm.make_llm_agent", agent_factory)
    # 相对仓库根解析，**不能靠 cwd**：同一测试里第二次调本函数时 cwd 已被 chdir 到 tmp_path，
    # 再 resolve() 会解析成 tmp 下的假路径，读不到入库任务集。
    data_root = Path(__file__).resolve().parents[1] / eval_run._C_TASKS_ROOT
    monkeypatch.chdir(tmp_path)
    assert eval_run.main(["--split", "orchestration", "--preset", "dev",
                          "--data-root", str(data_root), *extra]) == 0
    return tmp_path / "benchmarks/orchestration/dev"


def test_two_agents_in_one_preset_do_not_overwrite_each_other(monkeypatch, tmp_path) -> None:
    """**判据 1②：两个 agent 跑同一档落两个不同目录，先跑的产物还在。**

    这正是旧布局 ``benchmarks/orchestration/dev-results/records.jsonl`` 的病：
    后跑的静默盖掉先跑的，两次跑分的分数永远只能留下一份。
    """
    class _AgentA:
        name = "fake:a"

        def __call__(self, task, tools):
            return "甲"

    class _AgentB:
        name = "fake:b"

        def __call__(self, task, tools):
            return "乙"

    # 同一个仓库根先后跑两个 agent —— 正是「互相覆盖」会发生的场景
    for cls in (_AgentA, _AgentB):
        _run_c_as(monkeypatch, tmp_path, lambda model=None, c=cls: c())

    grade = tmp_path / "benchmarks/orchestration/dev"
    assert sorted(p.name for p in grade.iterdir() if p.is_dir()) == ["fake-a", "fake-b"]
    assert {r["agent"] for r in _read_records(grade / "fake-a")} == {"fake:a"}
    assert {r["agent"] for r in _read_records(grade / "fake-b")} == {"fake:b"}


def test_the_records_file_names_the_judge_that_scored_it(monkeypatch, tmp_path) -> None:
    """每行盖 ``agent`` + ``judge`` 署名 —— 补 TASK-027 §7 第 2 条那条欠账。

    换 judge 重跑同一批，两份产物各落各的目录，``records.jsonl`` 自己说清谁判的。
    """
    class _Probe:
        name = "fake:probe"

        def __call__(self, task, tools):
            return "我什么都没动"

    grade = _run_c_as(monkeypatch, tmp_path, lambda model=None: _Probe(),
                      "--judge-model", "gpt-6-astra")

    records = _read_records(grade / "fake-probe")
    assert {r["judge"] for r in records} == {"gpt-6-astra"}
    assert {r["agent"] for r in records} == {"fake:probe"}


def test_an_explicit_out_still_wins_over_the_agent_directory(monkeypatch, tmp_path) -> None:
    """``--out`` 给了就原样用，**不被 agent 目录改写** —— 旧命令行不能因此作废。"""
    _fake_llm_agent(monkeypatch)
    from zh_agent_bench import eval_run

    _patch_c_judge(monkeypatch)
    # 相对仓库根解析，**不能靠 cwd**：同一测试里第二次调本函数时 cwd 已被 chdir 到 tmp_path，
    # 再 resolve() 会解析成 tmp 下的假路径，读不到入库任务集。
    data_root = Path(__file__).resolve().parents[1] / eval_run._C_TASKS_ROOT
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "我自己指定的地方"

    assert eval_run.main(["--split", "orchestration", "--preset", "dev",
                          "--data-root", str(data_root), "--out", str(out)]) == 0
    assert (out / "records.jsonl").is_file()
    assert not (tmp_path / "benchmarks").exists()


def test_the_b_split_is_unaffected_by_the_c_preset_flag(monkeypatch, tmp_path) -> None:
    """A/B 路径零红：B 场景跑**加了** ``--preset`` 的命令行，产物逐字节不变。

    钉的是「加 ``--preset`` 不改变 B 的行为」而不是「B 拒绝 ``--preset``」——
    榜单 README 里 B 和 C 的命令会并排给，B 那条不能因为 C 多了一个参数就作废。
    """
    from zh_agent_bench import eval_run
    from zh_agent_bench.tasks.qa import build_qa_dataset

    tasks = build_qa_dataset(per_type=2, seed=7)
    _fake_llm_agent(monkeypatch)
    monkeypatch.setattr(eval_run, "load_formal_split", lambda split, data_root=None: tasks)

    plain = eval_run.main(["--split", "trial", "--out", str(tmp_path / "a")])
    with_preset = eval_run.main(
        ["--split", "trial", "--preset", "dev", "--out", str(tmp_path / "b")]
    )
    assert plain == with_preset == 0
    assert (tmp_path / "a/records.jsonl").read_bytes() == (
        tmp_path / "b/records.jsonl"
    ).read_bytes()


# --------------------------------------------------------------------------------------
# TASK-029：blind 封存闸 + sha256 封存清单
# --------------------------------------------------------------------------------------


def _blind_c_cli(monkeypatch, tmp_path, *extra: str) -> int:
    """跑一次 C 场景 blind 档的 CLI（假 agent + 纯本地 judge，不联网）。"""
    from zh_agent_bench import eval_run

    _patch_c_judge(monkeypatch)
    _fake_llm_agent(monkeypatch)
    # 相对仓库根解析，**不能靠 cwd**：同一测试里第二次调本函数时 cwd 已在 tmp_path 下。
    data_root = Path(__file__).resolve().parents[1] / eval_run._C_TASKS_ROOT
    monkeypatch.chdir(tmp_path)
    return eval_run.main(["--split", "orchestration", "--preset", "blind",
                          "--data-root", str(data_root), *extra])


def test_the_c_blind_preset_is_refused_unless_allow_blind_is_given(monkeypatch, tmp_path) -> None:
    """**封存闸的核心**：不带 ``--allow-blind`` 读 blind ⇒ exit 2，且一个产物都不许冒出来。

    静默降级到 dev 比直接拒绝更坏：命令 exit 0、榜单照出，调用方以为自己看的是
    盲测分，其实看的是 dev —— 那正是这份榜单唯一不能有的性质。
    """
    assert _blind_c_cli(monkeypatch, tmp_path) == 2
    assert not (tmp_path / "benchmarks").exists()
    assert _blind_c_cli(monkeypatch, tmp_path, "--allow-blind") == 0
    assert (tmp_path / "benchmarks/orchestration/blind/fake-preset-probe/records.jsonl").is_file()


def test_the_b_blind_split_is_refused_too(monkeypatch, tmp_path) -> None:
    """B 侧 ``--split blind`` 与 C 侧同一道闸 —— 只堵一半等于没堵。"""
    from zh_agent_bench import eval_run

    assert eval_run.main(["--split", "blind"]) == 2


def test_the_refusal_says_which_grade_and_how_to_unlock_it(capsys) -> None:
    """报错必须点名 blind 与 ``--allow-blind`` —— 只说「拒绝」等于让人去翻源码。"""
    from zh_agent_bench import eval_run

    assert eval_run.main(["--split", "blind"]) == 2
    err = capsys.readouterr().out
    assert "blind" in err and "--allow-blind" in err and "dev" in err


def test_a_refusal_never_degrades_to_dev(monkeypatch, tmp_path) -> None:
    """被拒时**不许读 dev 档顶替**。钉的是「一个字节都不读」。

    只要闸落在任务集加载器**之前**，「静默降级」就无从发生 ——
    这条比断言报错文案更能守住性质本身。
    """
    from zh_agent_bench import eval_run

    loaded: list[str] = []
    monkeypatch.setattr(
        eval_run, "load_formal_split",
        lambda split, data_root=None: loaded.append(split) or [],
    )
    monkeypatch.chdir(tmp_path)
    assert eval_run.main(["--split", "blind"]) == 2
    assert loaded == []          # 一个档都没读
    assert not list(tmp_path.iterdir())


def test_dev_and_trial_are_unaffected_by_the_gate(monkeypatch, tmp_path) -> None:
    """**判据 1④：非 blind 路径零影响。** 不带 ``--allow-blind`` 跑 dev 照常 exit 0。"""
    from zh_agent_bench import eval_run
    from zh_agent_bench.tasks.qa import build_qa_dataset

    tasks = build_qa_dataset(per_type=2, seed=7)
    _fake_llm_agent(monkeypatch)
    monkeypatch.setattr(eval_run, "load_formal_split", lambda split, data_root=None: tasks)

    assert eval_run.main(["--split", "dev", "--out", str(tmp_path / "x")]) == 0
    assert (tmp_path / "x/records.jsonl").is_file()


def test_the_committed_manifest_matches_what_is_on_disk() -> None:
    """判据 3：入库的清单与磁盘一致 —— 这就是「blind 没被改过」的凭据。

    路径**相对仓库根**解析，不能靠 cwd：pytest 可能在任何目录起跑，
    靠 cwd 的话这条在 CI 上会假红。
    """
    repo = Path(__file__).resolve().parents[1]
    manifest = repo / "benchmarks/blind-manifest.json"
    assert verify_blind_manifest(manifest, repo) == []
    # 按当前磁盘重生成必须逐字节等于入库那份 —— 校验与生成共用一条代码路径，
    # 两者要是各写一套，「一致」就会在两份清单之间来回横跳。
    assert build_blind_manifest(repo) == manifest.read_text(encoding="utf-8")


def test_a_missing_manifest_is_reported_not_crashed(tmp_path) -> None:
    """清单读不到 ⇒ 算不一致，不抛 traceback。

    清单被误删 / 被 .gitignore 吃掉时，这条要给人话，不是栈回溯。
    """
    diffs = verify_blind_manifest(tmp_path / "没有这份.json", tmp_path)
    assert len(diffs) == 1 and "读不了" in diffs[0]


def test_a_tampered_blind_file_is_reported_by_name() -> None:
    """改一份 blind 产物 ⇒ 校验点名那一条，并给出两侧的 sha 前缀。

    这是判据 2 的后半：清单校验不能只回一句「不一致」，得让人知道**是哪一条**。
    """
    repo = Path(__file__).resolve().parents[1]
    manifest = repo / "benchmarks/blind-manifest.json"
    target = repo / "benchmarks/qa/blind/qa.jsonl"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n")
        diffs = verify_blind_manifest(manifest, repo)
        assert len(diffs) == 1
        assert "benchmarks/qa/blind/qa.jsonl" in diffs[0] and "内容变了" in diffs[0]
    finally:
        target.write_bytes(original)   # 改完必须还原：这是封存集，测试不留痕
    assert verify_blind_manifest(manifest, repo) == []


def test_an_unlisted_blind_file_is_a_difference_too(tmp_path) -> None:
    """磁盘上多一份 blind 产物、清单里没有 ⇒ 算不一致。

    漏记和篡改是同一件事的两个方向：绕过清单塞一份新 blind 产物，
    比直接改内容更坏，因为它连「改了什么」都不留痕。
    """
    manifest = tmp_path / "m.json"
    root = tmp_path / "repo"
    (root / "benchmarks/qa/blind").mkdir(parents=True)
    (root / "benchmarks/qa/blind/qa.jsonl").write_text("{}\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")

    diffs = verify_blind_manifest(manifest, root)
    assert len(diffs) == 1 and "清单里没有" in diffs[0]

    (root / "benchmarks/qa/blind/偷加的.jsonl").write_text("{}\n", encoding="utf-8")
    diffs = verify_blind_manifest(manifest, root)
    # 新文件在磁盘上、不在清单里 ⇒ 方向是「磁盘上有、清单里没有」；
    # 断言必须点名那份文件，否则「多出两份」和「多出一份」都算过。
    assert len(diffs) == 2 and any("偷加的" in d for d in diffs)


def test_the_manifest_cli_exits_zero_on_a_consistent_disk(monkeypatch, capsys) -> None:
    """判据 3 的命令行形态：``--verify-blind-manifest`` 单独跑，不碰 agent。

    chdir 到仓库根：这条正是判据 3 逐字抄的那条命令，路径必须按 cwd 解析，
    所以这里也按 cwd 跑，而不是传 root 绕过去。
    """
    from zh_agent_bench import eval_run

    repo = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo)
    assert eval_run.main(["--verify-blind-manifest"]) == 0
    assert "一致" in capsys.readouterr().out


def test_check_blind_allowed_only_gates_the_blind_grade() -> None:
    """闸只认 blind 一档 —— dev/trial/heldout 走它一律放行。

    钉的是「闸不是全局开关」：以后有人把 ``check_blind_allowed`` 挪到只调一次的
    公共路径上，这条会立刻红，而不是等到某一档被误拒。
    """
    for grade in ("trial", "dev", "heldout"):
        check_blind_allowed(grade, allow_blind=False)
        check_blind_allowed(grade, allow_blind=True)
    with pytest.raises(PermissionError):
        check_blind_allowed("blind", allow_blind=False)
    check_blind_allowed("blind", allow_blind=True)


# --------------------------------------------------------------------------------------
# TASK-032：被测 agent 也能换通道（--agent-base-url / --agent-key-env / --agent-api-key）
#
# 与 TASK-027 的 ``--judge-*`` 同构，但方向相反：换的是**被测 agent**那一侧。
# 有了它，「一 agent 一行」才名副其实 —— 榜单能同时列 deepseek-chat 与 gpt-6-astra。
# --------------------------------------------------------------------------------------


class _AgentProbe:
    """记录 :func:`make_llm_agent` 收到什么的探针。

    ``name`` 决定产物目录名（``llm-agent:<model>`` → ``llm-agent-<model>``），
    所以它自己实现 ``name`` 而不是继承真 agent。
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.model = kwargs.get("model", "")

    @property
    def name(self) -> str:
        return f"llm-agent:{self.model}"

    def __call__(self, task: Task, tools: dict) -> Any:
        return "没动"


def _patch_agent_probe(monkeypatch, sink: list[dict[str, Any]]) -> None:
    """把真 agent 换成探针，并把收到的 kwargs 记进 ``sink``。"""
    def _factory(**kwargs: Any) -> _AgentProbe:
        sink.append(kwargs)
        return _AgentProbe(**kwargs)

    monkeypatch.setattr("zh_agent_bench.agents.llm.make_llm_agent", _factory)


def _run_b_cli(monkeypatch, tmp_path: Path, *extra: str,
               sink: list[dict[str, Any]] | None = None) -> Path:
    """跑一次 B 场景 CLI（假 agent + 纯本地 judge，不联网）。

    **真 agent 一定要替换掉**：留真身的话，在配了 ``DEEPSEEK_API_KEY`` 的机器上
    这条测试会真的发网络请求 —— 离线测试悄悄变成联网测试。
    """
    from zh_agent_bench import eval_run
    from zh_agent_bench.tasks.qa import build_qa_dataset

    monkeypatch.setattr(eval_run, "load_formal_split",
                        lambda split, data_root=None: build_qa_dataset(per_type=1, seed=7))
    monkeypatch.setattr(eval_run, "judge_task", _fake_judge({"x": 1.0}))
    _patch_agent_probe(monkeypatch, sink if sink is not None else [])
    out = tmp_path / "out"
    assert eval_run.main(["--split", "dev", "--out", str(out), *extra]) == 0
    return out


def test_the_agent_switches_default_to_the_previous_behaviour(monkeypatch) -> None:
    """缺省时三个开关必须是「什么都不覆盖」：base_url / api_key 传 ``None``。

    传 ``None`` 而不是「默认端点/默认 env 名」是有意的：:func:`agents.llm.make_llm_agent`
    的缺省值才是唯一真相源，CLI 再抄一份就会漂。
    """
    from zh_agent_bench import eval_run

    sink: list[dict[str, Any]] = []
    _patch_agent_probe(monkeypatch, sink)

    eval_run.build_agent(_agent_namespace([]))

    assert sink == [{"model": eval_run.DEFAULT_MODEL}]
    assert "base_url" not in sink[0]
    assert "api_key" not in sink[0]


def _agent_namespace(argv: list[str]) -> Any:
    """用**真 parser** 解析出 namespace（默认值与命令行一字不差）。

    用 :func:`eval_run.build_parser` 而不是自己另建一个 parser：另建的那份
    迟早会与命令行漂开，而漂开的方向恰好是「默认值不同」—— 那正是本单要钉的性质。
    """
    from zh_agent_bench import eval_run

    return eval_run.build_parser().parse_args(argv)


def test_the_agent_switches_reach_make_llm_agent_one_by_one(monkeypatch) -> None:
    """传了值就逐项对得上：base_url / key-env / api-key 各自独立生效。"""
    from zh_agent_bench import eval_run

    sink: list[dict[str, Any]] = []
    _patch_agent_probe(monkeypatch, sink)
    monkeypatch.setenv("LLMAPI_GPT_API_KEY", "sk-from-env")

    eval_run.build_agent(_agent_namespace([
        "--model", "gpt-6-astra",
        "--agent-base-url", "https://llmapi.example/v1",
        "--agent-key-env", "LLMAPI_GPT_API_KEY",
    ]))

    assert sink == [{
        "model": "gpt-6-astra",
        "base_url": "https://llmapi.example/v1",
        "api_key": "sk-from-env",
    }]


def test_the_agent_api_key_flag_wins_over_the_key_env(monkeypatch) -> None:
    """``--agent-api-key`` 优先级最高：显式给的 key 不该被 env 覆盖掉。"""
    from zh_agent_bench import eval_run

    sink: list[dict[str, Any]] = []
    _patch_agent_probe(monkeypatch, sink)
    monkeypatch.setenv("LLMAPI_GPT_API_KEY", "sk-from-env")

    eval_run.build_agent(_agent_namespace([
        "--agent-key-env", "LLMAPI_GPT_API_KEY",
        "--agent-api-key", "sk-explicit",
    ]))

    assert sink[0]["api_key"] == "sk-explicit"


def test_a_missing_key_env_yields_an_empty_key_not_a_crash(monkeypatch) -> None:
    """key env 没设 ⇒ key 取到空串，**不抛**。

    空 key 由 :func:`llm_client.chat` 转成 ``llm_no_api_key`` 哨兵，agent 侧转成
    「答不出来」交 judge 判错 —— 与缺省路径同一条降级纪律，不许因为换通道就崩。
    """
    from zh_agent_bench import eval_run

    sink: list[dict[str, Any]] = []
    _patch_agent_probe(monkeypatch, sink)
    monkeypatch.delenv("LLMAPI_GPT_API_KEY", raising=False)

    eval_run.build_agent(_agent_namespace(["--agent-key-env", "LLMAPI_GPT_API_KEY"]))

    assert sink[0]["api_key"] == ""


def test_the_agent_switches_reach_the_b_cli(monkeypatch, tmp_path: Path) -> None:
    """B 场景 CLI 吃三个开关 —— 产物落 ``<档>/<agent>/``，agent 目录名跟着模型走。"""
    from zh_agent_bench import eval_run

    sink: list[dict[str, Any]] = []
    _patch_agent_probe(monkeypatch, sink)
    monkeypatch.setenv("LLMAPI_GPT_API_KEY", "sk-from-env")

    out = _run_b_cli(
        monkeypatch, tmp_path,
        "--model", "gpt-6-astra",
        "--agent-base-url", "https://llmapi.example/v1",
        "--agent-key-env", "LLMAPI_GPT_API_KEY",
        sink=sink,
    )

    assert sink[0]["base_url"] == "https://llmapi.example/v1"
    assert sink[0]["api_key"] == "sk-from-env"
    assert sink[0]["model"] == "gpt-6-astra"


def test_the_agent_switches_reach_the_orchestration_cli(monkeypatch, tmp_path: Path) -> None:
    """C 场景 CLI 同样吃三个开关 —— 这是本单最容易漏的那条（它提前 return 到另一函数）。"""
    from zh_agent_bench import eval_run

    sink: list[dict[str, Any]] = []
    _patch_agent_probe(monkeypatch, sink)
    monkeypatch.setenv("LLMAPI_GPT_API_KEY", "sk-from-env")
    _patch_c_judge(monkeypatch)

    out = tmp_path / "c"
    assert eval_run.main([
        "--split", "orchestration", "--preset", "trial", "--per-type", "1",
        "--out", str(out),
        "--model", "gpt-6-astra",
        "--agent-base-url", "https://llmapi.example/v1",
        "--agent-key-env", "LLMAPI_GPT_API_KEY",
    ]) == 0

    assert sink[0]["base_url"] == "https://llmapi.example/v1"
    assert sink[0]["api_key"] == "sk-from-env"
    assert (out / "records.jsonl").is_file()


def test_two_agents_on_one_grade_land_in_two_directories(monkeypatch, tmp_path: Path) -> None:
    """「一 agent 一行」的最后一块砖：换模型跑同一档，两个 agent 各落各的目录。

    目录名由 agent 自报名派生（``llm-agent:gpt-6-astra`` → ``llm-agent-gpt-6-astra``），
    所以换了模型就换了目录 —— 不会互相覆盖。
    """
    from zh_agent_bench import eval_run
    from zh_agent_bench.tasks.qa import build_qa_dataset

    monkeypatch.setattr(eval_run, "load_formal_split",
                        lambda split, data_root=None: build_qa_dataset(per_type=1, seed=7))
    monkeypatch.setattr(eval_run, "judge_task", _fake_judge({"x": 1.0}))
    _patch_agent_probe(monkeypatch, [])
    monkeypatch.chdir(tmp_path)

    assert eval_run.main(["--split", "dev", "--model", "deepseek-chat"]) == 0
    assert eval_run.main(["--split", "dev", "--model", "gpt-6-astra"]) == 0

    assert (tmp_path / "benchmarks/qa/dev/llm-agent-deepseek-chat/records.jsonl").is_file()
    assert (tmp_path / "benchmarks/qa/dev/llm-agent-gpt-6-astra/records.jsonl").is_file()


def test_omitting_the_agent_switches_keeps_the_request_unchanged(monkeypatch) -> None:
    """缺省时请求逐字不变：端点仍是默认端点，密钥仍是 ``None``（由 llm_client 兜底取 env）。

    这是「三个都不传时行为与现在逐字节相同」的可观测落点 —— agent 侧请求里
    端点必须还是 :data:`llm_client.DEFAULT_BASE_URL`，密钥必须还是 ``None``
    （``None`` ⇒ :func:`llm_client.chat` 自己读 ``DEEPSEEK_API_KEY``）。
    """
    from zh_agent_bench import eval_run, llm_client

    calls: list[dict[str, Any]] = []

    def _chat(messages, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "答案"}}]}

    monkeypatch.setattr(llm_client, "chat", _chat)
    agent = eval_run.build_agent(_agent_namespace([]))
    agent(_qa_probe_task(), {})

    assert len(calls) == 1
    assert calls[0]["base_url"] == llm_client.DEFAULT_BASE_URL
    assert calls[0]["api_key"] is None
    assert calls[0]["model"] == llm_client.DEFAULT_MODEL


def test_passing_the_agent_switches_changes_the_request(monkeypatch) -> None:
    """传了开关 ⇒ 真发出去的那次请求就改道到指定端点 + 指定密钥。"""
    from zh_agent_bench import eval_run, llm_client

    calls: list[dict[str, Any]] = []

    def _chat(messages, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "答案"}}]}

    monkeypatch.setattr(llm_client, "chat", _chat)
    monkeypatch.setenv("LLMAPI_GPT_API_KEY", "sk-from-env")
    agent = eval_run.build_agent(_agent_namespace([
        "--model", "gpt-6-astra",
        "--agent-base-url", "https://llmapi.example/v1",
        "--agent-key-env", "LLMAPI_GPT_API_KEY",
    ]))
    agent(_qa_probe_task(), {})

    assert len(calls) == 1
    assert calls[0]["base_url"] == "https://llmapi.example/v1"
    assert calls[0]["api_key"] == "sk-from-env"
    assert calls[0]["model"] == "gpt-6-astra"


def _qa_probe_task() -> Task:
    """一条最小 B 场景任务：不需要工具，agent 直接答。"""
    return Task(
        id="probe-0001", category=Category.RETRIEVAL, schema_version="1.1",
        instruction="测试", initial_data={}, tool_names=(),
        ground_truth={"example_answer": "示例"},
        answer_spec=AnswerSpec(kind=AnswerKind.STRING, comparison=Comparison.JUDGE,
                               normalization=Normalization.NONE),
        split=Split.TRAIN, rubric=(),
    )
