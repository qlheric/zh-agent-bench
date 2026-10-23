"""轨迹落盘测试 —— TASK-007 判据 1。

落盘的轨迹是**事后唯一的证据**：「这条为什么答错」只能靠回看它当时调了哪个工具、
传了什么参、工具回了什么错。所以这组测试盯的不是「文件生成了」，而是
「落盘的内容够不够回答那个问题」，以及「它是否可复现」。

四条判据逐条对应：
- ① 每行可 ``json.loads``，含七个字段        → `test_each_line_is_valid_json_with_the_seven_fields`
- ② ``tool_calls`` 全量（arguments + result/error）→ `test_tool_calls_record_arguments_and_result_in_full`
- ③ 同输入两次逐字节一致                      → `test_two_runs_are_byte_identical`
- ④ 不传 ``--trace`` 时不落盘、行为不变        → `test_without_trace_nothing_is_written`
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zh_agent_bench.cli import load_tasks, main
from zh_agent_bench.harness import evaluate, run_trajectory, write_trace
from zh_agent_bench.reference import solve
from zh_agent_bench.tasks.schema import Category, Normalization, Task

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "benchmarks" / "tasks"


def _task(category=Category.STATS, truth="300", initial_data=None) -> Task:
    from zh_agent_bench.tasks.schema import AnswerKind, AnswerSpec, LEGAL_COMPARISONS, Split

    kind = {Category.STATS: AnswerKind.NUMBER, Category.FORMAT: AnswerKind.JSON}[category]
    return Task(
        id="stats-9001",
        category=category,
        schema_version="1.0",
        instruction="算平均销售额",
        initial_data=initial_data if initial_data is not None else {"rows": [{"销售额": 1}]},
        tool_names=("table.stats",),
        ground_truth={"answer": truth},
        answer_spec=AnswerSpec(
            kind=kind, comparison=LEGAL_COMPARISONS[kind], normalization=Normalization.NONE
        ),
        split=Split.TRAIN,
    )


def _sum_agent(task, tools):
    tools["table.stats"]({"table": [{"销售额": 100}], "column": "销售额", "op": "sum"})
    tools["table.stats"]({"table": [{"销售额": 200}], "column": "销售额", "op": "sum"})
    return "300"


def _failing_agent(task, tools):
    """调一次就踩到工具错误，然后放弃 —— 用来验证 error 分支落盘。"""
    tools["table.stats"]({"table": [{"销售额": 100}], "column": "不存在", "op": "avg"})
    return None


# --------------------------------------------------------------------------------------
# ① 每行合法 json + 七个字段
# --------------------------------------------------------------------------------------


def test_each_line_is_valid_json_with_the_seven_fields(tmp_path):
    results = evaluate([_task()], _sum_agent)
    path = write_trace(results, tmp_path / "trace.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert set(row) == {
        "task_id", "category", "tool_calls", "final_answer", "correct", "reason",
    }
    assert row["task_id"] == "stats-9001"
    assert row["category"] == "stats"
    assert row["final_answer"] == "300"
    assert row["correct"] is True
    assert isinstance(row["reason"], str)


def test_trace_is_utf8_with_lf_only_and_no_escapes(tmp_path):
    """中文不转义（``\\uXXXX``）、换行只 LF —— 与数据集同一套落盘口径（findings §二）。"""
    path = write_trace(evaluate([_task()], _sum_agent), tmp_path / "trace.jsonl")
    raw = path.read_bytes()
    assert b"\\u" not in raw, "中文被转义成 \\uXXXX 了"
    assert b"\r\n" not in raw
    assert "销售额".encode("utf-8") in raw


def test_one_line_per_task_in_input_order(tmp_path):
    tasks = load_tasks(DATA_ROOT, __import__("zh_agent_bench").tasks.schema.Split.HELDOUT)
    path = write_trace(evaluate(tasks, solve), tmp_path / "trace.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["task_id"] for r in rows] == [t.id for t in tasks]


# --------------------------------------------------------------------------------------
# ② tool_calls 全量
# --------------------------------------------------------------------------------------


def test_tool_calls_record_arguments_and_result_in_full(tmp_path):
    """传了什么参、工具回了什么，**原样全量**落盘。

    早期版本只留 ``payload_keys`` / ``result_keys``，省体积但回答不了落盘的目的：
    看到「这条错了」，查不出当时传了什么。这里断言的是真值本身，不是「有这个键」。
    """
    path = write_trace(evaluate([_task()], _sum_agent), tmp_path / "trace.jsonl")
    calls = json.loads(path.read_text(encoding="utf-8"))["tool_calls"]
    assert len(calls) == 2
    assert calls[0]["tool"] == "table.stats"
    assert calls[0]["arguments"] == {"table": [{"销售额": 100}], "column": "销售额", "op": "sum"}
    assert calls[0]["result"] == {"result": 100}, "结果必须原样落盘，不能只剩键名"
    assert calls[1]["result"] == {"result": 200}


def test_failed_tool_call_keeps_its_error_code(tmp_path):
    """失败的调用把 ``error`` 与原始 code 一起落盘 —— 这是排障时最该看见的那一格。

    TASK-007 的直接动机：RESULT-006 的 ``stats-0004`` 里模型说「工具持续报同样的错误」，
    但当时**没人能看到那个错误是什么**（error 被 check_contract 吞成了「缺 result」）。
    """
    path = write_trace(evaluate([_task(truth="1.00")], _failing_agent), tmp_path / "trace.jsonl")
    row = json.loads(path.read_text(encoding="utf-8"))
    call = row["tool_calls"][0]
    assert call["arguments"]["column"] == "不存在"
    assert call["error"]["code"] == "unknown_column", call
    assert call["result"] == {"error": call["error"]}
    assert row["correct"] is False


def test_trace_survives_values_json_cannot_encode(tmp_path):
    """final_answer 是 agent 随便给的，可能压根不是 JSON 可序列化的东西。

    落盘不能因为一条烂轨迹把整批判测炸掉 —— judge 铁律「烂答案绝不抛异常」
    在落盘这一层同样适用（否则最后一条轨迹的畸形答案会让前面 23 条白跑）。
    """
    path = write_trace(
        evaluate([_task(truth="x")], lambda task, tools: {"a", "b"}),  # 集合不可 JSON 序列化
        tmp_path / "trace.jsonl",
    )
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["final_answer"], "退化后的值也要留下来，不能整个丢掉"
    assert isinstance(row["final_answer"], str)


# --------------------------------------------------------------------------------------
# ③ 同输入两次逐字节一致
# --------------------------------------------------------------------------------------


def test_two_runs_are_byte_identical(tmp_path):
    """同输入两次落盘逐字节一致 —— 产物要能进 git、能 diff、能当证据引用。"""
    tasks = load_tasks(DATA_ROOT, __import__("zh_agent_bench").tasks.schema.Split.HELDOUT)
    first = write_trace(evaluate(tasks, solve), tmp_path / "a.jsonl").read_bytes()
    second = write_trace(evaluate(tasks, solve), tmp_path / "b.jsonl").read_bytes()
    assert first == second


def test_trace_has_no_timestamp_or_memory_address(tmp_path):
    """产物里不含时间戳/内存地址，否则每次 diff 都是噪音。"""
    raw = write_trace(evaluate([_task()], _sum_agent), tmp_path / "t.jsonl").read_text("utf-8")
    assert "0x" not in raw
    assert "20" + "26-" not in raw


def test_write_trace_creates_missing_parent_directories(tmp_path):
    """父目录不存在就自己建 —— 命令行给的路径不该要求用户先 mkdir。"""
    path = write_trace(evaluate([_task()], _sum_agent), tmp_path / "deep" / "a" / "trace.jsonl")
    assert path.exists()


# --------------------------------------------------------------------------------------
# ④ 不传 --trace 时不落盘、行为不变
# --------------------------------------------------------------------------------------


def test_without_trace_nothing_is_written(tmp_path, monkeypatch):
    """默认不落盘：不传 ``--trace`` 就一个文件都不许冒出来。"""
    monkeypatch.chdir(tmp_path)
    assert main(["--heldout-only", "--allow-heldout", "--limit", "2"]) == 0
    assert list(tmp_path.iterdir()) == [], f"不该有任何产物，却有: {list(tmp_path.iterdir())}"


def test_with_trace_writes_exactly_one_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["--heldout-only", "--allow-heldout", "--limit", "2", "--trace", "out/t.jsonl"]) == 0
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert files == ["out/t.jsonl"]
    rows = (tmp_path / "out" / "t.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2


def test_trace_does_not_change_the_leaderboard(tmp_path, capsys):
    """带不带 ``--trace``，榜单输出必须逐字节相同 —— 落盘是旁路，不能影响口径。"""
    assert main(["--heldout-only", "--allow-heldout", "--limit", "4"]) == 0
    plain = capsys.readouterr().out
    assert main(["--heldout-only", "--allow-heldout", "--limit", "4", "--trace", str(tmp_path / "t.jsonl")]) == 0
    assert capsys.readouterr().out == plain


# --------------------------------------------------------------------------------------
# 产物本身
# --------------------------------------------------------------------------------------


def test_committed_trace_example_is_still_reproducible(tmp_path):
    """入库的轨迹样例必须能被现生成的轨迹复现 —— 与数据集那条锁同构。

    没有这条，`benchmarks/results/**` 里的样例会随代码演进而悄悄过期，
    而它恰恰是对外展示「轨迹长什么样」的唯一实物。
    """
    import json as _json

    from zh_agent_bench.reference import REFERENCE_AGENT_NAME  # noqa: F401  (确认可用)

    committed = REPO_ROOT / "benchmarks" / "results" / "trace-sample.jsonl"
    if not committed.exists():
        pytest.skip("仓库里没有轨迹样例文件（首次跑 --trace 时生成）")
    tasks = load_tasks(DATA_ROOT, __import__("zh_agent_bench").tasks.schema.Split.HELDOUT)[:2]
    regen = write_trace(evaluate(tasks, solve), tmp_path / "t.jsonl").read_bytes()
    assert regen == committed.read_bytes()[: len(regen)]
