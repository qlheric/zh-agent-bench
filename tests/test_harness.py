"""harness 测试 —— 「轨迹 → 判分 → 榜单」闭环的验收。

重点四件事：
1. **工具只按任务分发**（不给 agent 全量工具，否则「该用哪个工具」这道考题就废了）；
2. **调用走 check_contract**（findings.md §四.5：契约漂移当场暴露）；
3. **榜单确定性**（同输入同输出，判据 4）；
4. **参考 agent 在真实 heldout 上 100%**（任务可解性的证明，判据 2）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zh_agent_bench import harness as harness_module
from zh_agent_bench import sandbox as sandbox_module
from zh_agent_bench.cli import load_tasks, main
from zh_agent_bench.harness import (
    DEFAULT_AGENT_NAME,
    _SandboxToolBox,
    evaluate,
    evaluate_orchestration,
    leaderboard,
    run_orchestration_trajectory,
    run_trajectory,
    sandbox_from_snapshot,
    summarize,
)
from zh_agent_bench.sandbox import tool_functions
from zh_agent_bench.reference import REFERENCE_AGENT_NAME, solve
from zh_agent_bench.tasks.schema import (
    LEGAL_COMPARISONS,
    A_SCENE_CATEGORIES,
    AnswerKind,
    AnswerSpec,
    Category,
    Comparison,
    Normalization,
    Split,
    Task,
)
from zh_agent_bench.tools.registry import TOOL_CONTRACT_VERSION, TOOL_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "benchmarks" / "tasks"


def _spec(kind: AnswerKind) -> AnswerSpec:
    return AnswerSpec(
        kind=kind,
        comparison=LEGAL_COMPARISONS[kind],
        normalization=Normalization.STRIP,
    )


def _task(category: Category, tool_names: tuple[str, ...], truth, initial_data=None) -> Task:
    """造一条任务。truth 直接填进 ground_truth，只用于测 harness 的分发/记录逻辑。

    只覆盖 A 场景四类（``A_SCENE_CATEGORIES`` 同名单）。B 场景的 ``RETRIEVAL`` 走
    rubric + judge 判分，harness 这一档（A 场景闭环）不参与，故不在此表内。
    """
    kind = {
        Category.QUERY: AnswerKind.STRING,
        Category.STATS: AnswerKind.NUMBER,
        Category.ORGANIZE: AnswerKind.ORDERED_LIST,
        Category.FORMAT: AnswerKind.JSON,
    }[category]
    return Task(
        id=f"{category.value}-9001",
        category=category,
        schema_version="1.0",
        instruction="测试",
        initial_data=initial_data or {},
        tool_names=tool_names,
        ground_truth={"answer": truth},
        answer_spec=_spec(kind),
        split=Split.TRAIN,
    )


# --------------------------------------------------------------------------------------
# 工具分发
# --------------------------------------------------------------------------------------


def test_toolbox_only_exposes_declared_tools():
    """铁律：agent 只看得到任务声明的工具。"""
    seen = {}

    def agent(task, tools):
        seen["contains_stats"] = "table.stats" in tools
        try:
            tools["table.stats"]
        except KeyError as exc:
            seen["keyerror"] = str(exc)
        return None

    task = _task(Category.QUERY, ("table.query",), "x")
    run_trajectory(task, agent)
    assert seen["contains_stats"] is False
    assert "table.stats" in seen["keyerror"]


def test_toolbox_call_goes_through_check_contract():
    """坏输入必须被契约校验拦下（ValueError），而不是默默调进工具里。"""
    def agent(task, tools):
        with pytest.raises(ValueError, match="不符合契约"):
            tools["table.stats"]({"table": [], "column": "销售额"})  # 缺 op
        return "1.00"

    run_trajectory(_task(Category.STATS, ("table.stats",), "1.00"), agent)


def test_toolbox_records_every_call():
    def agent(task, tools):
        tools["table.stats"]({"table": [{"销售额": 100}], "column": "销售额", "op": "sum"})
        tools["table.stats"]({"table": [{"销售额": 200}], "column": "销售额", "op": "sum"})
        return "300"

    result = run_trajectory(_task(Category.STATS, ("table.stats",), "300"), agent)
    assert len(result["tool_calls"]) == 2
    assert result["tool_calls"][0] == {
        "tool": "table.stats",
        "arguments": {"table": [{"销售额": 100}], "column": "销售额", "op": "sum"},
        "result": {"result": 100},
    }


def test_toolbox_records_arguments_and_result_in_full():
    """轨迹**全量**记录参数与结果，不只留键名。

    早期版本只留 ``payload_keys`` / ``result_keys``，省体积但**回答不了落盘的目的**：
    事后看到「这条错了」，却查不出当时到底传了什么参、工具回了什么。
    现在 ``arguments`` 与 ``result`` 都是原样全量（findings §七.3 / RESULT-007）。
    """
    def agent(task, tools):
        tools["table.stats"]({"table": [{"销售额": 1}], "column": "销售额", "op": "count"})
        return "1"

    result = run_trajectory(_task(Category.STATS, ("table.stats",), "1"), agent)
    call = result["tool_calls"][0]
    assert call["arguments"] == {"table": [{"销售额": 1}], "column": "销售额", "op": "count"}
    assert call["result"] == {"result": 1}
    assert "error" not in call, "成功调用不该有 error 键"


def test_toolbox_records_the_error_branch_of_a_failed_call():
    """失败的调用要把 error 单独提出来（``error`` 键），且 result 保持工具原样。

    这是落盘轨迹最要紧的一个字段：「模型为什么答错」十有八九在这里 ——
    哪个工具、传了什么参、工具说了什么 code。
    """
    def agent(task, tools):
        tools["table.stats"]({"table": [{"销售额": 1}], "column": "不存在", "op": "avg"})
        return None

    result = run_trajectory(_task(Category.STATS, ("table.stats",), "1.00"), agent)
    call = result["tool_calls"][0]
    assert call["tool"] == "table.stats"
    assert call["arguments"]["column"] == "不存在"
    assert call["error"]["code"] == "unknown_column", call["error"]
    assert call["result"] == {"error": call["error"]}, "result 应是工具原样返回的那份"


def test_run_trajectory_return_shape():
    def agent(task, tools):
        return "客户服务部"

    result = run_trajectory(_task(Category.QUERY, ("table.query",), "客户服务部"), agent)
    assert set(result) == {"task_id", "category", "tool_calls", "final_answer", "correct", "reason"}
    assert result["correct"] is True
    assert result["category"] == "query"


def test_run_trajectory_propagates_agent_exception():
    """agent 崩了必须原样上抛 —— 不能吞成「答错」，否则 agent 的 bug 和答案错分不清。"""
    def agent(task, tools):
        raise RuntimeError("agent 自己炸了")

    with pytest.raises(RuntimeError, match="agent 自己炸了"):
        run_trajectory(_task(Category.QUERY, ("table.query",), "x"), agent)


# --------------------------------------------------------------------------------------
# evaluate / summarize
# --------------------------------------------------------------------------------------


def test_evaluate_preserves_input_order():
    tasks = [_task(c, (), None) for c in (Category.FORMAT, Category.QUERY, Category.STATS)]
    results = evaluate(tasks, lambda task, tools: None)
    assert [r["task_id"] for r in results] == ["format-9001", "query-9001", "stats-9001"]


def test_summarize_counts():
    results = evaluate(
        [_task(Category.QUERY, (), "客户服务部"), _task(Category.STATS, (), "1.00")],
        lambda task, tools: {"query": "客户服务部", "stats": None}[task.category.value],
    )
    stats = summarize(results)
    assert stats["n_correct"] == 1
    assert stats["total"] == 2
    assert stats["score"] == 0.5
    assert stats["per_category"] == {
        "query": {"n_correct": 1, "total": 1},
        "stats": {"n_correct": 0, "total": 1},
    }


def test_summarize_empty_input_does_not_divide_by_zero():
    assert summarize([])["score"] == 0.0


# --------------------------------------------------------------------------------------
# 榜单
# --------------------------------------------------------------------------------------


#: 每类的「正确答案」，用来造全对榜（None 只是占位，够跑通 harness）。
_TRUTH_BY_KIND = {
    AnswerKind.STRING: "客户服务部",
    AnswerKind.NUMBER: "1.00",
    AnswerKind.ORDERED_LIST: ["甲", "乙"],
    AnswerKind.JSON: {"表名": "商品销售表", "rows": []},
}


def _passing_task(category: Category) -> Task:
    kind = {
        Category.QUERY: AnswerKind.STRING,
        Category.STATS: AnswerKind.NUMBER,
        Category.ORGANIZE: AnswerKind.ORDERED_LIST,
        Category.FORMAT: AnswerKind.JSON,
    }[category]
    return _task(category, (), _TRUTH_BY_KIND[kind])


def test_leaderboard_is_pure_ascii():
    """榜单框架纯 ASCII：本机控制台代码页非 UTF-8，中文表头会乱码（findings.md §二.2）。

    只有 ``missed`` 明细行带中文 —— 那是失败原因，信息量在中文，不值得为它把表头也汉化。
    全对时（参考 agent 的常态）整块榜单就是纯 ASCII。
    """
    passing = leaderboard(evaluate(
        [_passing_task(c) for c in A_SCENE_CATEGORIES], lambda task, tools: task.ground_truth["answer"],
    ))
    assert "100.00%" in passing, "先确认这确实是全对榜，下面这条断言才有意义"
    assert passing.isascii(), "全对榜单含非 ASCII 字符"

    failing = leaderboard(evaluate([_task(Category.QUERY, (), "客户服务部")], lambda t, tl: "研发部"))
    header = failing.split("missed")[0]
    assert header.isascii(), "missed 之前的框架部分含非 ASCII 字符"


def test_leaderboard_lists_categories_in_enum_order():
    """按 Category 枚举序，不是字典插入序 —— 否则榜单顺序会随任务顺序漂。"""
    tasks = [_task(c, (), None) for c in (Category.FORMAT, Category.QUERY)]
    text = leaderboard(evaluate(tasks, lambda task, tools: None))
    assert text.index("query") < text.index("format")


def test_leaderboard_reports_misses_with_reason():
    tasks = [_task(Category.QUERY, (), "客户服务部")]
    text = leaderboard(evaluate(tasks, lambda task, tools: "研发部"))
    assert "missed (1)" in text
    assert "query-9001" in text


def test_leaderboard_is_byte_identical_across_runs():
    """判据 4：同输入同输出，逐字节一致。"""
    tasks = [_task(c, (), None) for c in A_SCENE_CATEGORIES]
    runs = [leaderboard(evaluate(tasks, lambda task, tools: None)) for _ in range(3)]
    assert runs[0].encode() == runs[1].encode() == runs[2].encode()


def test_leaderboard_has_no_timestamp_or_address():
    """榜单不含时间戳/内存地址等易变内容。"""
    text = leaderboard(evaluate([_task(Category.QUERY, (), "a")], lambda t, tl: None))
    assert "0x" not in text
    assert "20" + "26-" not in text  # 不含日期


# --------------------------------------------------------------------------------------
# 数据加载 + 参考 agent（判据 2）
# --------------------------------------------------------------------------------------


def test_load_tasks_only_reads_requested_split():
    heldout = load_tasks(DATA_ROOT, Split.HELDOUT)
    assert heldout, "heldout 数据集读不到"
    assert all(t.split is Split.HELDOUT for t in heldout)
    # 入库的 A 场景数据集覆盖四类；B 场景数据在 benchmarks/qa/，不在这个目录里。
    assert {t.category for t in heldout} == set(A_SCENE_CATEGORIES)


def test_heldout_and_train_entity_pools_are_disjoint():
    """防污染（findings.md §一）：train/heldout 的人名实体池不相交。"""
    def names(split: Split) -> set[str]:
        out = set()
        for task in load_tasks(DATA_ROOT, split):
            if task.category is not Category.QUERY:
                continue
            for row in task.initial_data.get("rows", []):
                out.add(row.get("姓名"))
        return out

    train_names = names(Split.TRAIN)
    heldout_names = names(Split.HELDOUT)
    assert train_names & heldout_names == set(), "train/heldout 的人名池有交集，防污染失效"


def test_reference_agent_scores_100_on_heldout():
    """判据 2：参考 agent 在 heldout 上全对 —— 任务可解、工具与 judge 自洽。"""
    tasks = load_tasks(DATA_ROOT, Split.HELDOUT)
    results = evaluate(tasks, solve)
    missed = [(r["task_id"], r["reason"]) for r in results if not r["correct"]]
    assert missed == [], f"参考 agent 漏了: {missed}"
    assert summarize(results)["score"] == 1.0


def test_reference_agent_scores_100_on_train():
    """train 也全对 —— 说明 100% 不是 heldout 撞运气调出来的。"""
    tasks = load_tasks(DATA_ROOT, Split.TRAIN)
    results = evaluate(tasks, solve)
    missed = [(r["task_id"], r["reason"]) for r in results if not r["correct"]]
    assert missed == [], f"参考 agent 漏了: {missed}"


def test_reference_agent_never_reads_ground_truth():
    """硬约束：参考求解器不能读 ground_truth，否则 100% 就是自欺。

    做法是把每条任务的 ground_truth 换成毒值再跑一遍。**分数必须掉到 0** ——
    求解器算出的答案与毒值毫无关系，全错正说明它从头到尾没碰过 ground_truth。
    反过来若仍全对，那就是读了真值再吐出来，100% 毫无意义。
    """
    tasks = load_tasks(DATA_ROOT, Split.HELDOUT)
    poisoned = [
        Task(**{**t.__dict__, "ground_truth": {"answer": "__POISON__"}}) for t in tasks
    ]
    results = evaluate(poisoned, solve)
    assert not any(r["correct"] for r in results), "ground_truth 被污染后仍判对 —— 说明它读了 ground_truth"
    assert all(r["final_answer"] != "__POISON__" for r in results)
    # 反向自检：同一批任务在正常真值下确实是全对的，否则上面那条恒真、没有区分力。
    assert all(r["correct"] for r in evaluate(tasks, solve))


def test_reference_agent_only_calls_declared_tools():
    """参考 agent 调的每个工具都必须在任务的 tool_names 里（防「偷偷用了别的工具」）。"""
    tasks = load_tasks(DATA_ROOT, Split.HELDOUT)
    for result in evaluate(tasks, solve):
        declared = set(next(t for t in tasks if t.id == result["task_id"]).tool_names)
        assert {c["tool"] for c in result["tool_calls"]} <= declared


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def test_cli_heldout_only_exits_zero(capsys):
    assert main(["--heldout-only", "--allow-heldout"]) == 0
    out = capsys.readouterr()
    assert "100.00%" in out.out
    assert "24/24" in out.out  # heldout 每类 6 条 × 4 类


def test_cli_default_is_train(capsys):
    """不给参数就跑 train（dev 调试集）—— heldout 是封存口径，默认不许读。

    TASK-031 起默认从 heldout 改成 train；这条测试原来叫 ``test_cli_default_is_heldout``
    且只断言 ``"24/24" in out``，在 train 集下分类小计恰好也是 24/24，会**假通过**。
    改成直接看 stderr 的 split 与条数。
    """
    assert main([]) == 0
    err = capsys.readouterr().err
    assert "split=train" in err and "n=96" in err


def test_cli_split_train_limit(capsys):
    assert main(["--split", "train", "--limit", "8"]) == 0
    assert "n=8" in capsys.readouterr().err


def test_cli_rejects_bad_limit():
    with pytest.raises(SystemExit):
        main(["--limit", "0"])


def test_cli_output_is_byte_identical_across_runs(capsys):
    main(["--heldout-only", "--allow-heldout"])
    first = capsys.readouterr().out
    main(["--heldout-only", "--allow-heldout"])
    second = capsys.readouterr().out
    assert first.encode() == second.encode()


def test_cli_loads_every_task_file():
    """四类各一个文件、每个文件 2 行 —— 少读一个文件榜单就会缺类别。"""
    files = sorted(p.name for p in (DATA_ROOT / "heldout").glob("*.jsonl"))
    assert files == ["format.jsonl", "organize.jsonl", "query.jsonl", "stats.jsonl"]


def test_dataset_files_are_valid_jsonl():
    for path in sorted((DATA_ROOT / "heldout").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                Task.from_dict(json.loads(line))  # 解析失败会当场抛


# --------------------------------------------------------------------------------------
# 架构约束
# --------------------------------------------------------------------------------------


def test_runtime_has_no_third_party_import():
    """TASK-002 铁律的延续：判分/harness 链路也只许 stdlib + 本包。"""
    import ast

    root = REPO_ROOT / "src" / "zh_agent_bench"
    allowed = {"json", "re", "sys", "argparse", "pathlib", "dataclasses", "enum",
               "decimal", "typing", "collections", "abc", "__future__", "hashlib"}
    for name in ("judge.py", "harness.py", "reference.py", "cli.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in allowed, f"{name} 引入第三方: {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                assert node.module.split(".")[0] in allowed, f"{name} 引入第三方: {node.module}"


def test_leaderboard_pins_tool_contract_version():
    """榜单要报工具契约版本 —— 工具升级后老分数不可比，榜单必须自报家门。"""
    text = leaderboard(evaluate([_task(Category.QUERY, (), "a")], lambda t, tl: "a"))
    assert TOOL_CONTRACT_VERSION in text


def test_reference_agent_name_matches_hardcoded_board():
    """榜单默认署的名必须等于 reference.py 的常量，别各写一份。"""
    assert REFERENCE_AGENT_NAME == "reference-solver"
    assert DEFAULT_AGENT_NAME == REFERENCE_AGENT_NAME, (
        "leaderboard 的默认 agent 名必须还是 reference-solver —— "
        "它是 M1 的判分可复现口径，默认值一变老分数就不可比了"
    )
    text = leaderboard(evaluate([_task(Category.QUERY, (), "a")], lambda t, tl: "a"))
    assert REFERENCE_AGENT_NAME in text


def test_leaderboard_omitting_agent_name_is_unchanged_by_the_new_parameter():
    """加参数**不许**改变现有调用点的输出（M1 口径不动，findings §七.2）。

    显式传默认名与不传，必须逐字节一致 —— 这条挡住的是「顺手把默认改成了别的东西」。
    """
    results = evaluate([_passing_task(c) for c in A_SCENE_CATEGORIES], lambda t, tl: t.ground_truth["answer"])
    assert leaderboard(results).encode() == leaderboard(results, DEFAULT_AGENT_NAME).encode()


def test_leaderboard_reports_the_agent_name_it_is_given():
    """榜单正文署真名：`--agent llm` 时显示 `llm-agent:{model}` 而不是 `reference-solver`。

    TASK-006 之前这里写死 reference-solver，于是 LLM 跑出来的榜正文署着别人的名字 ——
    榜单第一行就是「这榜是谁跑的」，署名错了这张榜就不能当证据引用。
    """
    results = evaluate([_task(Category.QUERY, (), "a")], lambda t, tl: "a")
    text = leaderboard(results, agent_name="llm-agent:deepseek-chat")
    assert "agent            : llm-agent:deepseek-chat" in text
    assert "reference-solver" not in text, "传了别的 agent 名，榜单还在署默认名"


def test_leaderboard_agent_name_appears_exactly_once():
    """只出现一次：正文署名行改了，别在别处又冒出第二处署名。"""
    text = leaderboard(
        evaluate([_task(Category.QUERY, (), "a")], lambda t, tl: "a"),
        agent_name="llm-agent:deepseek-chat",
    )
    assert text.count("llm-agent:deepseek-chat") == 1


def test_cli_passes_the_real_agent_name_to_the_leaderboard(monkeypatch, capsys):
    """CLI 必须把 agent 名传进榜单 —— 上一版算好了 agent_name 却只打到 stderr。

    榜单正文才是给人看的那一份，stderr 的那行会被刷走。
    """
    monkeypatch.setattr(
        "zh_agent_bench.cli.make_llm_agent",
        lambda **kwargs: _FakeLlmAgent(kwargs.get("model")),
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    assert main(["--heldout-only", "--allow-heldout", "--agent", "llm",
                 "--model", "qwen-plus", "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "agent            : llm-agent:qwen-plus" in out, out


class _FakeLlmAgent:
    """只用于验证「CLI 把 agent.name 传进了榜单」—— 不联网、不跑真实调用。"""

    def __init__(self, model: str) -> None:
        self.model = model

    @property
    def name(self) -> str:
        return f"llm-agent:{self.model}"

    def __call__(self, task, tools):
        return task.ground_truth["answer"]


# --------------------------------------------------------------------------------------
# B 场景轨迹接线（TASK-010）
# --------------------------------------------------------------------------------------


def _retrieval_task(rubric: list[dict], initial_data: dict, tool_names=("knowledge.search",)) -> Task:
    """造一条 B 场景（rubric + judge）任务。judge 本体由调用方 monkeypatch。"""
    return Task(
        id="qa-9001",
        category=Category.RETRIEVAL,
        schema_version="1.1",
        instruction="测试",
        initial_data=initial_data,
        tool_names=tuple(tool_names),
        ground_truth={"example_answer": "示例答案"},
        answer_spec=AnswerSpec(
            kind=AnswerKind.STRING, comparison=Comparison.JUDGE, normalization=Normalization.NONE
        ),
        split=Split.TRAIN,
        rubric=tuple(rubric),
    )


def test_trajectory_reaches_judge_in_b_scene(monkeypatch):
    """**B 场景**：harness 跑出的 ``tool_calls`` 必须原样传到 judge —— T 与证据覆盖率
    全靠它，传丢了 judge 就只能判「不适用」。

    这条在 TASK-010 之前不成立：``run_trajectory`` 调的是 ``judge(answer, gt, spec)``，
    既没传 task 也没传轨迹，B 场景压根进不了 rubric 分支。
    """
    seen: dict[str, object] = {}

    def fake_judge_task(task, answer, trajectory=None):
        seen["trajectory"] = trajectory
        return {"correct": True, "reason": "fake"}

    monkeypatch.setattr("zh_agent_bench.harness.judge_task", fake_judge_task)

    def agent(task, tools):
        tools["knowledge.search"]({"query": "发布频率"})
        return "答案"

    task = _retrieval_task([], {"任务类型": "单跳"})
    run_trajectory(task, agent)

    traj = seen["trajectory"]
    assert isinstance(traj, list) and len(traj) == 1, "轨迹没传进 judge"
    assert traj[0]["tool"] == "knowledge.search", traj
    assert traj[0]["arguments"] == {"query": "发布频率"}


def test_b_scene_trajectory_carries_failed_calls_through(monkeypatch):
    """失败调用的 ``error`` 分支也必须在轨迹里 —— judge 的 T 就是靠它数非法调用。"""
    seen: dict[str, object] = {}

    def fake_judge_task(task, answer, trajectory=None):
        seen["trajectory"] = trajectory
        return {"correct": False, "reason": "fake"}

    monkeypatch.setattr("zh_agent_bench.harness.judge_task", fake_judge_task)

    def agent(task, tools):
        tools["table.stats"]({"table": [{"销售额": 1}], "column": "不存在", "op": "avg"})
        return "答案"

    run_trajectory(
        _retrieval_task([], {"任务类型": "单跳"}, tool_names=("table.stats",)), agent
    )
    assert seen["trajectory"][0]["error"]["code"], seen["trajectory"]


def test_a_scene_trajectory_does_not_change_the_verdict(monkeypatch):
    """**A 场景行为不变**：轨迹不影响结果，且仍走 ``judge`` 的客观比较算子。

    两条一起钉：传了轨迹后 A 场景分数不动（``judge_task`` 里 A 场景不读 trajectory），
    同时 ``judge`` 仍被调用（没被误绕到别的路径上）。
    """
    calls: list[str] = []
    real_judge_task = harness_module.judge_task

    def spy(task, answer, trajectory=None):
        calls.append("judge_task")
        return real_judge_task(task, answer, trajectory=trajectory)

    monkeypatch.setattr("zh_agent_bench.harness.judge_task", spy)

    def agent(task, tools):
        tools["table.query"]({"table": [{"部门": "客户服务部"}], "filters": {}})
        return "客户服务部"

    result = run_trajectory(_task(Category.QUERY, ("table.query",), "客户服务部"), agent)
    assert result["correct"] is True, "A 场景带轨迹判分结果不变"
    assert calls == ["judge_task"], "A 场景必须仍走 judge_task → judge 的客观比较路径"


def test_evaluate_passes_trajectory_for_every_b_task(monkeypatch):
    """整批 evaluate：每条任务各自的轨迹各传各的，不能串。"""
    seen: list[object] = []

    def fake_judge_task(task, answer, trajectory=None):
        seen.append((task.id, trajectory))
        return {"correct": True, "reason": "fake"}

    monkeypatch.setattr("zh_agent_bench.harness.judge_task", fake_judge_task)

    def make(task_id):
        return Task(
            id=task_id, category=Category.RETRIEVAL, schema_version="1.1", instruction="测试",
            initial_data={"任务类型": "单跳"}, tool_names=("knowledge.search",),
            ground_truth={"example_answer": "x"},
            answer_spec=AnswerSpec(kind=AnswerKind.STRING, comparison=Comparison.JUDGE,
                                   normalization=Normalization.NONE),
            split=Split.TRAIN, rubric=(),
        )

    evaluate([make("qa-1"), make("qa-2")], lambda task, tools: task.id)
    assert [tid for tid, _ in seen] == ["qa-1", "qa-2"]


def test_registry_has_exactly_four_tools():
    """A 场景四个工具 + B 场景的 ``knowledge.search``（TASK-008 起注册表是 5 个）。

    这条钉的是**harness 会把哪些工具发给 agent**，所以集合要写全，不能只写 A 场景那四个
    —— 只写四个等于放任第 5 个悄悄混进 A 场景 agent 的工具箱，那才是这道考题要防的事。
    """
    assert set(TOOL_REGISTRY) == {
        "table.query", "table.stats", "table.organize", "format.csv_to_json",
        "knowledge.search",
    }


# ======================================================================================
# C 场景：沙箱 → agent → 快照 → 判分
# ======================================================================================


def _c_task(task_id: str = "orch-0001", sandbox: dict | None = None) -> Task:
    if sandbox is None:
        sandbox = sandbox_module.snapshot(sandbox_module.make_sandbox(42))
    return Task(
        id=task_id, category=Category.ORCHESTRATION, schema_version="1.1",
        instruction="请处理这笔订单",
        initial_data={"任务类型": "改地址", "sandbox": sandbox},
        tool_names=("order.get", "order.change_address", "order.list", "order.refund"),
        ground_truth={"expected_final_state": {"orders": [], "coupons": []}},
        answer_spec=AnswerSpec(kind=AnswerKind.JSON, comparison=Comparison.ORCHESTRATION,
                               normalization=Normalization.NONE),
        split=Split.TRAIN, rubric=(),
    )


def test_sandbox_from_snapshot_rebuilds_a_detached_sandbox():
    """重建的沙箱与原快照**物理断开**：agent 的写操作不许污染 task.initial_data。

    不断开的话，同一个 Task 对象判第二次时环境已经不是出题时那个了 ——
    上一条退掉的单会让下一条的陷阱物理失效，分数随执行顺序变。
    """
    snap = sandbox_module.snapshot(sandbox_module.make_sandbox(42))
    sb = sandbox_from_snapshot(snap)
    tools = tool_functions(sb)
    target = sb.orders[0]["order_id"]
    tools["order.change_address"]({"order_id": target, "new_address": "新地址 A"})
    assert next(o for o in sb.orders if o["order_id"] == target)["address"] == "新地址 A"
    untouched = next(o for o in snap["orders"] if o["order_id"] == target)
    assert untouched["address"] != "新地址 A"


def test_sandbox_from_snapshot_rejects_a_malformed_snapshot():
    """坏快照当场报错，不是建出一个半残沙箱跑到一半才崩。"""
    with pytest.raises(ValueError, match="必须是沙箱快照字典"):
        sandbox_from_snapshot(["not", "a", "dict"])
    with pytest.raises(ValueError, match="缺字段"):
        sandbox_from_snapshot({"orders": []})


def test_the_agent_gets_functions_not_the_sandbox_object():
    """闭包隔离：agent 拿得到工具函数，拿不到 Sandbox 实例。

    拿得到就能直接 ``sb.orders.append(...)`` 绕过工具，副作用与顺序依赖就测不到了。
    """
    tools = tool_functions(sandbox_module.make_sandbox(42))
    assert all(callable(fn) for fn in tools.values())
    assert not any(isinstance(fn, sandbox_module.Sandbox) for fn in tools.values())


def test_the_sandbox_tool_box_only_exposes_declared_tools():
    """只暴露 task.tool_names 里声明的工具，与 A/B 同口径。"""
    box = _SandboxToolBox(("order.get",), tool_functions(sandbox_module.make_sandbox(42)))
    assert "order.get" in box
    assert "order.refund" not in box
    with pytest.raises(KeyError, match="任务未声明工具"):
        box["order.refund"]


def test_the_sandbox_tool_box_records_trajectory_and_errors():
    """轨迹全量记录，含失败时的 error —— 判分与事后复盘都靠它。"""
    sb = sandbox_module.make_sandbox(42)
    box = _SandboxToolBox(("order.get", "order.refund"), tool_functions(sb))
    box["order.get"]({"order_id": "SO-不存在"})
    unpaid = next(o["order_id"] for o in sb.orders if o["status"] != "paid")
    box["order.refund"]({"order_id": unpaid, "reason": "x"})
    assert [c["tool"] for c in box.calls] == ["order.get", "order.refund"]
    assert all(c["result"].get("error") == c["error"] for c in box.calls)


def test_each_orchestration_task_gets_a_fresh_sandbox():
    """每条任务一份全新沙箱：分数不许随执行顺序变。"""
    seen: list[str] = []

    def _probe(task, tools):
        tools["order.list"]({})
        seen.append(json.dumps(tools["order.list"]({}), ensure_ascii=False, sort_keys=True))
        return "查过了"

    tasks = [_c_task("orch-0001"), _c_task("orch-0002")]
    evaluate_orchestration(tasks, _probe, lambda *a, **k: {"correct": True, "reason": "fake"})
    assert len(seen) == 2


def test_an_orchestration_side_effect_does_not_leak_into_the_next_task():
    """上一条退掉的单会让下一条的陷阱物理失效 —— 两条任务不许共享环境。"""
    def _refund_first(task, tools):
        first = tools["order.list"]({})["orders"][0]
        tools["order.refund"]({"order_id": first["order_id"], "reason": "手滑"})
        return "退了"

    snap = sandbox_module.snapshot(sandbox_module.make_sandbox(42))
    tasks = [_c_task("orch-0001", json.loads(json.dumps(snap))),
             _c_task("orch-0002", json.loads(json.dumps(snap)))]
    rows = evaluate_orchestration(
        [tasks[0], tasks[1]], _refund_first,
        lambda *a, **k: {"correct": True, "reason": "fake"},
    )
    assert rows[0]["final_snapshot"] == rows[1]["final_snapshot"]


def test_run_orchestration_trajectory_hands_the_final_snapshot_to_the_judge():
    """闭环最后一跳：judge 拿到的是 agent 走完之后的沙箱快照，不是某个工具的返回值。"""
    captured: dict = {}

    def _capture(task, answer, trajectory=None, *, final_snapshot=None):
        captured["trajectory"] = trajectory
        captured["final_snapshot"] = final_snapshot
        return {"correct": True, "reason": "fake", "delta_ok": True, "reply_ok": True}

    def _agent(task, tools):
        target = tools["order.list"]({})["orders"][0]["order_id"]
        tools["order.change_address"]({"order_id": target, "new_address": "新地址 B"})
        return "改好了"

    row = run_orchestration_trajectory(_c_task(), _agent, _capture)
    assert [c["tool"] for c in captured["trajectory"]] == ["order.list", "order.change_address"]
    assert len([o for o in captured["final_snapshot"]["orders"]
                if o["address"] == "新地址 B"]) == 1
    assert row["correct"] is True


def test_orchestration_task_type_is_carried_into_the_record():
    """报告的分组靠 task_type，取不到标 unknown 而不是抛。"""
    row = run_orchestration_trajectory(
        _c_task(), lambda task, tools: "没动",
        lambda *a, **k: {"correct": True, "reason": "fake"},
    )
    assert row["task_type"] == "改地址"
    assert row["category"] == Category.ORCHESTRATION.value


def test_an_orchestration_agent_exception_is_not_swallowed_into_a_wrong_answer():
    """工具用错了该扣分、agent 崩了该修 bug —— 异常原样上抛，不吞成「答错」。"""
    def _boom(task, tools):
        raise RuntimeError("agent 崩了")

    with pytest.raises(RuntimeError, match="agent 崩了"):
        run_orchestration_trajectory(_c_task(), _boom,
                                     lambda *a, **k: {"correct": True, "reason": "fake"})


# --------------------------------------------------------------------------------------
# TASK-021：_SandboxToolBox.specs() —— schema 从 box 取，不再查全局注册表
# --------------------------------------------------------------------------------------


def _c_box(tool_names: tuple[str, ...]) -> _SandboxToolBox:
    sb = sandbox_module.make_sandbox(42)
    return _SandboxToolBox(tool_names, tool_functions(sb))


def test_sandbox_tool_box_specs_lists_only_the_declared_tools():
    """只列本任务 tool_names —— 给全量 6 个等于泄题（与 _ToolBox 同口径）。"""
    specs = _c_box(("order.get", "order.change_address")).specs()
    assert [s["name"] for s in specs] == ["order.get", "order.change_address"]
    assert len(specs) == 2


def test_sandbox_tool_box_specs_carry_the_sandbox_truth_source_schema():
    """parameters 必须**就是** sandbox.TOOL_SPECS 那份（同一对象，不是复述）。"""
    specs = _c_box(("order.refund",)).specs()
    assert specs[0]["parameters"] is sandbox_module.TOOL_SPECS["order.refund"]
    assert specs[0]["parameters"]["required"] == ["order_id", "reason"]
    assert specs[0]["description"]


def test_sandbox_tool_box_specs_survive_an_empty_declaration():
    """没有声明工具时返回空列表而不是抛 —— 空任务不该掀掉整批。"""
    assert _c_box(()).specs() == []
