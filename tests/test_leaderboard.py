"""leaderboard 的离线测试：**全程不读真实产物、不跑 agent**（临时目录里造三份 records）。

钉七件事（契约 §2.5）：

1. 三场景齐全 ⇒ 一张表，三列口径各按各的门槛算；
2. 缺场景显式「未跑」，**不许静默 0**（0 会被读成「跑了得零分」）；
3. 降级/失败**单列**，且不混进正确率；
4. 三条门槛写进表头（表头即口径，代码里藏着就等于没有）；
5. 确定性：同输入两次输出逐字节一致；
6. ``--scenario all`` 一个入口出一张表；
7. **一 agent 一行（TASK-028）**：两个 agent 同档各占一行；旧布局产物显式「未发现产物」。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from zh_agent_bench.judge import PASS_THRESHOLD_C, PASS_THRESHOLD_G
from zh_agent_bench.leaderboard import (
    SCENARIOS,
    aggregate_a,
    aggregate_b,
    aggregate_c,
    build_entry,
    main,
    render_markdown,
    scan_products,
    write_outputs,
)


# --------------------------------------------------------------------------------------
# 造产物
# --------------------------------------------------------------------------------------


def _a_record(correct: bool, degradation: str | None = None) -> dict[str, Any]:
    """A 场景（``calibrate`` 式）：键是 ``judge_correct`` / ``judge_G``，无 LLM 维度。"""
    return {
        "task_id": f"qa-{len(_A_ROWS):04d}",
        "task_type": "单跳",
        "judge_correct": correct,
        "judge_G": 1.0 if correct else 0.0,
        "degradation": degradation,
    }


def _b_record(correct: bool, *, G: float = 0.9, degradation: str | None = None,
              agent_failed: bool = False) -> dict[str, Any]:
    return {
        "task_id": f"qa-{_B_ROWS:04d}",
        "task_type": "单跳",
        "correct": correct,
        "G": G,
        "F": 1.0,
        "T": 0.5,
        "capped": False,
        "agent_failed": agent_failed,
        "judge_degradation": degradation,
    }


def _c_record(delta_ok: bool, reply_ok: bool, *, agent_failed: bool = False) -> dict[str, Any]:
    return {
        "task_id": f"orch-dev-{_C_ROWS:04d}",
        "task_type": "改地址",
        "correct": delta_ok and reply_ok,  # 产物里那个 correct 榜单**不读**，自己按口径算
        "delta_ok": delta_ok,
        "reply_ok": reply_ok,
        "orders_ok": delta_ok,
        "coupons_ok": delta_ok,
        "must_not_change_ok": True,
        "G": 0.9 if reply_ok else 0.2,
        "agent_failed": agent_failed,
    }


_A_ROWS: list[dict[str, Any]] = []
_B_ROWS = 0
_C_ROWS = 0


def _make_a(n: int = 8, n_correct: int = 6, n_degraded: int = 0) -> list[dict[str, Any]]:
    return [
        _a_record(i < n_correct, "llm_failure" if i < n_degraded else None)
        for i in range(n)
    ]


def _make_b(n: int = 10, n_correct: int = 7, n_degraded: int = 1,
            n_failed: int = 1) -> list[dict[str, Any]]:
    global _B_ROWS
    _B_ROWS = 0
    rows = []
    for i in range(n):
        degraded = "llm_failure" if i < n_degraded else None
        failed = i >= n - n_failed
        rows.append(_b_record(i < n_correct, degradation=degraded, agent_failed=failed))
        _B_ROWS += 1
    return rows


def _make_c(n: int = 16, n_delta: int = 12, n_reply: int = 10, n_failed: int = 2) -> list[dict[str, Any]]:
    global _C_ROWS
    _C_ROWS = 0
    rows = []
    for i in range(n):
        rows.append(_c_record(i < n_delta, i < n_reply, agent_failed=i >= n - n_failed))
        _C_ROWS += 1
    return rows


def _dump(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
        newline="\n",
    )
    return path


#: 测试里统一用的 agent 目录名。= ``safe_agent_dirname("llm:deepseek-chat")``，
#: 榜单的 agent 名**从目录取**，所以这里写什么，榜上就出现什么。
AGENT = "llm-deepseek-chat"


def _all_three(root: Path, agent: str = AGENT) -> None:
    """在三档齐全的仓库根里造齐 A/B/C 三份产物（``<场景>/<档>/<agent>/records.jsonl``）。"""
    _dump(root / f"benchmarks/office/heldout/{agent}/records.jsonl", _make_a())
    _dump(root / f"benchmarks/qa/dev/{agent}/records.jsonl", _make_b())
    _dump(root / f"benchmarks/orchestration/dev/{agent}/records.jsonl", _make_c())


def _sources(root: Path, agent: str = AGENT, **kwargs: str) -> dict[str, Path | None]:
    """:func:`scan_products` 里那个 agent 的来源映射 —— 榜单实际拿到的东西。"""
    # 缺省值必须能被 kwargs 覆盖：直接写 scan_products(root, b_split="dev", c_preset="dev", **kwargs)
    # 会在调用方传 c_preset="blind" 时撞成 "got multiple values for keyword argument"。
    scanned = scan_products(root, **{"b_split": "dev", "c_preset": "dev", **kwargs})
    return scanned.agents[agent]


def _entry(root: Path, agent: str = AGENT, **kwargs: str) -> dict[str, Any]:
    return build_entry(agent, _sources(root, agent, **kwargs))


# --------------------------------------------------------------------------------------
# 判据 1：三场景齐全 ⇒ 一张表
# --------------------------------------------------------------------------------------


def test_three_scenarios_land_in_one_row(tmp_path) -> None:
    _all_three(tmp_path)
    entry = _entry(tmp_path)

    assert entry["agent"] == AGENT
    assert entry["A"]["n"] == 8 and entry["A"]["n_correct"] == 6
    assert entry["B"]["n"] == 10 and entry["B"]["n_correct"] == 7
    assert entry["C"]["n"] == 16
    # C 的主榜 = delta_ok ∧ reply_ok：前 10 条两样都中 ⇒ 10，不是 delta 单独的 12
    assert entry["C"]["n_correct"] == 10

    text = render_markdown([entry])
    assert AGENT in text
    assert "6/8 = 75.00%" in text
    assert "7/10 = 70.00%" in text
    assert "10/16 = 62.50%" in text


def test_the_single_table_carries_every_required_column(tmp_path) -> None:
    """契约 §2.5 的表头逐列在（少一列这张表就答不出「它在哪一场景几分」）。"""
    _all_three(tmp_path)
    text = render_markdown([_entry(tmp_path)])

    for column in ("agent", "A 场景正确率", "B 场景正确率", "B G", "B F", "B T",
                   "C 场景正确率", "C delta_ok", "C reply_ok", "样本数", "降级/失败"):
        assert column in text, f"表头缺列：{column}"
    # 一个 agent 占一行。不能直接数 "\n| agent |"——渲染里有两张表（主表 + 样本来源表），
    # 表头都以 | agent | 开头；这里数主表独有的那一列，确保主表只有一个表头。
    assert text.count("| A 场景正确率") == 1


def test_the_sample_column_sums_all_ran_scenarios(tmp_path) -> None:
    _all_three(tmp_path)
    assert "34" in render_markdown([_entry(tmp_path)])


# --------------------------------------------------------------------------------------
# 判据 2：缺场景显式「未跑」
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("missing", SCENARIOS)
def test_a_missing_scenario_reads_not_run_not_zero(tmp_path, missing) -> None:
    """缺哪一场景，哪几列就是「未跑」—— **不是 0**。

    0 会被读成「跑了得零分」，那是两条完全不同的结论。
    """
    _all_three(tmp_path)
    if missing == "B":
        (tmp_path / f"benchmarks/qa/dev/{AGENT}/records.jsonl").unlink()
    elif missing == "C":
        (tmp_path / f"benchmarks/orchestration/dev/{AGENT}/records.jsonl").unlink()
    else:
        (tmp_path / f"benchmarks/office/heldout/{AGENT}/records.jsonl").unlink()

    entry = _entry(tmp_path)
    assert entry[missing] is None

    text = render_markdown([entry])
    assert "未跑" in text
    # 未跑的场景不贡献正确率，也不撑大样本数：齐全时 8+10+16=34，缺一档就少那一档的条数
    remaining = {"A": 8, "B": 10, "C": 16}
    expected = sum(n for s, n in remaining.items() if s != missing)
    assert f"| {expected} |" in text
    assert "| 34 |" not in text


def test_an_empty_records_file_is_not_a_zero_score(tmp_path) -> None:
    """文件在但 0 条 ≠ 跑了得 0 分。正确率格显示 ``0/0 = 未跑``，不是 ``0.00%``。"""
    _all_three(tmp_path)
    _dump(tmp_path / f"benchmarks/qa/dev/{AGENT}/records.jsonl", [])
    entry = _entry(tmp_path)

    assert entry["B"]["n"] == 0 and entry["B"]["rate"] is None
    assert "0/0 = 未跑" in render_markdown([entry])


def test_the_b_and_c_products_are_read_from_the_grade_the_caller_asked_for(tmp_path) -> None:
    """换档读另一份产物，不是永远读同一个目录。"""
    _all_three(tmp_path)
    _dump(tmp_path / f"benchmarks/orchestration/blind/{AGENT}/records.jsonl",
          _make_c(n=16, n_delta=16, n_reply=16))

    dev = build_entry(AGENT, _sources(tmp_path))
    blind = build_entry(AGENT, _sources(tmp_path, c_preset="blind"))
    assert dev["C"]["n_correct"] == 10
    assert blind["C"]["n_correct"] == 16


# --------------------------------------------------------------------------------------
# 判据 3：降级/失败单列，不混进正确率
# --------------------------------------------------------------------------------------


def test_degradation_and_failure_are_counted_apart_from_accuracy(tmp_path) -> None:
    """judge 降级与 agent 失败**单列**，且不影响主榜正确率的分子。"""
    _all_three(tmp_path)
    entry = _entry(tmp_path)
    text = render_markdown([entry])

    # B：1 条 judge 降级 + 1 条 agent 失败；C：2 条 agent 失败 ⇒ 合计 4
    assert entry["B"]["n_degraded"] == 1
    assert entry["B"]["n_agent_failed"] == 1
    assert entry["C"]["n_agent_failed"] == 2
    assert "| 4｜" in text
    assert "llm_failure" in text

    # 降级那几条没被算进正确率：B 的 7/10 分子里一条降级也没有
    assert "7/10 = 70.00%" in text


def test_the_degradation_column_names_the_kind(tmp_path) -> None:
    _dump(tmp_path / f"benchmarks/qa/dev/{AGENT}/records.jsonl", _make_b(n_degraded=3))
    assert "llm_failure×3" in render_markdown([_entry(tmp_path)])


def test_a_clean_run_reports_zero_degradation(tmp_path) -> None:
    _dump(tmp_path / f"benchmarks/qa/dev/{AGENT}/records.jsonl",
          _make_b(n=4, n_correct=4, n_degraded=0, n_failed=0))
    text = render_markdown([_entry(tmp_path)])
    assert "| 0 |" in text
    assert "×" not in text


# --------------------------------------------------------------------------------------
# 判据 4：三条门槛写进表头
# --------------------------------------------------------------------------------------


def test_the_thresholds_are_written_into_the_header_not_hidden_in_code(tmp_path) -> None:
    """B 的 G 门槛与 C 的回复点门槛必须**出现在渲染出来的那张表**里。

    三场景门槛各不相同，不写出来读者就得去翻代码才知道 70% 是什么意思。
    """
    _all_three(tmp_path)
    text = render_markdown([_entry(tmp_path)])
    assert f"G≥{PASS_THRESHOLD_G}" in text
    assert f"G≥{PASS_THRESHOLD_C}" in text
    assert "exact" in text
    assert "delta_ok∧reply_ok" in text


def test_b_accuracy_follows_the_g_threshold_not_the_raw_correct_field(tmp_path) -> None:
    """B 的主榜正确率按 ``correct``（= ``G ≥ 门槛`` 已判好的）统计，不按 G 的均值。

    G 均值是连续分，主榜是「过没过门槛」的二值 —— 两者混用会报出一个
    压根没人定的分数。
    """
    rows = _make_b(n=4, n_correct=2, n_degraded=0, n_failed=0)
    for row, G in zip(rows, (0.9, 0.85, 0.79, 0.5)):
        row["G"] = G
    stat = aggregate_b(rows)
    assert stat["n_correct"] == 2
    assert stat["rate"] == 0.5
    assert stat["G"] == pytest.approx((0.9 + 0.85 + 0.79 + 0.5) / 4)


def test_c_accuracy_is_delta_and_reply_not_the_records_own_correct_field() -> None:
    """C 的主榜**现算** ``delta_ok ∧ reply_ok``，不读产物里那个 ``correct``。

    口径写在榜单这一层 —— 读上一个写下的 ``correct`` 等于把口径藏起来，
    改判分口径的人不会想起来回来改榜单。
    """
    rows = [
        {"delta_ok": True, "reply_ok": True, "correct": False, "G": 0.9, "agent_failed": False},
        {"delta_ok": True, "reply_ok": False, "correct": True, "G": 0.1, "agent_failed": False},
    ]
    stat = aggregate_c(rows)
    assert stat["n_correct"] == 1  # 只认第一行；产物里那个 correct=True 的第二行不算
    assert stat["n_delta_ok"] == 2 and stat["n_reply_ok"] == 1


def test_a_accuracy_uses_the_exact_field_of_whichever_product_shape_is_present() -> None:
    """A 有两种产物格式，两种都能算，主榜口径都是 exact 判对。"""
    assert aggregate_a([{"judge_correct": True}, {"judge_correct": False}])["n_correct"] == 1
    assert aggregate_a([{"correct": True}, {"correct": True}])["n_correct"] == 2


def test_a_scenario_reports_no_llm_dimensions(tmp_path) -> None:
    """A 场景没有 G/F/T（客观题不是 LLM 打的分）⇒ 渲染成 ``n/a`` 而不是 0。"""
    _dump(tmp_path / f"benchmarks/office/heldout/{AGENT}/records.jsonl", _make_a())
    entry = _entry(tmp_path)
    stat = entry["A"]
    assert stat["G"] is None and stat["F"] is None and stat["T"] is None
    assert "n/a" in render_markdown([entry])


# --------------------------------------------------------------------------------------
# 判据 5：确定性
# --------------------------------------------------------------------------------------


def test_two_runs_over_the_same_inputs_are_byte_identical(tmp_path) -> None:
    """同输入两次输出**逐字节一致**（无时间戳、无字典序漂移）。"""
    _all_three(tmp_path)
    sources = _sources(tmp_path)

    first = build_entry(AGENT, sources)
    second = build_entry(AGENT, sources)
    assert render_markdown([first]) == render_markdown([second])

    _, json_a = write_outputs([first], tmp_path / "out1")
    _, json_b = write_outputs([second], tmp_path / "out2")
    assert json_a.read_bytes() == json_b.read_bytes()
    assert (tmp_path / "out1/leaderboard.md").read_bytes() == (
        tmp_path / "out2/leaderboard.md"
    ).read_bytes()


def test_the_output_contains_no_timestamp(tmp_path) -> None:
    _all_three(tmp_path)
    md_path, json_path = write_outputs([_entry(tmp_path)], tmp_path / "out")
    for path in (md_path, json_path):
        text = path.read_text(encoding="utf-8")
        assert "20" + "26-" not in text and "生成时间" not in text


# --------------------------------------------------------------------------------------
# 判据 6：一个入口出榜
# --------------------------------------------------------------------------------------


def test_the_cli_writes_both_products_and_exits_zero(tmp_path) -> None:
    _all_three(tmp_path)
    assert main(["--root", str(tmp_path), "--out-dir", str(tmp_path / "榜"),
                 "--b-split", "dev", "--c-preset", "dev"]) == 0
    assert (tmp_path / "榜/leaderboard.md").is_file()
    assert (tmp_path / "榜/leaderboard.json").is_file()


def test_the_cli_succeeds_even_when_every_scenario_is_missing(tmp_path) -> None:
    """一个场景都没跑 ⇒ **仍出榜**，三列全是「未跑」，exit 0。

    「没跑」不是命令失败，把 exit code 用掉，调用方就没法区分
    「还没跑」和「跑了但读不到产物」。
    """
    assert main(["--root", str(tmp_path), "--out-dir", str(tmp_path / "榜")]) == 0
    text = (tmp_path / "榜/leaderboard.md").read_text(encoding="utf-8")
    assert text.count("未跑") >= 3


def test_the_cli_can_publish_a_single_scenario(tmp_path) -> None:
    _all_three(tmp_path)
    assert main(["--root", str(tmp_path), "--out-dir", str(tmp_path / "榜"),
                 "--scenario", "B", "--b-split", "dev"]) == 0
    data = json.loads((tmp_path / "榜/leaderboard.json").read_text(encoding="utf-8"))
    entry = data["agents"][0]
    assert entry["B"]["n"] == 10
    # 只要一个场景就别装出另两个的键
    assert "A" not in entry and "C" not in entry


def test_the_cli_rejects_an_unknown_scenario(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["--root", str(tmp_path), "--scenario", "D"])


def test_a_corrupted_records_file_is_reported_not_skipped(tmp_path) -> None:
    """坏行报错，不静默丢 —— 静默丢记录等于报了一个偏乐观的分数。"""
    _all_three(tmp_path)
    (tmp_path / f"benchmarks/qa/dev/{AGENT}/records.jsonl").write_text(
        "{ 这不是 json\n", encoding="utf-8", newline="\n"
    )
    assert main(["--root", str(tmp_path), "--out-dir", str(tmp_path / "榜"),
                 "--b-split", "dev", "--c-preset", "dev"]) == 2


# --------------------------------------------------------------------------------------
# 判据 7：一 agent 一行（TASK-028）
# --------------------------------------------------------------------------------------


def test_two_agents_in_the_same_grade_become_two_rows(tmp_path) -> None:
    """**本单的主判据**：两个 agent 跑同一档各占一行，互不覆盖、互不影响。

    一 agent 一行之前「横评」在结构上做不到 —— 表只有一行，跑第二个必然盖掉第一个。
    """
    _all_three(tmp_path, agent="agent-a")
    _dump(tmp_path / f"benchmarks/qa/dev/agent-b/records.jsonl",
          _make_b(n=10, n_correct=3, n_degraded=0, n_failed=0))

    scanned = scan_products(tmp_path, b_split="dev", c_preset="dev")
    assert sorted(scanned.agents) == ["agent-a", "agent-b"]

    entries = [build_entry(name, sources) for name, sources in scanned.agents.items()]
    text = render_markdown(entries)

    assert "| agent-a |" in text and "| agent-b |" in text
    # 两个 agent 各自的 B 分数都在，谁也没被谁盖掉
    assert "7/10 = 70.00%" in text
    assert "3/10 = 30.00%" in text


def test_a_second_agent_only_runs_one_scenario_shows_not_run_for_the_rest(tmp_path) -> None:
    """只跑了一个场景的 agent，那几格写「未跑」—— 不是 0，也不是抄别人的分。"""
    _all_three(tmp_path, agent="agent-a")
    _dump(tmp_path / "benchmarks/orchestration/dev/agent-b/records.jsonl", _make_c())

    scanned = scan_products(tmp_path, b_split="dev", c_preset="dev")
    entries = [build_entry(name, sources) for name, sources in scanned.agents.items()]
    row_b = next(e for e in entries if e["agent"] == "agent-b")

    assert row_b["C"] is not None and row_b["A"] is None and row_b["B"] is None
    text = render_markdown(entries)
    assert "| agent-b | 未跑 | 未跑 |" in text


def test_the_agent_name_comes_from_the_directory_not_the_command_line(tmp_path) -> None:
    """agent 名**从目录取**。``--agent`` 只是个过滤器，冒充不了署名。"""
    _all_three(tmp_path)
    assert main(["--root", str(tmp_path), "--out-dir", str(tmp_path / "榜"),
                 "--b-split", "dev", "--c-preset", "dev",
                 "--agent", "somebody-else"]) == 2  # 目录里没这个 agent，不出榜
    assert not (tmp_path / "榜/leaderboard.md").exists()

    assert main(["--root", str(tmp_path), "--out-dir", str(tmp_path / "榜"),
                 "--b-split", "dev", "--c-preset", "dev",
                 "--agent", AGENT]) == 0
    text = (tmp_path / "榜/leaderboard.md").read_text(encoding="utf-8")
    assert f"| {AGENT} |" in text


def test_a_legacy_product_without_an_agent_directory_is_reported_not_counted(tmp_path) -> None:
    """旧布局（档目录下直接躺 ``records.jsonl``）显式「未发现产物」，**不进榜**。

    契约 §2.5：读不到就显式报，不许静默当 0。旧产物没有 agent 目录，
    猜不出是谁跑的 —— 把它塞进某一行等于编一个参赛者。
    """
    legacy = tmp_path / "benchmarks/qa/dev/records.jsonl"
    _dump(legacy, _make_b())

    scanned = scan_products(tmp_path, b_split="dev", c_preset="dev")
    assert scanned.agents == {}          # 一个 agent 都没扫到
    assert scanned.legacy["B"] == [legacy]

    text = render_markdown([], SCENARIOS, scanned.legacy)
    assert "未发现产物" in text
    assert "7/10 = 70.00%" not in text   # 旧产物的分数一个都没进榜
    assert "benchmarks/qa/dev/records.jsonl" in text  # 但路径得能被看见


def test_a_new_product_and_a_legacy_one_side_by_side(tmp_path) -> None:
    """新旧并存：agent 照常上榜，旧产物另起一行写「未发现产物」，两边都不含糊。"""
    _all_three(tmp_path)
    legacy = tmp_path / "benchmarks/orchestration/dev/records.jsonl"
    _dump(legacy, _make_c(n=16, n_delta=4, n_reply=4))

    scanned = scan_products(tmp_path, b_split="dev", c_preset="dev")
    entries = [build_entry(name, sources) for name, sources in scanned.agents.items()]
    text = render_markdown(entries, SCENARIOS, scanned.legacy)

    assert f"| {AGENT} |" in text
    assert "（旧布局·未纳入榜单）" in text
    # 新产物出分，旧产物那行 C 列是「未发现产物」而不是新产物的分
    assert "10/16 = 62.50%" in text
    assert "未发现产物" in text


def test_the_json_payload_is_one_object_per_agent(tmp_path) -> None:
    _all_three(tmp_path, agent="agent-a")
    _dump(tmp_path / f"benchmarks/qa/dev/agent-b/records.jsonl", _make_b(n_correct=3))

    scanned = scan_products(tmp_path, b_split="dev", c_preset="dev")
    _, json_path = write_outputs(
        [build_entry(n, s) for n, s in scanned.agents.items()], tmp_path / "out"
    )
    data = json.loads(json_path.read_text(encoding="utf-8"))

    assert [e["agent"] for e in data["agents"]] == ["agent-a", "agent-b"]
    assert data["legacy"] == {}  # 没有旧布局时就是个空 dict，不装出假条目


# --------------------------------------------------------------------------------------
# TASK-029：榜单侧封存闸（--b-split blind / --c-preset blind）
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--b-split", "--c-preset"])
def test_a_blind_grade_is_refused_unless_allow_blind_is_given(tmp_path, capsys, flag) -> None:
    """**判据 1①：榜单侧两个 blind 入口缺省 exit 2，且不落任何榜单产物。**

    榜单比 eval_run 更要紧 —— 它是「对外交分」那一面。闸漏了，
    开发期随手 `leaderboard --b-split blind` 就能把盲测分抄进对外表格。
    """
    _all_three(tmp_path)
    out_dir = tmp_path / "榜"
    assert main(["--root", str(tmp_path), "--out-dir", str(out_dir), flag, "blind"]) == 2
    assert not out_dir.exists()

    err = capsys.readouterr().out
    assert "blind" in err and "--allow-blind" in err
    assert main(["--root", str(tmp_path), "--out-dir", str(out_dir),
                 flag, "blind", "--allow-blind"]) == 0
    assert (out_dir / "leaderboard.md").is_file()


def test_the_refusal_never_degrades_to_dev(tmp_path) -> None:
    """被拒时**不许悄悄读 dev 档顶替** —— 榜上出的分必须是原来那一档的。

    钉的是「一个字节都不读」：不落榜单、不扫产物。
    静默降级比拒绝更坏：exit 0 的榜会被当真榜转出去。
    """
    _all_three(tmp_path)
    out_dir = tmp_path / "榜"
    assert main(["--root", str(tmp_path), "--out-dir", str(out_dir),
                 "--b-split", "blind"]) == 2
    assert not out_dir.exists()


def test_the_leaderboard_gate_does_not_touch_the_non_blind_grades(tmp_path) -> None:
    """**判据 1④：dev/trial 照常出榜。** 闸只拦 blind 一档。"""
    _all_three(tmp_path)
    assert main(["--root", str(tmp_path), "--out-dir", str(tmp_path / "榜"),
                 "--b-split", "dev", "--c-preset", "dev"]) == 0
    assert main(["--root", str(tmp_path), "--out-dir", str(tmp_path / "榜2")]) == 0
