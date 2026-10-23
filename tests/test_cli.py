"""``--out-records`` 的测试（TASK-025）。

钉四件事（契约 §3 判据 1）：

1. **不带 ``--out-records`` 行为逐字不变** —— 一个文件都不许冒出来，榜单输出逐字节相同；
2. **带则写出合格的 jsonl** —— 行数=跑分条数、六个字段齐全、按 task_id 排序、中文不转义；
3. **写失败报错不静默** —— 父目录不存在直接 exit 2（榜单是对外材料，写不出来必须看得见）；
4. **与 ``leaderboard.aggregate_a`` 对接** —— 产物是 B/C 同构格式，榜单 A 列读得动。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zh_agent_bench.cli import (
    _HELDOUT_MANIFEST,
    build_heldout_manifest,
    check_heldout_allowed,
    main,
    verify_heldout_manifest,
    write_records,
)
from zh_agent_bench.leaderboard import aggregate_a, scan_products
from zh_agent_bench.tasks.schema import Split


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------------------
# 判据 1：不带 --out-records，老行为逐字不变
# --------------------------------------------------------------------------------------


def test_without_out_records_nothing_is_written(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["--heldout-only", "--allow-heldout", "--limit", "3"]) == 0
    assert list(tmp_path.iterdir()) == [], f"不该有任何产物，却有: {list(tmp_path.iterdir())}"


def test_out_records_does_not_change_the_leaderboard_text(tmp_path, capsys) -> None:
    """带不带 ``--out-records``，**榜单正文（stdout）必须逐字节相同** ——
    落盘是旁路，不能顺带改了 M1 的可复现口径。

    stderr 只多一行落盘路径提示（与 ``--trace`` 同构），原有的 schema/agent 那行不变。
    """
    assert main(["--heldout-only", "--allow-heldout", "--limit", "3"]) == 0
    plain = capsys.readouterr()
    assert main(["--heldout-only", "--allow-heldout", "--limit", "3",
                 "--out-records", str(tmp_path / "r.jsonl")]) == 0
    with_records = capsys.readouterr()
    assert plain.out == with_records.out
    extra = [ln for ln in with_records.err.splitlines() if ln not in plain.err.splitlines()]
    assert len(extra) == 1 and extra[0].startswith("records          :")


def test_out_records_does_not_change_the_trace_product(tmp_path, monkeypatch) -> None:
    """``--trace`` 与 ``--out-records`` 互不干扰：两份产物各自的行数、内容都不因对方而变。"""
    monkeypatch.chdir(tmp_path)
    assert main(["--heldout-only", "--allow-heldout", "--limit", "3", "--trace", "a.jsonl"]) == 0
    trace_alone = (tmp_path / "a.jsonl").read_bytes()
    assert main(["--heldout-only", "--allow-heldout", "--limit", "3", "--trace", "b.jsonl",
                 "--out-records", "r.jsonl"]) == 0
    assert (tmp_path / "b.jsonl").read_bytes() == trace_alone


# --------------------------------------------------------------------------------------
# 判据 2：产物本身合格
# --------------------------------------------------------------------------------------


def test_out_records_writes_one_line_per_scored_task(tmp_path) -> None:
    path = tmp_path / "r.jsonl"
    assert main(["--heldout-only", "--allow-heldout", "--limit", "5", "--out-records", str(path)]) == 0
    assert len(_rows(path)) == 5


def test_each_record_carries_the_fields_the_leaderboard_reads(tmp_path) -> None:
    """字段与 B/C 同构：``task_id/category/correct/G/score/reason`` + ``agent``/``judge`` 署名。

    少一个榜单就答不出那一格；少 ``agent`` 署名，这份产物就答不出「谁跑的」。
    """
    path = tmp_path / "r.jsonl"
    main(["--heldout-only", "--allow-heldout", "--limit", "4", "--out-records", str(path)])
    for row in _rows(path):
        assert set(row) == {"task_id", "category", "correct", "G", "score", "reason",
                            "agent", "judge"}
        assert isinstance(row["task_id"], str) and isinstance(row["category"], str)
        assert isinstance(row["correct"], bool)
        assert isinstance(row["reason"], str)
        assert row["agent"] == "reference-solver"


def test_g_is_null_because_a_scenario_has_no_llm_dimension(tmp_path) -> None:
    """A 是客观题 exact 比对，**没有 LLM 打分维度** ⇒ ``G`` 必须是 ``null``。

    填 0 或 1 会被下游读成「judge 打过了 / 打了 0 分」，那是 B 场景的语义。
    """
    path = tmp_path / "r.jsonl"
    main(["--heldout-only", "--allow-heldout", "--limit", "4", "--out-records", str(path)])
    assert [row["G"] for row in _rows(path)] == [None] * 4


def test_score_is_per_row_and_the_rate_comes_from_the_leaderboard(tmp_path) -> None:
    """``score`` 是**单条**的 1.0/0.0，不是整批的正确率 ——
    每行都写总分的话，「按行平均」和「整批相除」会给出两个不同的数。"""
    path = tmp_path / "r.jsonl"
    main(["--heldout-only", "--allow-heldout", "--out-records", str(path)])
    rows = _rows(path)
    assert set(row["score"] for row in rows) <= {0.0, 1.0}
    assert all(row["score"] == (1.0 if row["correct"] else 0.0) for row in rows)

    stat = aggregate_a(rows)
    assert stat["n"] == len(rows)
    assert stat["n_correct"] == sum(1 for r in rows if r["correct"])
    assert stat["rate"] == pytest.approx(
        sum(1 for r in rows if r["correct"]) / len(rows)
    )


def test_records_are_sorted_by_task_id(tmp_path) -> None:
    """按 task_id 排序 ⇒ 与数据集文件布局解耦，两次跑逐字节一致（可 diff、可进 git）。"""
    path = tmp_path / "r.jsonl"
    main(["--heldout-only", "--allow-heldout", "--out-records", str(path)])
    ids = [row["task_id"] for row in _rows(path)]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids), "task_id 重复了：同一题被跑了两次"


def test_records_are_utf8_with_lf_only_and_unescaped(tmp_path) -> None:
    path = tmp_path / "r.jsonl"
    main(["--heldout-only", "--allow-heldout", "--limit", "4", "--out-records", str(path)])
    data = path.read_bytes()
    assert b"\\u" not in data, "中文被转义成 \\uXXXX 了"
    assert b"\r\n" not in data


def test_two_runs_over_the_same_inputs_are_byte_identical(tmp_path) -> None:
    main(["--heldout-only", "--allow-heldout", "--out-records", str(tmp_path / "a.jsonl")])
    main(["--heldout-only", "--allow-heldout", "--out-records", str(tmp_path / "b.jsonl")])
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


def test_limit_bounds_the_records_the_same_way_it_bounds_the_run(tmp_path) -> None:
    path = tmp_path / "r.jsonl"
    main(["--heldout-only", "--allow-heldout", "--limit", "2", "--out-records", str(path)])
    assert len(_rows(path)) == 2


# --------------------------------------------------------------------------------------
# 判据 3：写失败不静默
# --------------------------------------------------------------------------------------


def test_a_missing_parent_directory_is_reported_not_created(tmp_path) -> None:
    """父目录不存在 ⇒ exit 2 且**不自动 mkdir**。

    与 ``--trace`` 相反：轨迹是事后查证用的，路径打错就该立刻看见；
    自动建目录会把「路径打错了」变成「静默写到别处去了」。
    """
    missing = tmp_path / "不存在" / "r.jsonl"
    assert main(["--heldout-only", "--allow-heldout", "--limit", "2", "--out-records", str(missing)]) == 2
    assert not missing.exists()
    assert not missing.parent.exists(), "不该自动 mkdir"


def test_the_failure_message_names_the_path(tmp_path, capsys) -> None:
    assert main(["--heldout-only", "--allow-heldout", "--limit", "1",
                 "--out-records", str(tmp_path / "无此目录" / "r.jsonl")]) == 2
    err = capsys.readouterr().err
    assert "records 落盘失败" in err and "无此目录" in err


# --------------------------------------------------------------------------------------
# 判据 4：与榜单对接
# --------------------------------------------------------------------------------------


def test_the_leaderboard_picks_this_file_over_the_calibration_one(tmp_path) -> None:
    """产物落在 ``benchmarks/office/heldout/<agent>/records.jsonl`` ⇒ 榜单 A 列读它。

    这正是本单存在的理由：没有它，榜单退到校准集（32 条混 example+corrupted）报一个
    压根不是 agent 能力分的数。新布局下榜单只扫 ``office/<split>/<agent>/``，
    校准集那个裸 ``benchmarks/calibration/records.jsonl`` 压根不在扫描范围内。
    """
    root = tmp_path / "benchmarks"
    records = root / "office/heldout/llm-deepseek-chat/records.jsonl"
    records.parent.mkdir(parents=True)
    (root / "calibration").mkdir(parents=True)
    (root / "calibration/records.jsonl").write_text(
        json.dumps({"task_id": "qa-0001", "judge_correct": True}) + "\n", encoding="utf-8"
    )

    rows = [
        {"task_id": f"stats-{i:04d}", "category": "stats", "correct": i < 3,
         "G": None, "score": 1.0 if i < 3 else 0.0, "reason": "r"}
        for i in range(4)
    ]
    records.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8", newline="\n",
    )

    scanned = scan_products(tmp_path, b_split="dev", c_preset="dev")
    source = scanned.agents["llm-deepseek-chat"]["A"]
    assert source == records, "榜单还指着校准集，A 列报的仍不是 agent 能力分"
    assert aggregate_a(_rows(source))["n_correct"] == 3


def test_write_records_is_usable_without_the_cli(tmp_path) -> None:
    """``write_records`` 直接可用 —— 跑分脚本不必经 argparse（层间依赖是单向的）。"""
    path = write_records(
        [{"task_id": "b", "category": "query", "correct": False, "reason": "错了"},
         {"task_id": "a", "category": "stats", "correct": True, "reason": "对了"}],
        tmp_path / "r.jsonl",
        agent_name="reference-solver",
    )
    assert [r["task_id"] for r in _rows(path)] == ["a", "b"]


# --------------------------------------------------------------------------------------
# 判据 5：产物路径带 agent 名（TASK-028）
# --------------------------------------------------------------------------------------


def test_a_records_file_gives_itself_to_whoever_ran_it(tmp_path) -> None:
    """每行都带 ``agent`` 署名，``judge`` 恒为 ``null``。

    A 是逐字比对、**根本不经过 judge** —— 写个 judge 名等于谎称这条分是模型判的。
    """
    path = write_records(
        [{"task_id": "a", "category": "stats", "correct": True, "reason": "对了"}],
        tmp_path / "r.jsonl", agent_name="llm-agent:deepseek-chat",
    )
    row = _rows(path)[0]
    assert row["agent"] == "llm-agent:deepseek-chat"
    assert row["judge"] is None


def test_a_directory_target_derives_the_agent_subdirectory(tmp_path) -> None:
    """``--out-records <目录>`` ⇒ 落 ``<目录>/<agent 安全名>/records.jsonl``。

    这是「两个 agent 跑同一档不互相覆盖」的落点：agent 名成了目录名的一部分。
    """
    path = write_records(
        [{"task_id": "a", "category": "stats", "correct": True, "reason": "对了"}],
        tmp_path / "benchmarks/office/heldout", agent_name="llm:deepseek-chat",
    )
    assert path == (tmp_path / "benchmarks/office/heldout/llm-deepseek-chat/records.jsonl")


def test_two_agents_writing_to_one_directory_do_not_overwrite_each_other(tmp_path) -> None:
    """**本单的主判据**：同一目录跑两个 agent，两份产物都在，先跑的没被盖掉。"""
    rows = [{"task_id": "a", "category": "stats", "correct": True, "reason": "对了"}]
    out_dir = tmp_path / "benchmarks/office/heldout"

    first = write_records(rows, out_dir, agent_name="llm:deepseek-chat")
    second = write_records(rows, out_dir, agent_name="reference-solver")

    assert first != second
    assert first.is_file() and second.is_file()
    assert _rows(first)[0]["agent"] == "llm:deepseek-chat"
    assert _rows(second)[0]["agent"] == "reference-solver"


# --------------------------------------------------------------------------------------
# 判据 A：heldout 封存闸（TASK-031）—— 默认 train，heldout 要显式 --allow-heldout
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [
    ["--heldout-only", "--limit", "3"],
    ["--split", "heldout", "--limit", "3"],
])
def test_heldout_is_refused_unless_allow_heldout_is_given(argv, capsys) -> None:
    assert main(argv) == 2
    assert "封存闸拒绝" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["--heldout-only", "--allow-heldout", "--limit", "3"],
    ["--split", "heldout", "--allow-heldout", "--limit", "3"],
])
def test_heldout_runs_when_the_flag_is_given(argv, capsys) -> None:
    assert main(argv) == 0
    assert "split=heldout" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["--heldout-only", "--limit", "3"],
    ["--split", "heldout", "--limit", "3"],
])
def test_the_refusal_says_which_split_and_how_to_unlock_it(argv, capsys) -> None:
    """报错必须点名 heldout **和** `--allow-heldout` —— 说不清怎么解就只能去翻源码。"""
    main(argv)
    err = capsys.readouterr().err
    assert "heldout" in err and "--allow-heldout" in err


def test_a_refusal_never_degrades_to_train(tmp_path, monkeypatch) -> None:
    """**本单最硬的一条**：heldout 被拒就 exit 2，**绝不偷偷跑 train**。

    静默降级比直接报错坏得多 —— 跑完一堆 dev 分数，命令却报成功，
    榜单上那份「heldout 成绩」其实是 train 跑出来的，没人看得出来。
    """
    monkeypatch.chdir(tmp_path)
    import zh_agent_bench.cli as cli_mod

    spy: list[Split] = []
    real = cli_mod.load_tasks

    def spy_load(root, split):
        spy.append(split)
        return real(root, split)

    monkeypatch.setattr(cli_mod, "load_tasks", spy_load)

    assert main(["--heldout-only", "--limit", "3"]) == 2
    assert spy == [], f"闸都没拦住就去读盘了: {spy}"


def test_no_arguments_runs_train(tmp_path, monkeypatch, capsys) -> None:
    """**默认改 train**：无参数 = dev 调试集，不是官方集。"""
    monkeypatch.chdir(tmp_path)
    import zh_agent_bench.cli as cli_mod

    spy: list[Split] = []
    real = cli_mod.load_tasks
    monkeypatch.setattr(cli_mod, "load_tasks",
                        lambda root, split: (spy.append(split), real(root, split))[1])

    assert main(["--limit", "3"]) == 0
    assert spy == [Split.TRAIN]
    assert "split=train" in capsys.readouterr().err


def test_split_train_says_the_same_thing_as_no_arguments(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    import zh_agent_bench.cli as cli_mod

    spy: list[Split] = []
    real = cli_mod.load_tasks
    monkeypatch.setattr(cli_mod, "load_tasks",
                        lambda root, split: (spy.append(split), real(root, split))[1])

    assert main(["--split", "train", "--limit", "3"]) == 0
    assert spy == [Split.TRAIN]
    assert "split=train" in capsys.readouterr().err


def test_check_heldout_allowed_only_gates_the_heldout_split() -> None:
    """train 是日常调试口径，不设闸；只有 heldout 被挡 —— 加闸不等于把所有入口锁死。"""
    check_heldout_allowed(Split.TRAIN, allow_heldout=False)
    check_heldout_allowed(Split.HELDOUT, allow_heldout=True)
    with pytest.raises(PermissionError):
        check_heldout_allowed(Split.HELDOUT, allow_heldout=False)


# --------------------------------------------------------------------------------------
# 判据 B：heldout 封存清单（TASK-031）
# --------------------------------------------------------------------------------------


def _fake_heldout_tree(base: Path) -> Path:
    """造一棵最小的 A 场景树：heldout 任务集 + heldout 产物各一份。"""
    (base / "benchmarks/tasks/heldout").mkdir(parents=True)
    (base / "benchmarks/tasks/heldout/query.jsonl").write_text("{}\n", encoding="utf-8")
    (base / "benchmarks/office/heldout/llm-agent").mkdir(parents=True)
    (base / "benchmarks/office/heldout/llm-agent/records.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    return base


def test_the_manifest_write_then_verify_round_trips(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _fake_heldout_tree(tmp_path)

    assert main(["--write-heldout-manifest"]) == 0
    assert _HELDOUT_MANIFEST.is_file()
    assert main(["--verify-heldout-manifest"]) == 0


def test_the_manifest_covers_the_task_set_and_the_products(tmp_path, monkeypatch) -> None:
    """清单要盖住两处：``tasks/heldout``（任务集）与 ``office/heldout``（跑分产物）——
    只盖一处等于另一半可以悄悄换掉。
    """
    monkeypatch.chdir(tmp_path)
    _fake_heldout_tree(tmp_path)

    payload = json.loads(build_heldout_manifest())
    assert set(payload) == {
        "benchmarks/tasks/heldout/query.jsonl",
        "benchmarks/office/heldout/llm-agent/records.jsonl",
    }


def test_a_tampered_heldout_task_is_reported_by_name(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _fake_heldout_tree(tmp_path)
    assert main(["--write-heldout-manifest"]) == 0
    capsys.readouterr()

    (tmp_path / "benchmarks/tasks/heldout/query.jsonl").write_text('{"改过了"}\n', encoding="utf-8")

    assert main(["--verify-heldout-manifest"]) == 2
    assert "query.jsonl" in capsys.readouterr().err


def test_a_missing_manifest_is_reported_not_crashed(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["--verify-heldout-manifest"]) == 2
    assert "读不了" in capsys.readouterr().err


def test_verify_reports_differences_without_running_any_task(tmp_path, monkeypatch, capsys) -> None:
    """校验只比 sha，**不该顺手跑一遍 heldout** —— 校验命令读封存集就够了，
    跑分是另一件事（还要 key、还要 agent）。
    """
    monkeypatch.chdir(tmp_path)
    _fake_heldout_tree(tmp_path)
    main(["--write-heldout-manifest"])
    capsys.readouterr()

    assert main(["--verify-heldout-manifest"]) == 0
    assert "split=" not in capsys.readouterr().err
