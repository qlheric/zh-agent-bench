"""bias 的离线测试：**全程不联网**（fake judge 注入）。

钉判据 2 的第 ⑤ 项：偏置配对的三类构造与翻转率计算。

关键点是**让假 judge 表现出特定的偏置**，然后断言报告能把它抓出来 ——
一个恒返回同一个 G 的假 judge 会让翻转率恒为 0、什么都能「通过」，
那样的测试等于没测。所以这里三���分别用：位置敏感、偏爱长文、
以及「被行话唬住」的假 judge。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from zh_agent_bench import bias
from zh_agent_bench.bias import (
    irrelevant_answer,
    irrelevant_pairs,
    length_pairs,
    load_answers,
    position_pairs,
    render_report,
    reversed_task,
    verbose_variant,
)
from zh_agent_bench.judge import PASS_THRESHOLD_G, build_judge_prompt
from zh_agent_bench.tasks.qa import build_qa_dataset
from zh_agent_bench.tasks.schema import Task


@pytest.fixture
def tasks() -> list[Task]:
    """4 条真任务（每类 1 条）。每条 rubric 都 ≥ 2 个评分点，位置偏置才有得翻。"""
    tasks = build_qa_dataset(per_type=1, seed=7)
    assert all(len(t.rubric) >= 2 for t in tasks)
    return tasks


# --------------------------------------------------------------------------------------
# 假 judge
# --------------------------------------------------------------------------------------


def _verdict(g: float, *, reason: str = "逐条 rubric 判定完成") -> dict:
    return {
        "correct": g >= PASS_THRESHOLD_G, "G": g, "F": 0.5, "T": 1.0,
        "capped": False, "scores": {}, "reason": reason,
    }


def _length_loving_judge(task: Task, answer: Any) -> dict:
    """越长的答案分越高（不论内容）。分档卡在 100 字：简答 6 字、冗长版 130 字，
    两侧必然分居两档 —— 这条常量与 :data:`bias.MIN_ADDED_CHARS` 是同一个量级。"""
    long = isinstance(answer, str) and len(answer) >= 100
    return _verdict(0.95 if long else 0.5)


def _fooled_judge(task: Task, answer: Any) -> dict:
    """被行话唬住：只要答案里出现本任务的**证据 id**就算答到。
    （:func:`bias.irrelevant_answer` 的行话正是这些 id，所以它骗得过这个假 judge。）"""
    hit = isinstance(answer, str) and any(
        doc_id in answer for point in task.rubric for doc_id in (point.get("evidence_ids") or [])
    )
    return _verdict(0.9 if hit else 0.3)


def _always(g: float) -> Any:
    def _run(_task: Task, _answer: Any) -> dict:
        return _verdict(g)
    return _run


def _failing(task: Task, _answer: Any) -> dict:
    return _verdict(0.0, reason="judge 调用失败：503 熔断")


def _true_versus_irrelevant(task: Task, answer: Any) -> dict:
    """分得出真答案与空话：0.9 vs 0.2。"""
    is_true = answer == (task.ground_truth or {}).get("example_answer")
    return _verdict(0.9 if is_true else 0.2)


def _raising(_task: Task, _answer: Any) -> dict:
    raise RuntimeError("连接被重置")


# --------------------------------------------------------------------------------------
# 1. 位置偏置：构造
# --------------------------------------------------------------------------------------


def _occurrences(text: str, needle: str) -> list[int]:
    """全部出现位置。``criterion_id`` 带 task_id 前缀（``qa-0001-c1``），
    只按后缀查会被别的 id 顺带命中 —— 所以全量返回再比序列。"""
    spots: list[int] = []
    at = text.find(needle)
    while at >= 0:
        spots.append(at)
        at = text.find(needle, at + 1)
    return spots


def test_reversed_task_only_flips_the_rubric_order(tasks):
    """副本只翻 rubric 顺序，**其余字段一字不动** —— 否则测的不再是位置偏置。"""
    task = tasks[0]
    rev = reversed_task(task)
    assert [p["criterion_id"] for p in rev.rubric] == \
        [p["criterion_id"] for p in reversed(task.rubric)]
    assert sorted(map(str, rev.rubric)) == sorted(map(str, task.rubric))   # 点本身没变
    assert rev.instruction == task.instruction
    assert rev.ground_truth == task.ground_truth
    assert rev.initial_data == task.initial_data
    assert task.rubric[0]["criterion_id"] != rev.rubric[0]["criterion_id"]  # 原件未被就地改


def test_reversed_task_actually_reverses_the_prompt_order(tasks):
    """顺序偏置只可能从 prompt 顺序进来。这条钉住「渲染确实跟着翻了」，
    而 :func:`judge` 里按 criterion_id 配权重 ⇒ G 的定义没变，只有读到的顺序变了。"""
    task = tasks[0]
    ids = [p["criterion_id"] for p in task.rubric]
    forward = _occurrences(build_judge_prompt(task, "答案")[1]["content"], ids[0])
    backward = _occurrences(build_judge_prompt(reversed_task(task), "答案")[1]["content"], ids[0])
    # 第一个点在逆序 prompt 里应当移到最后一个点在原序 prompt 里的位置（之后）
    assert len(forward) == len(backward) == 1        # 每个 id 在 prompt 里只出现一次
    assert forward[0] < backward[0]


# --------------------------------------------------------------------------------------
# 2. 位置偏置：翻转率
# --------------------------------------------------------------------------------------


def test_position_pairs_counts_flips_with_an_order_sensitive_judge(tasks):
    """假 judge 只看**第一个**评分点，别的点一律判 0 ⇒ 顺序一翻，结论就翻。

    敏感的是「首点答没答到」，所以靠答案文本触发（不靠 rubric 内容）——
    这样两侧唯一的差别就只有 rubric 的呈现顺序。
    """
    marker = "首点已覆盖"

    def _judge(task: Task, answer: Any) -> dict:
        # 判分**只取决于读到的头一个评分点是谁**：是原序的 c1（权重最高的那几个之一）
        # 才给高分，否则给低分。触发条件用答案文本而非 rubric 内容 ——
        # 两侧唯一的差别就只有呈现顺序。
        first_is_original_head = task.rubric[0]["criterion_id"].endswith("c1")
        hit = isinstance(answer, str) and marker in answer
        return _verdict(0.9 if (hit and first_is_original_head) else 0.4)

    answers = {t.id: f"{marker}。" for t in tasks}
    out = position_pairs(tasks, _judge, answers=answers, judge_name="fake-order-sensitive")
    assert out["n"] == len(tasks)
    assert out["n_usable"] == len(tasks)
    assert out["flip_rate"] == 1.0        # 每条都因顺序反转而翻转
    assert out["mean_delta"] < 0           # 逆序在前的是原序末尾的点 ⇒ 系统性掉分


def test_position_pairs_reports_zero_flip_for_an_order_blind_judge(tasks):
    out = position_pairs(tasks, _always(0.9))
    assert out["flip_rate"] == 0.0
    assert out["mean_delta"] == pytest.approx(0.0)


def test_position_pairs_separates_judge_failure_from_a_low_score(tasks):
    """judge 挂了**不是**「判低」—— 必须标 failed 并排除出翻转率。"""
    out = position_pairs(tasks, _failing)
    assert out["n_failed"] == len(tasks)
    assert out["n_usable"] == 0
    assert out["flip_rate"] is None
    assert out["mean_delta"] is None
    assert all(r["failed"] for r in out["rows"])


def test_position_pairs_never_lets_a_crash_escape(tasks):
    out = position_pairs(tasks, _raising)
    assert out["n_failed"] == len(tasks)
    assert out["flip_rate"] is None


def test_position_pairs_scores_a_non_empty_answer(tasks):
    """空答案测不出位置偏置（顺序对「什么都没答」无感）。没给答案时必须退回 example。"""
    def _judge(task: Task, answer: Any) -> dict:
        if not answer:
            return _verdict(0.0)
        return _verdict(0.9)

    out = position_pairs(tasks, _judge)
    assert out["n_usable"] == len(tasks)
    assert all(r["g_base"] == 0.9 for r in out["rows"])


def test_position_pairs_prefers_the_given_agent_answers(tasks):
    """给了 agent 答案就用它，不再退回 example_answer。"""
    calls: list[str] = []

    def _judge(task: Task, answer: Any) -> dict:
        calls.append(answer)
        return _verdict(0.9)

    answers = {t.id: f"这是 agent 给 {t.id} 的答案" for t in tasks}
    position_pairs(tasks, _judge, answers=answers)
    assert set(calls) == set(answers.values())


# --------------------------------------------------------------------------------------
# 3. 长度偏置：构造 + 差值
# --------------------------------------------------------------------------------------


def test_verbose_variant_lengthens_without_adding_facts(tasks):
    """冗长版必须**只加长度**。混进新事实会让长度实验退化成「加信息值不值」。"""
    answer = "延长试用期一个月，低于下限不予录用。"
    long = verbose_variant(answer, added_chars=120)
    assert long.startswith(answer)
    assert len(long) > len(answer) + 50
    for point in tasks[0].rubric:
        assert point["requirement"] not in long   # 没有把评分点原文抄进去


def test_verbose_variant_is_a_noop_on_empty_or_structured_answers():
    """空答案加填充段就是纯长度作弊；结构化答案硬拼字符串会改类型与判分路径。"""
    assert verbose_variant("") == ""
    assert verbose_variant("   ") == "   "
    assert verbose_variant(None) is None
    items = ["a", "b"]
    assert verbose_variant(items) is items


def test_length_pairs_measures_the_penalty_for_padding(tasks):
    answers = {t.id: "简短答案。" for t in tasks}
    out = length_pairs(tasks, answers, _length_loving_judge, judge_name="fake-length-loving")
    assert out["n"] == len(tasks)
    assert out["n_missing_answer"] == 0
    assert out["n_usable"] == len(tasks)
    assert out["mean_delta"] > 0.4          # 语义没变，只因为变长就多拿了 0.45
    assert out["flip_rate"] == 1.0
    assert all(r["len_verbose"] > r["len_base"] for r in out["rows"])


def test_length_pairs_skips_tasks_without_an_answer(tasks):
    answers = {tasks[0].id: "有答案"}
    out = length_pairs(tasks, answers, _length_loving_judge)
    assert out["n"] == 1
    assert out["n_missing_answer"] == len(tasks) - 1
    assert out["rows"][0]["task_id"] == tasks[0].id


def test_length_pairs_with_no_answers_is_not_a_clean_zero(tasks):
    """没答案 = 没数据，报告必须说「无数据」而不是报 0 条 / 翻转率 0。"""
    out = length_pairs(tasks, {}, _length_loving_judge)
    assert out["n"] == 0
    assert out["n_usable"] == 0
    assert out["flip_rate"] is None
    assert out["mean_delta"] is None


def test_length_pairs_reports_nothing_for_an_order_blind_judge(tasks):
    answers = {t.id: "简短答案。" for t in tasks}
    out = length_pairs(tasks, answers, _always(0.9))
    assert out["mean_delta"] == pytest.approx(0.0)
    assert out["flip_rate"] == 0.0


# --------------------------------------------------------------------------------------
# 4. 答非所问：构造 + 抗性
# --------------------------------------------------------------------------------------


def test_irrelevant_answer_borrows_this_task_own_vocabulary(tasks):
    """空话必须**像行话**才测得出抗性。凭空造不相干词，任何 judge 都会判 0。"""
    for task in tasks:
        answer = irrelevant_answer(task)
        assert len(answer) > 40
        assert "\n" not in answer                    # 无填充段，长度 confound 由 paired 侧控制
        # 行话取自**本任务**的证据 id
        for point in task.rubric:
            for doc_id in point.get("evidence_ids") or []:
                assert doc_id in answer


def test_irrelevant_answer_matches_the_verbose_variant_filler(tasks):
    """走同一个填充函数 ⇒ 「答非所问」与「冗长」之间没有长度这个混淆变量。"""
    task = tasks[0]
    plain = irrelevant_answer(task)
    padded = irrelevant_answer(task, added_chars=120)
    assert padded.startswith(plain)
    assert len(padded) > len(plain)


def test_irrelevant_pairs_flags_a_judge_that_is_fooled(tasks):
    """假 judge 只要见到文档 id 就给高分 ⇒ 空话分高于真答案，gap 为负、抗空话率 0。"""
    out = irrelevant_pairs(tasks, _fooled_judge, judge_name="fake-fooled")
    assert out["n"] == len(tasks)
    assert out["n_usable"] == len(tasks)
    assert out["mean_gap"] < 0
    assert out["resistance_rate"] == 0.0
    assert all(r["judge_got_it"] is False for r in out["rows"])
    assert out["mean_g_true"] == pytest.approx(0.3)


def test_irrelevant_pairs_treats_a_tie_as_no_resistance(tasks):
    """打平不算识破。恒定分的假 judge 靠 `>=` 就能把抗空话率刷成 100%。

    这是常量替身最容易糊弄过去的地方 —— 分数不变不代表抗空话，只代表测不出差别。
    """
    out = irrelevant_pairs(tasks, _always(0.3))
    assert out["mean_gap"] == pytest.approx(0.0)
    assert out["resistance_rate"] == 0.0
    assert all(r["judge_got_it"] is False for r in out["rows"])
    # 「谁都不给分」也一并报出来，否则抗空话率高会掩盖真答案被判 0 的事实
    assert out["mean_g_true_pass_rate"] == 0.0


def test_irrelevant_pairs_passes_a_discerning_judge(tasks):
    def _judge(task: Task, answer: Any) -> dict:
        is_true = answer == (task.ground_truth or {}).get("example_answer")
        return _verdict(0.9 if is_true else 0.2)

    out = irrelevant_pairs(tasks, _judge)
    assert out["resistance_rate"] == 1.0
    assert out["mean_gap"] > 0
    assert out["mean_g_true_pass_rate"] == 1.0
    assert all(r["judge_got_it"] for r in out["rows"])


def test_irrelevant_pairs_separates_judge_failure(tasks):
    out = irrelevant_pairs(tasks, _failing)
    assert out["n_failed"] == len(tasks)
    assert out["resistance_rate"] is None
    assert out["mean_gap"] is None


# --------------------------------------------------------------------------------------
# 5. 报告
# --------------------------------------------------------------------------------------


def _all(tasks, judge) -> tuple[dict, dict, dict]:
    return (
        position_pairs(tasks, judge),
        length_pairs(tasks, {t.id: "简短答案。" for t in tasks}, judge),
        irrelevant_pairs(tasks, judge),
    )


def test_render_report_covers_all_three_biases(tasks):
    pos, length, irr = _all(tasks, _fooled_judge)
    text = render_report(pos, length, irr, judge_name="fake-fooled")
    assert "位置偏置" in text
    assert "长度偏置" in text
    assert "答非所问" in text
    assert "抗空话失败" in text                      # 结论不是「正常」
    for task in tasks:
        assert task.id in text


def test_render_report_says_unmeasured_instead_of_claiming_clean(tasks):
    """judge 全挂时结论必须是「未测出来」，不是「无偏置」。"""
    pos, length, irr = _all(tasks, _failing)
    text = render_report(pos, length, irr, judge_name="fake-failing")
    assert text.count("未测出来") == 3
    assert "未检出" not in text


def test_render_report_flags_a_length_bias(tasks):
    """位置正常 / 长度有偏 / 抗空话正常 —— 三个结论各归各位，别互相串台。"""
    def _position_neutral(task: Task, answer: Any) -> dict:
        # 内容型判分：只看「有没有把评分点原文抄进答案」，与呈现顺序无关。
        hit = isinstance(answer, str) and any(p["requirement"][:8] in answer for p in task.rubric)
        return _verdict(0.9 if hit else 0.5)

    def _judge(task: Task, answer: Any) -> dict:
        # 长答案判高，忽略内容
        return _verdict(0.95 if isinstance(answer, str) and len(answer) >= 100 else 0.5)

    answers = {t.id: "简短答案。" for t in tasks}
    text = render_report(
        position_pairs(tasks, _position_neutral),
        length_pairs(tasks, answers, _judge),
        irrelevant_pairs(tasks, _true_versus_irrelevant),
        judge_name="fake",
    )
    assert "有长度偏置" in text
    assert "未检出明显位置偏置" in text
    assert "抗空话正常" in text


def test_render_report_is_deterministic(tasks):
    pos, length, irr = _all(tasks, _always(0.9))
    first = render_report(pos, length, irr, judge_name="fake")
    second = render_report(pos, length, irr, judge_name="fake")
    assert first == second
    assert "\r\n" not in first                       # 产物统一 LF
    assert f"{PASS_THRESHOLD_G}" in first


# --------------------------------------------------------------------------------------
# 6. 接线：--answers 读取 + CLI
# --------------------------------------------------------------------------------------


def test_load_answers_reads_task_id_and_answer(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"task_id": "a", "G": 1.0, "answer": "答案甲"}, ensure_ascii=False),
            json.dumps({"task_id": "b", "G": 0.5, "answer": None}, ensure_ascii=False),
            "",
        ]),
        encoding="utf-8",
    )
    answers = load_answers(path)
    assert answers == {"a": "答案甲"}     # answer=None 的条目不进（agent 没答出来）
    assert "b" not in answers


def test_load_answers_missing_file_is_empty_not_an_error(tmp_path):
    """缺文件 = 没数据，长度那一项会报「无可用数据」；不该在这里抛。"""
    assert load_answers(tmp_path / "没有.jsonl") == {}


def test_load_answers_without_the_answer_field_is_empty(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"task_id": "a", "G": 1.0}), encoding="utf-8")
    assert load_answers(path) == {}


def test_main_writes_both_artifacts(tasks, tmp_path, monkeypatch):
    monkeypatch.setattr(bias, "load_formal_split", lambda split, root=None: tasks)
    out = tmp_path / "out"
    assert bias.main(["--split", "trial", "--out", str(out)]) == 0
    assert (out / "bias.md").is_file()
    assert (out / "bias-results.json").is_file()
    payload = json.loads((out / "bias-results.json").read_text(encoding="utf-8"))
    assert set(payload) == {"position", "length", "irrelevant"}
    assert payload["position"]["n"] == len(tasks)


def test_main_without_records_still_writes_a_position_and_irrelevant_section(tasks, tmp_path, monkeypatch):
    """没有 --answers 时长度项是「无可用数据」，但另两项照跑 —— 不该整单失败。"""
    monkeypatch.setattr(bias, "load_formal_split", lambda split, root=None: tasks)
    out = tmp_path / "out"
    assert bias.main(["--split", "trial", "--out", str(out), "--limit", "2"]) == 0
    text = (out / "bias.md").read_text(encoding="utf-8")
    assert "无可用数据" in text
