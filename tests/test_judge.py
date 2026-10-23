"""判分器测试 —— M1 的命门，逐条钉死四种比较算子的语义。

判据要求覆盖：
- ``exact``：strip / 不 strip
- ``decimal_exact``：字符串 / Decimal / 不等
- ``ordered_list_exact``：顺序敏感 / 长度不等
- ``json_canonical``：键序无关 / 解析失败判错
- agent 答案 ``None`` / 类型不符 → 判错且**不抛异常**

最后一条是重点：判分器绝不能因为 agent 答得烂就崩，否则一批烂答案会让整批评测挂掉，
而那恰恰是最需要判分器工作的场景。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from zh_agent_bench.judge import judge, judge_task
from zh_agent_bench.tasks.schema import (
    AnswerKind,
    AnswerSpec,
    Comparison,
    Normalization,
    Task,
)


def _spec(kind: AnswerKind, normalization: Normalization = Normalization.NONE) -> AnswerSpec:
    """按 schema 的 LEGAL_COMPARISONS 造一个合法 spec（kind 决定 comparison）。"""
    from zh_agent_bench.tasks.schema import LEGAL_COMPARISONS

    return AnswerSpec(
        kind=kind,
        comparison=LEGAL_COMPARISONS[kind],
        normalization=normalization,
    )


def _judge(answer, truth, kind, normalization=Normalization.NONE):
    return judge(answer, {"answer": truth}, _spec(kind, normalization))


# --------------------------------------------------------------------------------------
# exact
# --------------------------------------------------------------------------------------


def test_exact_match():
    r = _judge("客户服务部", "客户服务部", AnswerKind.STRING)
    assert r["correct"] is True


def test_exact_mismatch():
    r = _judge("研发部", "客户服务部", AnswerKind.STRING)
    assert r["correct"] is False
    assert r["reason"]


def test_exact_with_strip_ignores_surrounding_whitespace():
    """normalization=strip 时首尾空白不算差异（数据集里 query 任务就是这个口径）。"""
    r = _judge("  客户服务部\n", "客户服务部", AnswerKind.STRING, Normalization.STRIP)
    assert r["correct"] is True


def test_exact_without_strip_is_whitespace_sensitive():
    """normalization=none 时空白有意义 —— 这就是 strip 与不 strip 的区分力。"""
    r = _judge("  客户服务部", "客户服务部", AnswerKind.STRING, Normalization.NONE)
    assert r["correct"] is False


def test_exact_strip_does_not_affect_inner_whitespace():
    """strip 只削首尾，不动中间的空格。"""
    r = _judge("客户 服务部", "客户 服务部", AnswerKind.STRING, Normalization.STRIP)
    assert r["correct"] is True


def test_exact_accepts_number_answer_as_text():
    """agent 把答案写成数字（如销售额）也算合法文本，按 str() 比。"""
    r = _judge(123, "123", AnswerKind.STRING)
    assert r["correct"] is True


# --------------------------------------------------------------------------------------
# decimal_exact
# --------------------------------------------------------------------------------------


def test_decimal_exact_string_vs_string():
    assert _judge("83337.90", "83337.90", AnswerKind.NUMBER)["correct"] is True


def test_decimal_exact_accepts_decimal_answer():
    """agent 常返回 Decimal；不能因为类型不同就判错。"""
    r = _judge(Decimal("83337.90"), "83337.90", AnswerKind.NUMBER)
    assert r["correct"] is True


def test_decimal_exact_accepts_int_answer():
    assert _judge(42, "42.00", AnswerKind.NUMBER)["correct"] is True


def test_decimal_exact_accepts_float_answer():
    """float 经 str() 再进 Decimal —— 保住人写的十进制意图，不吃二进制误差。"""
    assert _judge(0.1, "0.1", AnswerKind.NUMBER)["correct"] is True
    # 若直接 Decimal(0.1)，这条会挂 —— 0.1 的二进制值 ≠ 十进制 0.1
    assert Decimal(0.1) != Decimal("0.1")


def test_decimal_exact_ignores_trailing_zero_scale():
    r = _judge("83337.900", "83337.9", AnswerKind.NUMBER)
    assert r["correct"] is True


def test_decimal_exact_not_equal():
    r = _judge("86833.87", "86833.86", AnswerKind.NUMBER)
    assert r["correct"] is False


def test_decimal_exact_does_not_hide_float_artifacts():
    """agent 用 float 算 0.1+0.2 得 0.30000000000000004 —— 判分器**不**替它四舍五入掉。

    这是有意的：schema 的契约是「金额路径不出现 float」（findings.md §四.2），
    float 误差是 agent 的账实不符，该判错就该判错。判分器悄悄 ``quantize(0.01)``
    反而会把真 bug 藏起来，还让 `decimal_exact` 名不副实。
    """
    assert 0.1 + 0.2 != 0.3, "若 float 恰好相等，本用例失去区分力"
    r = _judge(0.1 + 0.2, "0.3", AnswerKind.NUMBER)
    assert r["correct"] is False
    assert "0.30000000000000004" in r["reason"]


def test_decimal_exact_accepts_clean_float():
    """干净的 float 仍能判对 —— 宽容的是「类型」，不是「错误」。"""
    assert _judge(0.1, "0.1", AnswerKind.NUMBER)["correct"] is True


def test_decimal_exact_rejects_garbage_string():
    r = _judge("很多钱", "83337.90", AnswerKind.NUMBER)
    assert r["correct"] is False


# --------------------------------------------------------------------------------------
# ordered_list_exact
# --------------------------------------------------------------------------------------


def test_ordered_list_exact_match():
    r = _judge(["周涛", "朱婷", "刘洋"], ["周涛", "朱婷", "刘洋"], AnswerKind.ORDERED_LIST)
    assert r["correct"] is True


def test_ordered_list_exact_is_order_sensitive():
    """这是本算子存在的理由：同样三个人，顺序不同就是错。"""
    r = _judge(["朱婷", "周涛", "刘洋"], ["周涛", "朱婷", "刘洋"], AnswerKind.ORDERED_LIST)
    assert r["correct"] is False
    assert "第 0 项" in r["reason"]


def test_ordered_list_exact_length_mismatch():
    r = _judge(["周涛", "朱婷"], ["周涛", "朱婷", "刘洋"], AnswerKind.ORDERED_LIST)
    assert r["correct"] is False
    assert "长度不等" in r["reason"]


def test_ordered_list_exact_accepts_tuple():
    r = _judge(("周涛", "朱婷"), ["周涛", "朱婷"], AnswerKind.ORDERED_LIST)
    assert r["correct"] is True


def test_ordered_list_exact_strips_items_under_strip_normalization():
    r = _judge([" 周涛 ", "朱婷"], ["周涛", "朱婷"], AnswerKind.ORDERED_LIST, Normalization.STRIP)
    assert r["correct"] is True


def test_ordered_list_rejects_string_answer():
    """agent 把列表写成 "周涛,朱婷" 是错的，不能因为字符串也能拆就放行。"""
    r = _judge("周涛,朱婷", ["周涛", "朱婷"], AnswerKind.ORDERED_LIST)
    assert r["correct"] is False


# --------------------------------------------------------------------------------------
# json_canonical
# --------------------------------------------------------------------------------------


def test_json_canonical_key_order_irrelevant():
    """不比原文：键序不同但内容相同，必须判对。"""
    answer = {"表名": "商品销售表", "rows": [{"姓名": "田甜", "城市": "西安"}]}
    truth = {"rows": [{"城市": "西安", "姓名": "田甜"}], "表名": "商品销售表"}
    r = _judge(answer, truth, AnswerKind.JSON)
    assert r["correct"] is True


def test_json_canonical_accepts_json_string_answer():
    """agent 常把 JSON 序列化成字符串再交回来；能 parse 就该判对。"""
    r = _judge('{"a": 1}', {"a": 1}, AnswerKind.JSON)
    assert r["correct"] is True


def test_json_canonical_not_equal():
    r = _judge({"a": 1}, {"a": 2}, AnswerKind.JSON)
    assert r["correct"] is False


def test_json_canonical_inner_key_order_irrelevant():
    r = _judge([{"x": 1, "y": 2}], [{"y": 2, "x": 1}], AnswerKind.JSON)
    assert r["correct"] is True


def test_json_canonical_list_order_is_significant():
    """顶层键序无关，但**数组顺序有关** —— JSON 数组是有序的。"""
    r = _judge([{"x": 1}, {"x": 2}], [{"x": 2}, {"x": 1}], AnswerKind.JSON)
    assert r["correct"] is False


def test_json_canonical_unparseable_answer_is_wrong():
    r = _judge("{不是json", {"a": 1}, AnswerKind.JSON)
    assert r["correct"] is False
    assert "JSON" in r["reason"]


# --------------------------------------------------------------------------------------
# 铁律：烂答案不抛异常
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [None, "", 0, [], {}, object(), b"bytes", 3.5, float("nan")])
@pytest.mark.parametrize("kind", list(AnswerKind))
def test_never_raises_on_garbage_answer(bad, kind):
    """**铁律**：任何垃圾答案都只返回 correct=False，不抛异常。"""
    r = _judge(bad, "真值", kind)
    assert r["correct"] is False
    assert isinstance(r["reason"], str) and r["reason"]


def test_missing_answer_key_is_wrong_not_crash():
    r = judge("某答案", {}, _spec(AnswerKind.STRING))
    assert r["correct"] is False


def test_ground_truth_none_is_wrong_not_crash():
    r = judge("某答案", {"answer": None}, _spec(AnswerKind.STRING))
    assert r["correct"] is False


def test_ground_truth_not_dict_is_wrong_not_crash():
    r = judge("某答案", ["不是字典"], _spec(AnswerKind.STRING))
    assert r["correct"] is False


def test_illegal_comparison_raises_valueerror():
    """数据集自身的 bug 必须当场暴露，不能被兜底 except 吞掉。

    ``AnswerSpec.from_dict`` 会挡掉 (kind, comparison) 不匹配的组合，但构造器直接 new 一个
    非法 comparison 是挡不住的 —— 所以 judge 层还得自己兜一道。
    """
    bad_spec = AnswerSpec(
        kind=AnswerKind.STRING,
        comparison="没实现的算子",  # type: ignore[arg-type]  # 绕过枚举，模拟脏数据
        normalization=Normalization.NONE,
    )
    with pytest.raises(ValueError):
        judge("x", {"answer": "x"}, bad_spec)


def test_return_shape_is_stable():
    """榜单与测试都依赖这四个键，别悄悄改名。"""
    r = _judge("a", "a", AnswerKind.STRING)
    assert set(r) == {"correct", "reason", "normalized_answer", "normalized_truth"}
    assert r["correct"] is True


# --------------------------------------------------------------------------------------
# judge_task 便捷封装 + 确定性
# --------------------------------------------------------------------------------------


def _task(kind: AnswerKind, truth) -> Task:
    return Task(
        id="t-0001",
        category=kind,  # type: ignore[arg-type]
        schema_version="1.0",
        instruction="随便",
        initial_data={},
        tool_names=(),
        ground_truth={"answer": truth},
        answer_spec=_spec(kind, Normalization.STRIP),
        split="train",  # type: ignore[arg-type]
    )


def test_judge_task_uses_task_ground_truth_and_spec():
    task = _task(AnswerKind.ORDERED_LIST, ["a", "b"])
    r = judge_task(task, ["a", "b"])
    assert r["correct"] is True
    assert judge_task(task, ["b", "a"])["correct"] is False


def test_judge_task_never_raises_on_none():
    assert judge_task(_task(AnswerKind.JSON, {"a": 1}), None)["correct"] is False


@pytest.mark.parametrize("kind,truth,answer", [
    (AnswerKind.STRING, "值", "值"),
    (AnswerKind.NUMBER, "1.50", "1.5"),
    (AnswerKind.ORDERED_LIST, ["x"], ["x"]),
    (AnswerKind.JSON, {"k": 1}, {"k": 1}),
])
def test_judge_is_deterministic(kind, truth, answer):
    """同输入同输出 —— 判据 4 的前置：judge 里没有任何随机或时间依赖。"""
    task = _task(kind, truth)
    first = judge_task(task, answer)
    second = judge_task(task, answer)
    assert first == second


# --------------------------------------------------------------------------------------
# A 场景四算子回归（TASK-009 追加）：B 场景 judge 落地后，这四算子必须一字未动
#
# 这几条钉的是「分流点加在四算子之前」这件事：``judge()`` 多了 ``task``/``trajectory``
# 两个带默认值的关键字参数、多了 ``JUDGE`` 分支，**A 场景走的是哪条路、返回什么结构**
# 都不能因此改变。原有的 74 条覆盖四个算子的语义，这里覆盖的是「分流没串味」。
# --------------------------------------------------------------------------------------


A_SCENE_RESULT_KEYS = {"correct", "reason", "normalized_answer", "normalized_truth"}

#: ``JUDGE`` 不在这些里面 —— 它必须**只**在 ``answer_spec.comparison is JUDGE`` 时被分流。
NON_JUDGE_COMPARISONS = (
    Comparison.EXACT,
    Comparison.DECIMAL_EXACT,
    Comparison.ORDERED_LIST_EXACT,
    Comparison.JSON_CANONICAL,
)


@pytest.mark.parametrize("comparison", NON_JUDGE_COMPARISONS)
def test_a_scene_keeps_its_exact_result_shape(comparison, monkeypatch):
    """非 JUDGE 分支返回的键与从前逐字相同 —— 不多 G/F/T，也不少任何一个。"""
    monkeypatch.setattr("zh_agent_bench.llm_client.chat",
                        lambda *a, **k: pytest.fail("A 场景绝不该调 LLM"))
    spec = AnswerSpec(kind=AnswerKind.STRING, comparison=comparison, normalization=Normalization.NONE)
    assert set(judge("x", {"answer": "x"}, spec)) == A_SCENE_RESULT_KEYS


@pytest.mark.parametrize("comparison", NON_JUDGE_COMPARISONS)
def test_a_scene_ignores_a_b_scene_task_argument(comparison, monkeypatch):
    """即使传了带 rubric 的 B 场景 task，A 场景判定结果也不许变（分流只看 comparison）。"""
    monkeypatch.setattr("zh_agent_bench.llm_client.chat",
                        lambda *a, **k: pytest.fail("A 场景绝不该调 LLM"))
    spec = AnswerSpec(kind=AnswerKind.STRING, comparison=comparison, normalization=Normalization.NONE)
    without = judge("x", {"answer": "x"}, spec)
    with_task = judge("x", {"answer": "x"}, spec, task=_task(AnswerKind.STRING, "x"))
    assert with_task == without


@pytest.mark.parametrize("kind,truth,answer", [
    (AnswerKind.STRING, "客户服务部", "客户服务部"),
    (AnswerKind.NUMBER, "1.50", "1.5"),
    (AnswerKind.ORDERED_LIST, ["a", "b"], ["a", "b"]),
    (AnswerKind.JSON, {"a": 1}, {"a": 1}),
])
def test_a_scene_is_untouched_by_the_trajectory_argument(kind, truth, answer, monkeypatch):
    """``trajectory`` 是 B 场景才用的参数；A 场景传什么都不改变结果。"""
    monkeypatch.setattr("zh_agent_bench.llm_client.chat",
                        lambda *a, **k: pytest.fail("A 场景绝不该调 LLM"))
    task = _task(kind, truth)
    spec = task.answer_spec
    noisy = [{"tool": "format.csv_to_json", "error": {"code": "x", "message": "y"}}]
    assert judge(answer, task.ground_truth, spec, task=task, trajectory=noisy) == \
        judge(answer, task.ground_truth, spec)


def test_the_judge_branch_is_reached_only_by_the_judge_comparison():
    """``Comparison.JUDGE`` 是 B 场景唯一入口 —— 分流条件就这一个。"""
    from zh_agent_bench.tasks.schema import B_SCENE_COMPARISON

    assert B_SCENE_COMPARISON is Comparison.JUDGE
    assert Comparison.JUDGE not in NON_JUDGE_COMPARISONS
