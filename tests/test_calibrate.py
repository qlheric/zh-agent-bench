"""calibrate 测试 —— 全部 mock，**绝不联网**。

这里验的是**算术与口径**，不是 judge 的判分能力（那要真 key，见判据 4）。
fake judge 传进来的分数是人为摆好的，所以每一条断言都能手算复核。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import pytest

from zh_agent_bench import calibrate
from zh_agent_bench.calibrate import (
    ANSWER_CORRUPTED,
    ANSWER_EXAMPLE,
    BAND_HIGH,
    BAND_LOW,
    build_answers,
    classify_degradation,
    confusion,
    degradation_counts,
    discrimination,
    load_tasks,
    max_per_type,
    per_type_agreement,
    render_report,
    run_calibration,
    write_outputs,
)
from zh_agent_bench.tasks.qa import TASK_TYPES

REPO_ROOT = Path(__file__).resolve().parents[1]
QA_JSONL = REPO_ROOT / "benchmarks" / "qa" / "qa.jsonl"


def _fake_verdict(g: float, *, correct: bool | None = None, reason: str = "逐条 rubric 判定完成",
                  scores: dict | None = None, ids: Sequence[str] = ()) -> dict:
    """造一条 judge 返回。``correct=None`` 时按 G 是否达门槛推，贴近真 judge 口径。

    ``ids`` 传该任务的真 criterion_id —— 键不匹配的记录会被算成降级，
    而那不是这里要测的东西（降级在 :func:`classify_degradation` 那节单独测）。
    """
    return {
        "correct": (g >= 0.8) if correct is None else correct,
        "G": g,
        "F": 1.0,
        "T": None,
        "scores": scores if scores is not None else {cid: 1.0 for cid in ids},
        "reason": reason,
        "model": "fake-model",
    }


def _judge_by_answer_kind(spec: dict[str, float]):
    """fake judge：按 answer_kind 给固定分。

    用可变 dict 而不是闭包计数器，是为了让每条测试的「摆什么分数」一眼可见，
    不用数调用顺序去反推跑到了第几次 judge。
    """
    def _fn(task, answer):
        ids = [p["criterion_id"] for p in task.rubric]
        return _fake_verdict(spec[_answer_kind(task, answer)], ids=ids)
    return _fn


def _answer_kind(task, answer) -> str:
    """fake judge 拿不到 answer_kind，靠答案文本反查（变体必然不同于 example）。"""
    example = task.ground_truth["example_answer"]
    return ANSWER_EXAMPLE if answer == example else ANSWER_CORRUPTED


# --------------------------------------------------------------------------------------
# 答案构造与预期标签
# --------------------------------------------------------------------------------------


def test_each_task_gets_exactly_two_answers_with_opposite_expectations():
    for task in load_tasks(2, 42):
        items = build_answers(task)
        assert [i["answer_kind"] for i in items] == [ANSWER_EXAMPLE, ANSWER_CORRUPTED]
        assert [i["expected_correct"] for i in items] == [True, False]
        assert [i["expected_band"] for i in items] == [BAND_HIGH, BAND_LOW]


def test_expected_bands_define_the_two_arms_of_the_confusion_table():
    """分带与「对错」标签必须一致 —— 混淆表和 G 分带是同一份预期的两种读法。

    若两者脱节，报告会同时出现「example 判对但落在 low 带」这种自相矛盾的行。
    """
    assert calibrate.EXPECTED_BAND[ANSWER_EXAMPLE] == BAND_HIGH
    assert calibrate.EXPECTED_BAND[ANSWER_CORRUPTED] == BAND_LOW
    assert calibrate.EXPECTED_CORRECT[ANSWER_EXAMPLE] is True
    assert calibrate.EXPECTED_CORRECT[ANSWER_CORRUPTED] is False


def test_corrupted_answer_actually_differs_from_the_example():
    """错误变体必须**真的**被破坏过，否则「预期错」是凭空来的。"""
    for task in load_tasks(4, 42):
        example, corrupted = build_answers(task)
        assert corrupted["answer"] != example["answer"], task.id
        assert corrupted["answer"].strip(), task.id


def test_corruption_rules_are_deterministic():
    """同一任务两次构造得到逐字节相同的变体 —— 校准记录表必须可复现。"""
    for task in load_tasks(4, 42):
        assert build_answers(task) == build_answers(task), task.id


def test_number_corruption_keeps_the_answer_readable_and_only_touches_one_number():
    """事实型破坏规则：改一个数字、其余原样。

    只改一个数字的意义在于**丢分点可归因** —— 分数掉了就是这一个点掉的，
    不用在「答案整体重写」里猜是哪条 rubric 没判对。
    """
    task = next(t for t in load_tasks(4, 42) if t.initial_data["任务类型"] == "单跳")
    example, corrupted = build_answers(task)
    assert corrupted["corruption"] == "corrupt_number"
    example_digits = [c for c in example["answer"] if c.isdigit()]
    corrupted_digits = [c for c in corrupted["answer"] if c.isdigit()]
    assert len(example_digits) == len(corrupted_digits)
    assert sum(1 for a, b in zip(example_digits, corrupted_digits) if a != b) == 1


def test_hard_answer_corruption_drops_the_critical_point_of_declarative_tasks():
    """声明型破坏规则：把「答不了」整体换成硬答（TASK-011 修，TASK-012 扩池）。

    TASK-010 的教训：旧规则 ``keep_first_sentence`` 删的是**末句建议**，
    三个 critical 点原句全留着 ⇒ judge 给 G=0.85 判「对」是**正确行为**，
    混淆表却把这 8 条 FP 记成 judge 的冤案。改硬答后不依赖删句准不准：
    声明型 critical 全是「明说答不了 / 要求澄清」这一族，硬答必然整条打空。
    """
    for task_type in ("证据不足", "不可回答"):
        task = next(t for t in load_tasks(4, 42) if t.initial_data["任务类型"] == task_type)
        _example, corrupted = build_answers(task)
        assert corrupted["corruption"] == "hard_answer"
        assert corrupted["answer"] in calibrate.HARD_ANSWER_POOL
        # 「答不了」的锚词必须在**每一句**变体里彻底消失 —— 残留即 critical 可能仍被判到。
        for anchor in ("无法", "不能", "答不了", "不足", "请"):
            assert not any(anchor in text for text in calibrate.HARD_ANSWER_POOL), anchor


def test_hard_answer_pool_has_multiple_distinct_wordings():
    """池里必须 ≥2 个**措辞不同**的硬答句（TASK-012）。

    池只放一句的话，「judge 认的是硬答行为还是这一句话」根本测不出来 ——
    TASK-011 的 100% 一致率可能只是 judge 认得那句话。
    同义改写还不够，措辞得真的拉开（一句话递进推诿、一句话轻描淡写），
    否则两句其实是同源句，换了也测不出东西。
    """
    assert len(calibrate.HARD_ANSWER_POOL) >= 2
    assert len(set(calibrate.HARD_ANSWER_POOL)) == len(calibrate.HARD_ANSWER_POOL), "池里有重复文本"
    # 任何一句都不许塞具体数字/事实 —— 变体要证明的是「不声明证据不足」被扣分，
    # 混进「答错内容」就成了两件事挤一格，诊断不出根因。
    for text in calibrate.HARD_ANSWER_POOL:
        assert not re.search(r"\d", text), text


def test_declarative_tasks_spread_across_the_whole_pool():
    """池里**每一句**都要真被用到 —— 否则加了池等于没加（TASK-012 判据）。

    轮换是 ``crc32(task_id) % len(pool)``：同一个 seed 下任务集合固定，
    若某些文本一条任务都没轮到，多句就退化成单句，白扩。
    """
    tasks = [t for t in load_tasks(4, 42) if t.initial_data["任务类型"] in ("证据不足", "不可回答")]
    used = {build_answers(t)[1]["answer"] for t in tasks}
    assert used == set(calibrate.HARD_ANSWER_POOL), (
        f"只用到 {len(used)}/{len(calibrate.HARD_ANSWER_POOL)} 句；"
        f"没轮到的：{set(calibrate.HARD_ANSWER_POOL) - used}"
    )


def test_hard_answer_rotation_is_deterministic():
    """同 seed 两次跑必须挑到同一句 —— 校准产物要逐字节可复现（findings §二）。"""
    first = [build_answers(t)[1]["answer"] for t in load_tasks(4, 42)]
    second = [build_answers(t)[1]["answer"] for t in load_tasks(4, 42)]
    assert first == second


def test_factual_tasks_still_use_number_corruption():
    """事实型不能被硬答池波及 —— 它考的是「数字对不对」，换硬答等于换了一道题。"""
    for task_type in ("单跳", "多跳"):
        task = next(t for t in load_tasks(4, 42) if t.initial_data["任务类型"] == task_type)
        _example, corrupted = build_answers(task)
        assert corrupted["corruption"] == "corrupt_number"
        assert corrupted["answer"] not in calibrate.HARD_ANSWER_POOL


def test_corruption_that_keeps_critical_point_is_rejected():
    """破坏规则没打掉 critical 点时必须**报错**，而不是悄悄流进混淆表。

    规则和 judge 一样会错。规则悄悄没打中的话，报告里所有结论都会跟着错，
    而且没有任何信号提示 —— 这正是 TASK-010 那 8 条 FP 冤案的成因。
    """
    task = next(t for t in load_tasks(4, 42) if t.initial_data["任务类型"] == "证据不足")
    example = task.ground_truth["example_answer"]
    with pytest.raises(ValueError, match="没打掉任何 critical 点"):
        calibrate._assert_corruption_hits_critical(task, example, example)


def test_corruption_rules_cover_every_task_type():
    """四类任务都必须有破坏规则 —— 漏一类就有一条任务悄悄拿不到错误变体。"""
    assert set(calibrate.CORRUPT_RULES) == set(TASK_TYPES)
    for task in load_tasks(4, 42):
        assert build_answers(task)[1]["corruption"] is not None, task.id


# --------------------------------------------------------------------------------------
# 混淆表算术
# --------------------------------------------------------------------------------------


def _record(expected: bool, got: bool, task_type: str = "单跳", g: float = 1.0) -> dict:
    """造一条统计用的假记录。``answer_kind`` 由 ``expected`` 推出 ——
    两臂的 G 分带统计（:func:`discrimination`）靠这个字段分组，手填必错。"""
    return {
        "task_id": "qa-x", "task_type": task_type,
        "answer_kind": ANSWER_EXAMPLE if expected else ANSWER_CORRUPTED,
        "judge_G": g, "judge_correct": got, "expected_correct": expected,
        "expected_band": BAND_HIGH if expected else BAND_LOW,
        "reason": "", "degradation": None,
    }


def test_confusion_table_arithmetic():
    """TP/TN/FP/FN 四格逐一手算复核 —— 混淆表算错了后面全错。"""
    records = [
        _record(True, True),    # 好答案放过 -> TP
        _record(True, False),   # 好答案误杀 -> FN
        _record(False, True),   # 坏答案放过 -> FP（最危险）
        _record(False, False),  # 坏答案抓住 -> TN
    ]
    table = confusion(records)
    assert (table["tp"], table["fn"], table["fp"], table["tn"]) == (1, 1, 1, 1)
    assert table["total"] == 4
    assert table["agreement"] == 0.5


def test_agreement_is_one_when_all_correct():
    """边界：全对 → 一致率 1.0（不是 0.5 —— 分母是总样本数，不是「改过的样本数」）。"""
    records = [_record(True, True) for _ in range(5)] + [_record(False, False) for _ in range(3)]
    assert confusion(records)["agreement"] == 1.0


def test_agreement_is_zero_when_everything_is_inverted():
    """边界：全错 → 一致率 0.0，不能因为「坏答案都判错」这种子集看起来不错而虚高。"""
    records = [_record(True, False) for _ in range(4)] + [_record(False, True) for _ in range(2)]
    table = confusion(records)
    assert table["agreement"] == 0.0
    assert (table["fp"], table["fn"]) == (2, 4)


def test_confusion_of_empty_input_does_not_divide_by_zero():
    assert confusion([])["agreement"] == 0.0


def test_false_positive_is_the_cell_that_matters_most():
    """坏答案被判对 = FP。定向钉死：把一条坏答案翻成判对，FP 必须 +1、TN 必须 -1。

    不写这条，「混淆表 2×2 算术」就只能靠肉眼扫四格 —— 而 FP 恰恰是四格里
    最容易被写反的一格（正类是「判对」还是「预期对」很容易搞混）。
    """
    before = confusion([_record(False, False)])
    after = confusion([_record(False, True)])
    assert (before["tn"], before["fp"]) == (1, 0)
    assert (after["tn"], after["fp"]) == (0, 1)


# --------------------------------------------------------------------------------------
# 按类型切片
# --------------------------------------------------------------------------------------


def test_per_type_agreement_splits_the_four_types():
    """整体一致率高不能掩盖某一类全错 —— 切片必须能单独把那一类拎出来。"""
    records = (
        [_record(True, True, "单跳")] * 4 + [_record(False, False, "单跳")]
        + [_record(True, True, "多跳")] * 2 + [_record(True, False, "多跳")] * 2
        + [_record(True, True, "证据不足")] + [_record(False, False, "证据不足")]
        + [_record(False, True, "不可回答")] * 2
    )
    sliced = per_type_agreement(records)
    assert sliced["单跳"]["agreement"] == 1.0
    assert sliced["多跳"]["agreement"] == 0.5
    assert sliced["证据不足"]["agreement"] == 1.0
    # 不可回答这一类整类都是坏答案被放过 —— 整体算术几乎看不出来
    assert sliced["不可回答"]["fp"] == 2
    # 总计 13 条，对角线 9 条（单跳 5 + 多跳 2 + 证据不足 2），9/13
    assert confusion(records)["total"] == 13
    assert confusion(records)["agreement"] == pytest.approx(9 / 13)


def test_per_type_agreement_omits_types_with_no_records():
    """没有记录的类别不进表（而不是补一行 0/0 拉低观感）。"""
    assert list(per_type_agreement([_record(True, True, "单跳")])) == ["单跳"]


# --------------------------------------------------------------------------------------
# 降级计数
# --------------------------------------------------------------------------------------


def test_degradation_counts_are_zero_for_a_healthy_batch():
    records = [_record(True, True) for _ in range(6)]
    counts = degradation_counts(records)
    assert counts == {"llm_failure": 0, "invalid_score": 0, "missing_criterion": 0}


def test_degradation_counts_each_failure_mode_separately():
    """三类降级各计各的 —— 混成一个总数就看不出「是网络挂了」还是「模型乱给分」。"""
    records = [
        _record(True, True),
        _record(True, True),
        {"degradation": "llm_failure"}, {"degradation": "llm_failure"},
        {"degradation": "invalid_score"},
        {"degradation": "missing_criterion"},
    ]
    counts = degradation_counts(records)
    assert counts == {"llm_failure": 2, "invalid_score": 1, "missing_criterion": 1}
    assert sum(counts.values()) == 4


def test_classify_degradation_detects_llm_failure():
    assert classify_degradation("judge 调用失败：HTTP 500 @ api.deepseek.com", {}, ["c1"]) == "llm_failure"


def test_classify_degradation_detects_illegal_score():
    reason = "c1: 分值 0.7 不在 0/0.5/1 内，按 0 记"
    assert classify_degradation(reason, {"c1": 0.0}, ["c1"]) == "invalid_score"


def test_classify_degradation_detects_missing_criterion():
    assert classify_degradation(
        "judge 未给出分值（收到 None），按 0 记", {"c1": 0.0, "c2": 0.0}, ["c1", "c2"]
    ) == "missing_criterion"


def test_classify_degradation_detects_scores_key_mismatch():
    """scores 的键与 rubric 不一致也算降级 —— judge 漏判不得被当成「这点没提到」静默放过。"""
    assert classify_degradation("逐条 rubric 判定完成", {"c1": 1.0}, ["c1", "c2"]) == "missing_criterion"
    assert classify_degradation("...", None, ["c1"]) == "missing_criterion"
    assert classify_degradation("...", {}, ["c1"]) == "missing_criterion"


def test_healthy_record_is_not_a_degradation():
    assert classify_degradation("逐条 rubric 判定完成", {"c1": 1.0, "c2": 0.5}, ["c1", "c2"]) is None


def test_degradation_survives_into_the_confusion_table():
    """降级样本仍计入混淆表（不能悄悄剔除），但降级数必须同时被报出来。

    剔除掉会让一致率虚高，而计入且报数才是「这批数字不可直接引用」的诚实口径。
    """
    records = [
        _record(True, True),
        _record(False, False),
        _record(True, True),
        _record(False, False),
        _record(False, False, g=0.0),   # judge 挂了 → 这条被降级
    ]
    records[-1]["degradation"] = "llm_failure"
    table = confusion(records)
    assert table["total"] == 5
    assert (table["tp"], table["tn"]) == (2, 3)
    assert degradation_counts(records)["llm_failure"] == 1


# --------------------------------------------------------------------------------------
# 判别力
# --------------------------------------------------------------------------------------


def test_discrimination_reports_the_gap_between_the_two_arms():
    records = [
        _record(True, True, g=1.0), _record(True, True, g=0.8), _record(True, True, g=0.9),
        _record(False, False, g=0.4), _record(False, False, g=0.5), _record(False, False, g=0.4),
    ]
    disc = discrimination(records)
    assert disc[ANSWER_EXAMPLE] == pytest.approx((1.0 + 0.8 + 0.9) / 3)
    assert disc[ANSWER_CORRUPTED] == pytest.approx((0.4 + 0.5 + 0.4) / 3)
    assert disc["gap"] == pytest.approx(disc[ANSWER_EXAMPLE] - disc[ANSWER_CORRUPTED])


def test_discrimination_is_none_when_one_arm_has_no_scores():
    assert discrimination([_record(True, True, g=None)])["gap"] is None


# --------------------------------------------------------------------------------------
# run_calibration（mock judge，不联网）
# --------------------------------------------------------------------------------------


def test_run_calibration_produces_two_records_per_task():
    tasks = load_tasks(2, 42)
    records = run_calibration(tasks, judge_fn=_judge_by_answer_kind(
        {ANSWER_EXAMPLE: 1.0, ANSWER_CORRUPTED: 0.3}
    ))
    assert len(records) == 2 * len(tasks)
    assert [r["answer_kind"] for r in records[:2]] == [ANSWER_EXAMPLE, ANSWER_CORRUPTED]


def test_run_calibration_with_a_perfect_judge_gives_full_agreement():
    """fake judge 完全正确 ⇒ 一致率 100%、零降级、混淆表全在对角线上。"""
    tasks = load_tasks(4, 42)
    records = run_calibration(tasks, judge_fn=_judge_by_answer_kind(
        {ANSWER_EXAMPLE: 1.0, ANSWER_CORRUPTED: 0.0}
    ))
    table = confusion(records)
    assert table["agreement"] == 1.0
    assert (table["tp"], table["fn"], table["fp"], table["tn"]) == (len(tasks), 0, 0, len(tasks))
    assert sum(degradation_counts(records).values()) == 0


def test_run_calibration_with_a_blind_judge_gives_zero_agreement_and_reports_discrimination():
    """**判别力这张表的反面证据**：judge 两臂都给满分 ⇒ 一致率跌到 50%、降级仍为 0。

    这两条一起证明「零降级」和「判得对」是两回事 —— 前者是 judge 活着，
    后者才是 judge 会判。降级 0 却一致率 50%，唯一解释就是 FP 全开。

    50% 而不是 0%：分母含全部样本，好答案那半仍判对。**若看到 0%，那才该先怀疑
    混淆表算错了**（连 judge 都会做对的样本不可能全错）。
    """
    tasks = load_tasks(2, 42)
    records = run_calibration(tasks, judge_fn=_judge_by_answer_kind(
        {ANSWER_EXAMPLE: 1.0, ANSWER_CORRUPTED: 1.0}
    ))
    table = confusion(records)
    assert table["agreement"] == 0.5
    assert (table["fp"], table["tp"]) == (len(tasks), len(tasks))
    assert table["fn"] == 0 and table["tn"] == 0
    assert sum(degradation_counts(records).values()) == 0, "这不是 judge 坏了，是 judge 不判"
    assert discrimination(records)["gap"] == 0.0


def test_run_calibration_flags_a_failing_judge_as_degradation():
    """judge 全程调用失败 ⇒ 每条都记 llm_failure，且分数全是 0。"""
    def broken(task, answer):
        return {"correct": False, "G": 0.0, "F": None, "T": None, "scores": {},
                "reason": "judge 调用失败：未提供 API key", "model": "fake"}

    records = run_calibration(load_tasks(2, 42), judge_fn=broken)
    assert degradation_counts(records)["llm_failure"] == len(records)
    assert all(r["judge_G"] == 0.0 for r in records)
    assert all(r["judge_correct"] is False for r in records)


def test_run_calibration_records_are_json_serializable_and_ordered():
    records = run_calibration(load_tasks(2, 42), judge_fn=_judge_by_answer_kind(
        {ANSWER_EXAMPLE: 1.0, ANSWER_CORRUPTED: 0.2}
    ))
    for record in records:
        json.dumps(record, ensure_ascii=False)  # 抛了就说明混进了不可序列化的值
    ids = [r["task_id"] for r in records]
    assert ids == sorted(ids), "记录必须按 task_id 升序，重跑才对得上"


# --------------------------------------------------------------------------------------
# --per-type 边界（不许静默循环复用）
# --------------------------------------------------------------------------------------


def test_per_type_above_pool_capacity_is_rejected():
    """场景池每类 4 条，取 5 条就得循环复用 —— 一致率会被同一条样本计两次，必须报错。"""
    with pytest.raises(ValueError, match="超过场景池每类容量"):
        load_tasks(max_per_type() + 1, 42)


def test_per_type_zero_is_rejected():
    with pytest.raises(ValueError, match=">= 1"):
        load_tasks(0, 42)


def test_per_type_at_pool_capacity_is_allowed_and_never_repeats_a_task():
    tasks = load_tasks(max_per_type(), 42)
    assert len(tasks) == max_per_type() * len(TASK_TYPES)
    assert len({t.id for t in tasks}) == len(tasks), "同一条任务被选了两次"


def test_load_tasks_gives_each_type_exactly_per_type_records():
    for per_type in (1, 2, 4):
        tasks = load_tasks(per_type, 42)
        assert len(tasks) == per_type * len(TASK_TYPES)
        for task_type in TASK_TYPES:
            rows = [t for t in tasks if t.initial_data["任务类型"] == task_type]
            assert len(rows) == per_type, task_type


def test_load_tasks_from_committed_dataset_matches_the_generated_one():
    """读现成 qa.jsonl 与按 (per_type=2, seed=42) 生成必须逐条相同。

    不等就说明「读路径」与「生成路径」已经漂了 —— 那么校准跑的是哪一批任务
    就要看当时传了哪个参数，报告的可复现性也就没了。
    """
    generated = load_tasks(2, 42)
    from_file = load_tasks(2, 42, QA_JSONL)
    assert [t.id for t in from_file] == [t.id for t in generated]
    assert [t.ground_truth for t in from_file] == [t.ground_truth for t in generated]
    assert [t.rubric for t in from_file] == [t.rubric for t in generated]


# --------------------------------------------------------------------------------------
# 产物
# --------------------------------------------------------------------------------------


def test_report_contains_confusion_table_agreement_and_degradation_counts():
    records = run_calibration(load_tasks(2, 42), judge_fn=_judge_by_answer_kind(
        {ANSWER_EXAMPLE: 1.0, ANSWER_CORRUPTED: 0.3}
    ))
    text = render_report(records, per_type=2, seed=42)
    assert "| **预期对** | TP" in text
    assert "| **预期错** | FP" in text
    assert "一致率：" in text
    assert "llm_failure" in text and "invalid_score" in text and "missing_criterion" in text
    for task_type in TASK_TYPES:
        assert task_type in text, f"按类型切片里少了 {task_type}"
    assert "逐条 rubric 判定完成" in text or "不一致样本" in text


def test_report_is_byte_identical_across_runs():
    """报告无时间戳：同输入两次逐字节一致（findings §二的确定性承诺）。"""
    def once() -> str:
        records = run_calibration(load_tasks(2, 42), judge_fn=_judge_by_answer_kind(
            {ANSWER_EXAMPLE: 1.0, ANSWER_CORRUPTED: 0.3}
        ))
        return render_report(records, per_type=2, seed=42)

    assert once().encode() == once().encode()


def test_report_flags_a_batch_with_degradation():
    """有降级时报告必须显眼地提示「上面的分数不可直接引用」。"""
    records = run_calibration(load_tasks(1, 42), judge_fn=lambda task, answer: {
        "correct": False, "G": 0.0, "F": None, "T": None, "scores": {},
        "reason": "judge 调用失败：网络错误", "model": "fake",
    })
    text = render_report(records, per_type=1, seed=42)
    assert "不可直接引用" in text


def test_report_separates_f_kills_from_g_failures():
    """G 达门槛却判错的样本要单独点名 —— 那批问题的根因在 F 的 claim 口径，不在 rubric。

    混在一张表里只看得到「judge 判错了」，看不出该修 judge 的哪一段；
    而这两种根因分属两个模块（``judge.py`` 的 F vs rubric 判分），修错地方等于白干。
    """
    def _fake(task, answer):
        ids = [p["criterion_id"] for p in task.rubric]
        if _answer_kind(task, answer) == ANSWER_EXAMPLE:
            return {
                "correct": False, "G": 1.0, "F": 0.5, "T": None,
                "scores": {cid: 1.0 for cid in ids},
                "reason": "逐条 rubric 判定完成", "model": "fake",
                "n_contradicted": 0, "n_unsupported": 1,
            }
        return _fake_verdict(0.3, ids=ids)

    text = render_report(run_calibration(load_tasks(1, 42), judge_fn=_fake),
                         per_type=1, seed=42)
    assert "G 已达门槛却被判错" in text
    assert "| n_unsup |" in text


def test_write_outputs_emits_report_and_records(tmp_path):
    records = run_calibration(load_tasks(2, 42), judge_fn=_judge_by_answer_kind(
        {ANSWER_EXAMPLE: 1.0, ANSWER_CORRUPTED: 0.3}
    ))
    report = write_outputs(records, tmp_path, per_type=2, seed=42)
    assert report == tmp_path / "report.md"
    lines = (tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
    assert json.loads(lines[0])["task_id"] == records[0]["task_id"]


# --------------------------------------------------------------------------------------
# CLI（用 fake judge，不联网）
# --------------------------------------------------------------------------------------


def test_cli_rejects_per_type_above_pool_capacity(capsys):
    with pytest.raises(SystemExit):
        calibrate.main(["--per-type", str(max_per_type() + 1), "--out", "unused"])
    assert "超过场景池每类容量" in capsys.readouterr().err


def test_calibrate_has_no_third_party_import():
    """沿用 ``test_harness.py::test_runtime_has_no_third_party_import`` 的铁律。

    校准模块进了运行时热路径，「clone 即跑、零第三方依赖」这条承诺对它一视同仁。
    """
    import ast

    allowed = {"json", "re", "sys", "argparse", "pathlib", "dataclasses", "enum",
               "decimal", "typing", "collections", "abc", "zlib", "__future__"}
    source = REPO_ROOT / "src" / "zh_agent_bench" / "calibrate.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed, f"calibrate.py 引入第三方: {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            assert node.module.split(".")[0] in allowed, f"calibrate.py 引入第三方: {node.module}"


def test_cli_writes_a_report_and_prints_the_table(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(calibrate, "run_calibration", lambda tasks, **kw: [
        {"task_id": t.id, "task_type": t.initial_data["任务类型"],
         "answer_kind": kind, "corruption": None,
         "judge_G": g, "judge_correct": kind == ANSWER_EXAMPLE,
         "expected_correct": kind == ANSWER_EXAMPLE,
         "expected_band": BAND_HIGH if kind == ANSWER_EXAMPLE else BAND_LOW,
         "judge_model": "fake", "degradation": None, "reason": "fake"}
        for t in tasks for kind, g in ((ANSWER_EXAMPLE, 1.0), (ANSWER_CORRUPTED, 0.3))
    ])
    assert calibrate.main(["--per-type", "2", "--seed", "7", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "agreement = 100.00%" in out
    assert "degradation:" in out
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "records.jsonl").exists()
