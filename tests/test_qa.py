"""B 场景任务与 rubric 测试 —— TASK-008 判据 1。

五件事逐条对应判据：
① retrieval 任务 rubric 非空，每点含 criterion_id/requirement/weight/evidence_ids/critical
② 每点的 0/0.5/1 ``partial_rule`` 可判
③ ``Comparison.JUDGE`` 合法、A 场景任务 ``rubric is None`` 且序列化结构不变（**回归**）
④ 同 seed 两次产物逐字节一致
⑤ ``evidence_ids`` 指向的文档都存在

另加两条这套 rubric 专属的：权重归一（跨任务可比）、critical 漏点封顶。
判据②里「可判」靠人写，本文件能机械验的是**规则形状**：每点都必须落在
「三值（写了 0.5 触发情形）」或「二值（显式宣告半分不存在）」两种形态之一，
含糊规则一律拒收 —— 那正是校准时分歧的来源。
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pytest

from zh_agent_bench.tasks.build import build_dataset, write_dataset
from zh_agent_bench.tasks.qa import (
    ALLOWED_SCORES,
    B_SPLITS,
    CRITICAL_CAP,
    FORMAL_SIZES,
    MAX_POINTS,
    MIN_POINTS,
    QA_TOOL_NAME,
    RUBRIC_FIELDS,
    SCENARIOS,
    TASK_TYPES,
    TEMPLATES,
    _expand,
    aggregate_rubric,
    build_formal_dataset,
    build_qa_dataset,
    partial_rule_kind,
    main,
    validate_rubric,
    write_formal_dataset,
    write_qa_dataset,
)
from zh_agent_bench.tasks.schema import (
    A_SCENE_CATEGORIES,
    A_SCENE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    AnswerKind,
    AnswerSpec,
    Category,
    Comparison,
    Normalization,
    Split,
    Task,
)

SEED = 42
PER_TYPE = 2


@pytest.fixture(scope="module")
def qa_tasks():
    return build_qa_dataset(per_type=PER_TYPE, seed=SEED)


@pytest.fixture(scope="module")
def all_scenarios() -> list:
    """四个类型的**全部**场景（不只 --per-type 抽中的那几条）。

    rubric 是逐场景手写的，只验抽中的两条等于让另外 14 条手写内容没人看 ——
    那些恰恰是最容易把 evidence_ids 写错成上一篇的地方。
    """
    return [s for t in TASK_TYPES for s in SCENARIOS[t]]


# ======================================================================
# ① 任务与 rubric 的基本形状
# ======================================================================


def test_dataset_has_four_types_each_with_per_type_tasks(qa_tasks):
    assert len(qa_tasks) == PER_TYPE * len(TASK_TYPES) == 8
    for task_type in TASK_TYPES:
        rows = [t for t in qa_tasks if t.initial_data["任务类型"] == task_type]
        assert len(rows) == PER_TYPE, f"{task_type} 应 {PER_TYPE} 条，实际 {len(rows)}"


def test_every_task_is_retrieval_with_the_search_tool(qa_tasks):
    for task in qa_tasks:
        assert task.category is Category.RETRIEVAL
        assert task.category.is_b_scene
        assert task.tool_names == (QA_TOOL_NAME,)
        assert task.answer_spec.comparison is Comparison.JUDGE
        assert task.schema_version == SCHEMA_VERSION == "1.1"


def test_every_retrieval_task_has_a_non_empty_rubric(qa_tasks):
    for task in qa_tasks:
        assert task.rubric, f"{task.id} 的 rubric 为空 —— 无 rubric 就无从判分"
        assert MIN_POINTS <= len(task.rubric) <= MAX_POINTS


def test_every_rubric_point_carries_all_six_fields(all_scenarios):
    """逐字段钉全，不只是「有这个键」——键在、值是空串或 None 一样不可判。

    场景里存六个字段，``criterion_id`` 在 ``build_task`` 里按位置补上（另有一条断言
    钉补出来的 id 是否唯一、带任务前缀）。
    """
    for scenario in all_scenarios:
        for point in scenario.points:
            for field in (f for f in RUBRIC_FIELDS if f != "criterion_id"):
                assert field in point, f"{scenario.task_type} 评分点缺 {field}"
            assert isinstance(point["requirement"], str) and point["requirement"].strip()
            assert isinstance(point["weight"], float) and 0 < point["weight"] <= 1.0
            assert isinstance(point["evidence_ids"], list)
            assert isinstance(point["accepted_alternatives"], list)
            assert isinstance(point["critical"], bool)


def test_criterion_ids_are_unique_and_namespaced_to_the_task(qa_tasks):
    for task in qa_tasks:
        ids = [p["criterion_id"] for p in task.rubric]
        assert len(ids) == len(set(ids)), f"{task.id} 的 criterion_id 有重复"
        for cid in ids:
            assert cid.startswith(f"{task.id}-c"), f"{cid} 应当以任务 id 开头"


def test_rubric_has_no_unknown_fields(all_scenarios):
    """多余字段是漂移的开始：字段一多，judge 侧就会各自解释它的含义。

    场景里存的是**未编号**的要点（``criterion_id`` 由 ``_materialize`` 按位置补），
    所以比的是去掉 criterion_id 后的那六个字段。
    """
    expected = sorted(f for f in RUBRIC_FIELDS if f != "criterion_id")
    for scenario in all_scenarios:
        for point in scenario.points:
            assert sorted(point) == expected, f"评分点字段不符: {sorted(point)}"


def test_some_points_are_marked_critical(all_scenarios):
    """必须真的有 critical 点，否则「关键错误不可被抵消」这条设计就没落地。

    「证据不足」和「不可回答」两类尤其要有：漏掉「指出缺口」而把别的答全，
    正是这类任务要防的失败模式。
    """
    for task_type in ("证据不足", "不可回答"):
        for scenario in SCENARIOS[task_type]:
            assert any(p["critical"] for p in scenario.points), f"{task_type} 无 critical 点"


def test_weight_sums_to_one(all_scenarios):
    """权重归一 ⇒ G 直接读作百分比，跨任务、跨类型的分数才可比。"""
    for scenario in all_scenarios:
        total = sum(p["weight"] for p in scenario.points)
        assert abs(total - 1.0) < 1e-9, f"{scenario.task_type} 权重和 {total!r} != 1.0"


def test_validate_rubric_rejects_unnormalized_weights():
    # 五点各 0.3（和为 1.5）才单独撞上「权重和 != 1」这一条；点数不够会被前一道
    # MIN_POINTS 闸先拦下来，那样就测不到本条规则了。
    points = [
        {"criterion_id": f"x-c{i}", "requirement": "r", "weight": 0.3,
         "evidence_ids": [], "partial_rule": "满足→1，部分→0.5",
         "accepted_alternatives": [], "critical": False}
        for i in range(1, 6)
    ]
    with pytest.raises(ValueError, match="权重之和"):
        validate_rubric(points, set())


def test_validate_rubric_accepts_weights_that_do_sum_to_one():
    """反向：归一化的一组权重必须通过 —— 上一条不能是「怎么写都报错」。"""
    points = [
        {"criterion_id": f"x-c{i}", "requirement": "r", "weight": 0.2,
         "evidence_ids": [], "partial_rule": "满足→1，部分→0.5",
         "accepted_alternatives": [], "critical": False}
        for i in range(1, 6)
    ]
    validate_rubric(points, set())  # 不抛即通过


def test_validate_rubric_rejects_an_undecidable_partial_rule():
    """只写「部分满足可给半分」而不说什么算部分满足 —— 判分员各打各的，必须拒收。"""
    points = [
        {"criterion_id": f"x-c{i}", "requirement": "r", "weight": 0.2,
         "evidence_ids": [], "partial_rule": "部分满足可给半分",
         "accepted_alternatives": [], "critical": False}
        for i in range(1, 6)
    ]
    with pytest.raises(ValueError, match="partial_rule"):
        validate_rubric(points, set())


def test_validate_rubric_rejects_evidence_ids_that_point_nowhere():
    """judge 核证据时找不到出处，这一条就废了 —— 生成期就该炸。"""
    points = [
        {"criterion_id": f"x-c{i}", "requirement": "r", "weight": 0.2,
         "evidence_ids": ["kb-不存在"], "partial_rule": "满足→1，部分→0.5",
         "accepted_alternatives": [], "critical": False}
        for i in range(1, 6)
    ]
    with pytest.raises(ValueError, match="不存在的文档"):
        validate_rubric(points, {"kb-real"})


# ======================================================================
# ② partial_rule 的分值可判性
# ======================================================================


def test_every_partial_rule_is_decidable(all_scenarios):
    """每点必须落在两种形态之一：三值（写明 0.5 触发情形）或二值（显式宣告半分不存在）。

    判据②要「可判」。含糊规则（「部分满足可给半分」而不说什么算部分满足）
    是校准分歧的头号来源，所以宁可生成期就拒收，也不要留给标注员自由发挥。
    """
    for scenario in all_scenarios:
        for point in scenario.points:
            rule = point["partial_rule"]
            assert rule.strip(), "partial_rule 不能为空"
            assert partial_rule_kind(rule) != "undecidable", f"partial_rule 不可判: {rule!r}"


def test_binary_marker_is_not_confused_by_the_half_credit_marker():
    """``"→ 0"`` 是 ``"→ 0.5"`` 的子串 —— 用 in 判会把每个三值点误判成二值。

    这条钉的是当年真踩到的那个坑（生成器对着满是 ``→ 0.5`` 的规则报「必须写明 0.5」）。
    """
    assert partial_rule_kind("完全满足 → 1，部分满足 → 0.5") == "ternary"
    assert partial_rule_kind("编造了具体口径 → 0") == "binary"
    assert partial_rule_kind("编造了具体口径 → 0.0") == "binary"
    assert partial_rule_kind("没说清楚就给了个大概说法") == "undecidable"


def test_ternary_points_name_the_half_credit_condition(all_scenarios):
    """三值点不只是「有 0.5」，还得把触发情形写出来（有箭头前半段）。"""
    for scenario in all_scenarios:
        for point in scenario.points:
            rule = point["partial_rule"]
            if "0.5" not in rule:
                continue
            head = rule.split("0.5")[0].strip()
            assert head, f"0.5 之前没写触发情形: {rule!r}"
            assert len(head) >= 6, f"触发情形太短，判分员读不出边界: {rule!r}"


def test_binary_points_never_award_partial_credit(all_scenarios):
    """二值点不该同时又给半分 —— 那两条规则会打架。"""
    for scenario in all_scenarios:
        for point in scenario.points:
            assert partial_rule_kind(point["partial_rule"]) == "ternary" \
                   or "0.5" not in point["partial_rule"], \
                f"二值与半分规则同时出现: {point['partial_rule']!r}"


def test_aggregate_rejects_a_score_outside_the_three_values(qa_tasks):
    task = qa_tasks[0]
    ids = [p["criterion_id"] for p in task.rubric]
    scores = {cid: 1.0 for cid in ids}
    scores[ids[0]] = 0.7
    with pytest.raises(ValueError, match="不在"):
        aggregate_rubric(task.rubric, scores)


def test_aggregate_rejects_a_score_set_that_does_not_match_the_rubric(qa_tasks):
    task = qa_tasks[0]
    ids = [p["criterion_id"] for p in task.rubric]
    with pytest.raises(ValueError, match="criterion_id"):
        aggregate_rubric(task.rubric, {ids[0]: 1.0})


def test_aggregate_accepts_exactly_the_three_allowed_scores(qa_tasks):
    for task in qa_tasks:
        ids = [p["criterion_id"] for p in task.rubric]
        for value in ALLOWED_SCORES:
            aggregate_rubric(task.rubric, {cid: value for cid in ids})


def test_perfect_scores_aggregate_to_one(qa_tasks):
    for task in qa_tasks:
        g, capped = aggregate_rubric(task.rubric,
                                     {p["criterion_id"]: 1.0 for p in task.rubric})
        assert g == 1.0 and capped is False


def test_zero_scores_aggregate_to_zero(qa_tasks):
    for task in qa_tasks:
        g, _ = aggregate_rubric(task.rubric, {p["criterion_id"]: 0.0 for p in task.rubric})
        assert g == 0.0


def test_partial_scores_land_between(qa_tasks):
    """半分点只出现在写了 0.5 的点上；只对这类点给 0.5，结果应严格落在 (0, 1) 之内。"""
    for task in qa_tasks:
        ids = [p["criterion_id"] for p in task.rubric]
        scores = {cid: (0.5 if "0.5" in p["partial_rule"] else 0.0)
                  for cid, p in zip(ids, task.rubric)}
        g, _ = aggregate_rubric(task.rubric, scores)
        assert 0.0 < g < 1.0, f"{task.id} 全给半分竟得 {g}"


def test_a_missed_critical_point_caps_the_score(qa_tasks):
    """关键点漏了封顶到 CRITICAL_CAP：漏掉「指出证据不足」而把别的答全，不该拿高分。"""
    capped_hits = 0
    for task in qa_tasks:
        critical = [p for p in task.rubric if p["critical"]]
        if not critical:
            continue
        capped_hits += 1
        scores = {p["criterion_id"]: 1.0 for p in task.rubric}
        scores[critical[0]["criterion_id"]] = 0.0
        g, was_capped = aggregate_rubric(task.rubric, scores)
        assert was_capped is True, f"{task.id} 漏关键点却没封顶"
        assert g == pytest.approx(CRITICAL_CAP)
    assert capped_hits >= 4, "一半以上的任务带 critical 点才谈得上封顶有意义"


def test_capping_never_raises_a_score(qa_tasks):
    """封顶只压不抬：已低于 cap 的分数原样通过。"""
    for task in qa_tasks:
        critical = [p for p in task.rubric if p["critical"]]
        if not critical:
            continue
        scores = {p["criterion_id"]: 0.0 for p in task.rubric}
        g, was_capped = aggregate_rubric(task.rubric, scores)
        assert g == 0.0 and was_capped is True


# ======================================================================
# ③ A 场景回归：rubric is None、序列化结构不变
# ======================================================================


def test_judge_comparison_is_legal_in_answer_spec():
    spec = AnswerSpec(kind=AnswerKind.STRING, comparison=Comparison.JUDGE,
                      normalization=Normalization.NONE)
    assert AnswerSpec.from_dict(spec.to_dict()) == spec


def test_judge_comparison_does_not_loosen_the_other_combinations():
    """放行 JUDGE 的是**这一条**，不是整套闸门：其余非法组合照旧报错。"""
    with pytest.raises(ValueError, match="非法"):
        AnswerSpec.from_dict({"kind": "string", "comparison": "decimal_exact",
                              "normalization": "strip"})


def test_a_scene_tasks_have_no_rubric_and_keep_schema_1_0():
    for task in build_dataset(per_category=1, seed=SEED):
        assert task.rubric is None
        assert task.category in A_SCENE_CATEGORIES
        assert task.schema_version == A_SCENE_SCHEMA_VERSION == "1.0"


def test_a_scene_serialized_shape_has_no_rubric_key():
    for task in build_dataset(per_category=1, seed=SEED):
        payload = task.to_dict()
        assert "rubric" not in payload, "A 场景多出 rubric 键会打破已入库 120 条的字节锁"
        assert Task.from_dict(payload).rubric is None


def test_a_scene_round_trip_is_lossless():
    for task in build_dataset(per_category=1, seed=SEED):
        assert Task.from_dict(task.to_dict()) == task


def test_retrieval_tasks_round_trip_losslessly(qa_tasks):
    for task in qa_tasks:
        restored = Task.from_dict(task.to_dict())
        assert restored == task
        assert restored.rubric is not None


def test_a_scene_dataset_is_still_byte_identical_to_its_committed_copy():
    """**A 场景 318 一个不红**的机器证据：本次改动没碰过已入库的 120 条。

    这条比任何「我确认没改」的自述都硬 —— 逐字节比对，任何 A 场景字段的漂移都躲不过。
    """
    committed = Path(__file__).resolve().parents[1] / "benchmarks" / "tasks"
    with tempfile.TemporaryDirectory() as tmp:
        regen = Path(tmp) / "regen"
        write_dataset(build_dataset(30, SEED), regen)
        names = sorted(p.relative_to(regen) for p in regen.rglob("*.jsonl"))
        assert names, "重新生成没产出文件"
        for rel in names:
            assert (regen / rel).read_bytes() == (committed / rel).read_bytes(), \
                f"{rel} 与入库产物不再逐字节一致"


# ======================================================================
# ④ 确定性
# ======================================================================


def test_same_seed_twice_is_byte_identical():
    with tempfile.TemporaryDirectory() as tmp:
        first = write_qa_dataset(build_qa_dataset(PER_TYPE, SEED), Path(tmp) / "a").read_bytes()
        second = write_qa_dataset(build_qa_dataset(PER_TYPE, SEED), Path(tmp) / "b").read_bytes()
        assert first == second


def test_a_different_seed_selects_different_scenarios():
    """seed 得真的起作用：换 seed 换了内容，否则「同 seed 一致」证明不了任何确定性。"""
    a = {t.id: t.instruction for t in build_qa_dataset(PER_TYPE, SEED)}
    b = {t.id: t.instruction for t in build_qa_dataset(PER_TYPE, SEED + 1)}
    assert a != b


def test_output_has_no_timestamp_or_memory_address(qa_tasks):
    with tempfile.TemporaryDirectory() as tmp:
        text = write_qa_dataset(qa_tasks, Path(tmp)).read_text(encoding="utf-8")
    assert "0x" not in text
    assert "20" + "26-01-01T" not in text  # 不含 ISO 时间戳


# ======================================================================
# ⑤ evidence_ids 指向真实存在的文档
# ======================================================================


def test_evidence_ids_point_at_documents_that_exist(all_scenarios):
    for scenario in all_scenarios:
        doc_ids = {d["doc_id"] for d in scenario.docs}
        for point in scenario.points:
            for eid in point["evidence_ids"]:
                assert eid in doc_ids, f"{scenario.task_type} 的 {eid} 不在知识库里"


def test_every_task_document_has_a_unique_id(qa_tasks):
    for task in qa_tasks:
        ids = [d["doc_id"] for d in task.initial_data["knowledge"]]
        assert len(ids) == len(set(ids)), f"{task.id} 的 doc_id 有重复"


def test_evidence_ids_are_shared_across_questions_within_one_pool():
    """同一知识库里，不同题目的 evidence 必须指向不同文档 —— 全指同一篇说明是抄的。"""
    for task_type, pool in SCENARIOS.items():
        signature = {
            s.instruction: tuple(sorted({e for p in s.points for e in p["evidence_ids"]}))
            for s in pool
        }
        assert len(set(signature.values())) == len(pool), f"{task_type} 的证据组合重复了"


def test_multi_hop_scenarios_cite_at_least_two_documents():
    """多跳的定义就是跨 >=2 篇；引用不到两篇就不叫多跳。"""
    for scenario in SCENARIOS["多跳"]:
        cited = {e for p in scenario.points for e in p["evidence_ids"]}
        assert len(cited) >= 2, f"{scenario.instruction} 只引用了 {cited}"


# ======================================================================
# 参考答案只是示例
# ======================================================================


def test_reference_answer_is_labelled_as_an_example(qa_tasks):
    """键名就叫 ``example_answer`` —— judge 侧读代码时一眼能看出它不是判分依据。

    如果哪天有人拿它做语义等价比对（P0-1 明令禁止），这道判据就是拦他的那道门。
    """
    for task in qa_tasks:
        assert set(task.ground_truth) == {"example_answer"}
        assert task.ground_truth["example_answer"].strip()


def test_insufficient_and_unanswerable_examples_refuse_to_answer():
    """后两类的「正确答案」就是「答不出来」。示例答案若给了实答案，
    等于在教 judge 把「编造」当标准答案 —— 直接判它不合格。"""
    for task_type in ("证据不足", "不可回答"):
        for scenario in SCENARIOS[task_type]:
            answer = scenario.example_answer
            assert any(k in answer for k in ("无法", "需要先", "问题", "请", "确认")), \
                f"{task_type} 的示例答案没有体现「不作答/先澄清」: {answer[:40]}"


# ======================================================================
# CLI（判据 3）
# ======================================================================


def test_cli_exits_zero_and_writes_a_parseable_jsonl(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        code = main(["--seed", "42", "--per-type", "2", "--output-dir", tmp])
        assert code == 0
        payload = Path(tmp) / "qa.jsonl"
        assert payload.exists()
        rows = [json.loads(line) for line in payload.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 8
    for task_type in TASK_TYPES:
        assert [r["initial_data"]["任务类型"] for r in rows].count(task_type) == PER_TYPE
    for r in rows:
        assert r["category"] == "retrieval"
        assert r["rubric"]
    out = capsys.readouterr().out
    assert "TOTAL" in out and "8" in out


def test_cli_output_is_utf8_without_unicode_escapes():
    """中文原样落盘：``\\uXXXX`` 一律拒收，否则人工审题要先去解码。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = write_qa_dataset(build_qa_dataset(PER_TYPE, SEED), Path(tmp))
        raw = path.read_bytes()
        assert b"\\u" not in raw
        raw.decode("utf-8")  # 不是「恰好能解码」，是**确实是** utf-8
        text = raw.decode("utf-8")
        assert any("一" <= ch <= "鿿" for ch in text)
        assert b"\r\n" not in raw, "只允许 LF"


def test_cli_rejects_a_non_positive_per_type_before_writing(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "qa.jsonl"
        with pytest.raises(SystemExit) as exc:
            main(["--per-type", "0", "--output-dir", tmp])
        assert exc.value.code != 0
        assert not target.exists(), "非法输入不该落任何产物"


# ======================================================================
# TASK-013：场景模板 + 要点参数化 → 330 条三 split
#
# 上面那组验的是「手写 16 条还对不对」，这里验的是「构造器生成 330 条还对不对」。
# 七条对应判据 1 的 ①–⑦。
# ======================================================================


@pytest.fixture(scope="module")
def formal_tasks():
    return build_formal_dataset(seed=SEED)


def test_every_task_type_has_at_least_three_templates():
    """① 每种任务类型 ≥3 个模板（句法骨架）。"""
    for task_type in TASK_TYPES:
        assert len(TEMPLATES[task_type]) >= 3, f"{task_type} 只有 {len(TEMPLATES[task_type])} 个模板"


def test_every_template_slot_table_expands_into_distinct_scenarios():
    """① 每条槽位行展开出的 Scenario 两两不同 —— 去重的前置条件。

    指纹取整条 Scenario：问题、参考答案、文档全文、rubric 全部带上。
    只比其中一段会漏 —— 有模板的槽位只落在 rubric 上（kb-g1/kb-g5 的缺失项），
    问题一字不变，判分口径却不同。
    """
    for task_type in TASK_TYPES:
        for template in TEMPLATES[task_type]:
            seen = set()
            for slots in template.slots:
                scenario = _expand(template, slots)
                key = json.dumps([scenario.task_type, scenario.instruction, scenario.example_answer,
                                  scenario.docs, scenario.points], sort_keys=True, ensure_ascii=False)
                assert key not in seen, f"{template.base_id} 两条槽位展开后是同一道题"
                seen.add(key)


def test_every_expanded_rubric_meets_the_task_008_spec(formal_tasks):
    """② 330 条全部满足 TASK-008 rubric 规范（4–8 点、0/0.5/1 可判、critical、证据可定位）。

    不抽样 —— 全跑。参数化最容易出的岔子就是「某个槽位组合下 points 少了一个」，
    抽样恰好抽不中那种。

    两处刻意不写成「一律有」：
    - ``evidence_ids`` 可为空 —— 「不要编造{缺失项}」这类否定点本就没有出处可引，
      ``validate_rubric`` 同样不要求每点都带证据；这里只钉死**非空时必须指向真实文档**。
    - ``critical`` 只出现在否定两类（证据不足 / 不可回答）。这是 TASK-008 已定的口径
      （``CRITICAL_CAP`` 封顶是给「硬答」的惩罚，事实题没有可封顶的关键点），
      手写的 16 条场景也是这个分布 —— 断言按事实类「没有 critical」才是在钉语义。
    """
    for row in formal_tasks:
        task = row.task
        assert MIN_POINTS <= len(task.rubric) <= MAX_POINTS, f"{task.id} rubric 点数越界"
        doc_ids = {d["doc_id"] for d in task.initial_data["knowledge"]}
        validate_rubric(task.rubric, doc_ids)  # 复用生产校验器，不复写一遍规则
        assert abs(sum(p["weight"] for p in task.rubric) - 1.0) < 1e-9, f"{task.id} 权重没归一"
        for point in task.rubric:
            assert set(point["evidence_ids"]) <= doc_ids, f"{task.id} evidence_ids 指向不存在的文档"

        has_critical = any(p["critical"] for p in task.rubric)
        if row.task_type in ("证据不足", "不可回答"):
            assert has_critical, f"{task.id} 是否定类，却没有 critical 点"
        else:
            assert not has_critical, f"{task.id} 是事实类，不该有 critical 点（TASK-008 口径）"


def test_formal_ids_are_unique_and_content_is_not_duplicated(formal_tasks):
    """③ 330 条 id 唯一、内容不重复。

    内容签名只取「问题 + 参考答案 + 文档全文」，漏了任何一段都可能让两条其实
    同题的任务蒙混过关。id 重复则签名必然重复，分开断言是为了报错能指到真因。
    """
    ids = [row.task.id for row in formal_tasks]
    assert len(ids) == len(set(ids)), "有重复 task_id"
    signatures = [
        (row.task.instruction, row.task.ground_truth["example_answer"],
         json.dumps(row.task.initial_data["knowledge"], sort_keys=True, ensure_ascii=False))
        for row in formal_tasks
    ]
    assert len(signatures) == len(set(signatures)), "有两条内容完全相同（会重复计分）"


def test_formal_split_sizes_are_sixty_one_twenty_one_fifty(formal_tasks):
    """④ 三 split 数量 60/120/150，且每个 split 里四类任务都有覆盖。"""
    counts = {split: 0 for split in B_SPLITS}
    for row in formal_tasks:
        counts[row.split] += 1
    assert counts == FORMAL_SIZES == {"trial": 60, "dev": 120, "blind": 150}
    for split in B_SPLITS:
        covered = {row.task_type for row in formal_tasks if row.split == split}
        assert covered == set(TASK_TYPES), f"{split} 缺任务类型 {set(TASK_TYPES) - covered}"


def test_no_template_or_knowledge_base_leaks_across_splits(formal_tasks):
    """⑤ 防泄漏：同一模板 / 同一知识库的变体不跨 split。

    切分**按「模板 + 知识库来源」分组**，不按行随机切。文档 id 的前缀段就是知识库
    编号（kb-d1-002 → kb-d1），所以「知识库不跨 split」等价于「doc 前缀不跨 split」。
    """
    by_template: dict[str, set[str]] = {}
    by_kb: dict[str, set[str]] = {}
    for row in formal_tasks:
        by_template.setdefault(row.base_id, set()).add(row.split)
        for doc in row.task.initial_data["knowledge"]:
            kb = doc["doc_id"].rsplit("-", 1)[0]
            by_kb.setdefault(kb, set()).add(row.split)

    for base_id, splits in by_template.items():
        assert len(splits) == 1, f"模板 {base_id} 同时出现在 {sorted(splits)} —— 会泄漏"
    for kb, splits in by_kb.items():
        assert len(splits) == 1, f"知识库 {kb} 同时出现在 {sorted(splits)} —— 会泄漏"


def test_formal_generation_is_byte_identical_for_the_same_seed():
    """⑥ 同 seed 两次逐字节一致（先比较结构，再真的落盘比字节）。"""
    first, second = build_formal_dataset(seed=SEED), build_formal_dataset(seed=SEED)
    assert [(r.task.id, r.split, r.base_id) for r in first] == [
        (r.task.id, r.split, r.base_id) for r in second
    ]
    with tempfile.TemporaryDirectory() as tmp:
        paths_a = write_formal_dataset(first, Path(tmp) / "a")
        paths_b = write_formal_dataset(second, Path(tmp) / "b")
        for split in B_SPLITS:
            assert paths_a[split].read_bytes() == paths_b[split].read_bytes(), f"{split} 两次产物不一致"


def test_per_type_beyond_pool_capacity_is_rejected_not_recycled():
    """⑦ `--per-type > 场景池容量` 直接报错，不再循环复用同一批场景。"""
    capacity = min(len(pool) for pool in SCENARIOS.values())
    build_qa_dataset(per_type=capacity, seed=SEED)  # 正好等于容量：合法
    with pytest.raises(ValueError, match="超过场景池容量"):
        build_qa_dataset(per_type=capacity + 1, seed=SEED)
    with pytest.raises(ValueError, match="超过场景池容量"):
        main(["--seed", "42", "--per-type", str(capacity + 1), "--output-dir", "."])


def test_formal_cli_writes_three_split_files_with_unescaped_chinese():
    """判据 2：CLI 落三个目录、中文不转义、每类任务在各 split 有覆盖。"""
    with tempfile.TemporaryDirectory() as tmp:
        assert main(["--seed", "42", "--preset", "formal", "--output-dir", tmp]) == 0
        for split, expected in FORMAL_SIZES.items():
            path = Path(tmp) / split / "qa.jsonl"
            assert path.exists(), f"缺 {split}/qa.jsonl"
            raw = path.read_bytes()
            assert b"\\u" not in raw, f"{split} 出现 unicode 转义"
            raw.decode("utf-8")
            assert b"\r\n" not in raw
            rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
            assert len(rows) == expected
            assert {r["initial_data"]["任务类型"] for r in rows} == set(TASK_TYPES)


# ======================================================================
# partial_rule 覆盖性（TASK-015）—— 补中间态，不是放宽核心要求
# ======================================================================


def _unanswerable_templates() -> list:
    return list(TEMPLATES["不可回答"])


def test_unanswerable_c1_names_the_intermediate_credit_line():
    """不可回答类的 critical c1 必须定义「指出不足但未要求补充」这一中间态 → 0.5。

    TASK-014 的实测根因：c1 只有「硬答 → 0」一档，judge 面对「我这边信息不全，
    麻烦补充一下」这种**没点名要什么**的答案无档可给，只能判 0 ⇒ critical 封顶 ⇒
    整题作废（trial 不可回答 1/15 = 6.7%）。

    这条钉的是「三档都写出来」，不是「半分给得更容易」——「硬答 → 0」那档必须还在，
    下面 ``test_unanswerable_c1_still_zeroes_a_confident_hard_answer`` 单独钉它。
    """
    for template in _unanswerable_templates():
        point = template.points[0]
        assert point["critical"] is True, f"{template.base_id} 的 c1 不是 critical 点"
        rule = point["partial_rule"]
        assert "0.5" in rule, f"{template.base_id} 的 c1 没定义中间态: {rule!r}"
        assert partial_rule_kind(rule) == "ternary", f"{template.base_id} 的 c1 仍被判成二值"
        head = rule.split("0.5")[0].strip()
        assert head, f"{template.base_id} 的 c1 没写出 0.5 的触发情形: {rule!r}"
        assert len(head) >= 6, f"{template.base_id} 的 c1 触发情形太短，判分员读不出边界"


def test_unanswerable_c1_still_zeroes_a_confident_hard_answer():
    """补中间态不等于放过硬答：「直接给答复」那一档必须仍然是 → 0。

    没有这条，上面那条测试就能靠**删掉 0 档**来通过 —— 分数会好看，
    但「不可回答」从此不再惩罚编造，整类题失���意义。
    """
    for template in _unanswerable_templates():
        rule = template.points[0]["partial_rule"]
        assert "→ 0" in rule, f"{template.base_id} 的 c1 把「硬答」的 0 档丢了: {rule!r}"


def test_every_unanswerable_point_is_decidable_after_the_rewrite(formal_tasks):
    """改完仍全体过 ``validate_rubric``，且不可回答类每点都落在三值/二值之一。

    ``validate_rubric`` 已是生产校验器，复用它而不是复写一遍规则 ——
    这里真正多验的是「改完没有把某点改成不可判」。
    """
    for row in formal_tasks:
        if row.task_type != "不可回答":
            continue
        doc_ids = {d["doc_id"] for d in row.task.initial_data["knowledge"]}
        validate_rubric(row.task.rubric, doc_ids)
        for point in row.task.rubric:
            assert partial_rule_kind(point["partial_rule"]) != "undecidable", \
                f"{row.task.id} {point['criterion_id']} 不可判: {point['partial_rule']!r}"


def test_the_multi_hop_makeup_deadline_point_has_a_half_credit_line():
    """kb-m1 c5「特殊情况可补报」也补了中间态（抽检发现的第二处）。

    「提到还能补报、但没给宽限期天数」在这类题里既不是全对也不是全错，
    原来只有 → 0 一档，把「知道有补报」这种半对答案判成全错。
    """
    point = TEMPLATES["多跳"][0].points[4]
    assert point["critical"] is False and point["weight"] == 0.15
    assert partial_rule_kind(point["partial_rule"]) == "ternary", \
        f"kb-m1 c5 应为三值: {point['partial_rule']!r}"
    assert "→ 0" in point["partial_rule"], f"kb-m1 c5 把「以为过期一律不可报名」的 0 档丢了"


# ======================================================================
# 文本自相矛盾扫描（TASK-016）—— 缺陷 A / B 的常驻牙齿
# ======================================================================
#
# 这两条不是走查脚本，是**冻结后的长期防线**。正式集冻结之后不再改
# rubric/slots/docs，任何一次改动都不该让这两种矛盾悄悄回来 —— 而
# 「上一次改模板改出了矛盾」这件事，恰恰是没人会主动想到去查的那类。


def _all_task_texts(task) -> dict[str, str]:
    """一条任务里所有面向人的文本。矛盾只可能出现在给人看的地方。"""
    return {
        "instruction": task.instruction,
        "docs": " ".join(d["text"] for d in task.initial_data["knowledge"]),
        "example_answer": task.ground_truth["example_answer"],
        "rubric": " ".join(p["requirement"] for p in task.rubric),
    }


def test_no_slot_value_contradicts_the_parenthetical_note(formal_tasks):
    """缺陷 A：括号备注与主值不得打架（原 34 条：``B座4层（B座三层）``）。

    缺陷成因是「备注槽位在 ``_cross`` 的两个维度里都声明，第二维度只给备注值
    却没覆盖主槽位」，笛卡尔积展开后主值和备注来自不同维度。
    判据取「备注里的最后一个数词/量词 token 必须等于它前面最近的同一个 token」。
    """
    token = re.compile(r"([零一二三四五六七八九十百]+|\d+)(层|日)")
    bad = []
    for row in formal_tasks:
        answer = row.task.ground_truth["example_answer"]
        for m in re.finditer(r"（([^（）]*)）", answer):
            note = token.findall(m.group(1))
            if not note:
                continue
            head = token.findall(answer[: m.start()])
            if head and "".join(head[-1]) != "".join(note[-1]):
                bad.append(f'{row.task.id}: {"".join(head[-1])} vs 备注={"".join(note[-1])}')
    assert not bad, "主值与括号备注自相矛盾：\n" + "\n".join(bad)


def test_no_doubled_character_in_any_task_text(formal_tasks):
    """缺陷 B：相邻重复字（原 15 条：``60分分（含）至85分分为``）。

    成因是槽位值自带单位（``60分``）而模板又补了一次（``{延长下限}分``）。
    注意「分级分类」这类正常词里也有「分分」子串，所以只查**紧邻重复的单字**。
    """
    doubled = re.compile(r"(分|为|层|日|号|天)\1")
    bad = [
        f"{row.task.id}/{where}: {m.group(0)}"
        for row in formal_tasks
        for where, text in _all_task_texts(row.task).items()
        for m in doubled.finditer(text)
    ]
    assert not bad, "文本里有相邻重复字：\n" + "\n".join(bad)
