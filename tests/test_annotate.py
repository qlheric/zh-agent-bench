"""annotate 的离线测试：**全程不联网**（假 call 注入）。

钉判据 2 的五件事：

1. **双标独立**：A / B 是两次互不参考的调用（不同 call、不共享上下文）；
2. **一致率 / κ**：逐点一致率、Cohen's κ、无定义情形返回 ``None`` 而不是 0；
3. **judge vs 共识**：G 一致率、MAE、通过率翻转；
4. **降级计数**：失败/非法分值整条作废，且**不进一致率**；
5. 接线：prompt 复用 judge 的 rubric 渲染、reports 可写盘、CLI 参数齐。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from zh_agent_bench import annotate
from zh_agent_bench.annotate import (
    ANNOTATORS,
    LABELS,
    aggregate_g,
    annotate_one,
    build_annotator_prompt,
    cohens_kappa,
    consensus_scores,
    judge_vs_consensus,
    load_answer_records,
    parse_scores,
    point_pairs,
    render_report,
    run_double_annotation,
    summarize,
    write_outputs,
)
from zh_agent_bench.judge import build_judge_prompt
from zh_agent_bench.tasks.qa import build_qa_dataset
from zh_agent_bench.tasks.schema import Task


# --------------------------------------------------------------------------------------
# 替身
# --------------------------------------------------------------------------------------


@pytest.fixture
def tasks() -> list[Task]:
    return build_qa_dataset(per_type=1, seed=7)


def _ok_response(scores: dict[str, float]) -> dict:
    return {
        "choices": [
            {"message": {"content": json.dumps(
                {"scores": scores, "notes": "总体不错"}, ensure_ascii=False
            )}}
        ]
    }


def _perfect(task: Task) -> dict[str, float]:
    return {p["criterion_id"]: 1.0 for p in task.rubric}


def _half(task: Task) -> dict[str, float]:
    return {p["criterion_id"]: 0.5 for p in task.rubric}


def _recording_call(responses: list[dict], *, seen: list | None = None):
    """按序回放 ``responses`` 的假 call，并把每次收到的 messages 记进 ``seen``。"""
    pending = list(responses)

    def _call(messages: list[dict[str, Any]], **kwargs: Any) -> dict:
        if seen is not None:
            seen.append(messages)
        return pending.pop(0) if pending else make_error_response("假 call 没响应了")

    return _call


def make_error_response(message: str) -> dict:
    return {"error": {"code": "mock", "message": message}}


# --------------------------------------------------------------------------------------
# 1. prompt 复用
# --------------------------------------------------------------------------------------


def test_annotator_prompt_reuses_the_judge_rubric_rendering(tasks):
    """标注点必须与 judge **逐条一致** —— 两边看到不同细则，A/B 分歧就分不清是谁的锅。"""
    task = tasks[0]
    judge_user = build_judge_prompt(task, "答案文本")[1]["content"]
    annot_user = build_annotator_prompt(task, "答案文本")[1]["content"]

    for point in task.rubric:
        assert point["requirement"] in judge_user
        assert point["requirement"] in annot_user

    # 只换输出契约：claims（忠实度）不进标注者 prompt
    assert "claims" in judge_user
    assert "claims" not in annot_user
    # 判分原则整段保留
    assert annot_user.count("评分点") >= 1
    assert "只输出 JSON 对象" in annot_user


def test_annotator_prompt_keeps_the_rubric_order(tasks):
    """渲染顺序 = ``task.rubric`` 顺序。annotate 不做位置扰动（那是 bias 的活）。"""
    task = tasks[0]
    user = build_annotator_prompt(task, "答案")[1]["content"]
    positions = [user.find(p["criterion_id"]) for p in task.rubric]
    assert all(pos >= 0 for pos in positions)
    assert positions == sorted(positions)


# --------------------------------------------------------------------------------------
# 2. 单条标注：解析、降级
# --------------------------------------------------------------------------------------


def test_parse_scores_accepts_the_three_labels(tasks):
    task = tasks[0]
    scores, why = parse_scores(task.rubric, _perfect(task))
    assert why == ""
    assert scores == _perfect(task)
    assert set(scores.values()) <= set(LABELS)


@pytest.mark.parametrize(
    "raw, expect",
    [
        ({"c1": 0.7}, "不在"),
        ({"c1": "满分"}, "不是数字"),
        ({"c1": True}, "未给出分值"),
        ({"c1": None}, "未给出分值"),
    ],
)
def test_parse_scores_rejects_illegal_values(tasks, raw, expect):
    """非法分值整条作废。``True`` 单独测 —— ``float(True) == 1.0``，不挡会被当成满分。"""
    rubric = [{"criterion_id": "c1", "weight": 1.0}]
    scores, why = parse_scores(rubric, raw)
    assert scores is None
    assert expect in why


def test_parse_scores_rejects_a_missing_point(tasks):
    """漏判**不是** 0 分。judge 可以补 0（没提到就是 0），标注者补 0 就是伪造意见。"""
    task = tasks[0]
    partial = dict(list(_perfect(task).items())[:-1])
    scores, why = parse_scores(task.rubric, partial)
    assert scores is None
    assert "未给出分值" in why


def test_annotate_one_degrades_on_error_payload(tasks):
    task = tasks[0]
    record = annotate_one(task, "答案", model="m", call=lambda *a, **k: make_error_response("503"))
    assert record["ok"] is False
    assert record["scores"] is None
    assert "503" in record["reason"]


def test_annotate_one_degrades_on_raised_exception(tasks):
    """mock 直接抛异常也不该掀掉整批 —— 与 judge 的降级同形。"""

    def _boom(*_a: Any, **_k: Any) -> dict:
        raise RuntimeError("连接被重置")

    record = annotate_one(tasks[0], "答案", model="m", call=_boom)
    assert record["ok"] is False
    assert "RuntimeError" in record["reason"]


def test_annotate_one_strips_json_fence(tasks):
    """模型爱包围栏，剥不掉就等于扔掉一份完全正确的打分。"""
    task = tasks[0]
    fenced = {
        "choices": [{"message": {"content":
            "```json\n" + json.dumps({"scores": _perfect(task)}, ensure_ascii=False) + "\n```"}}]
    }
    record = annotate_one(task, "答案", model="m", call=lambda *a, **k: fenced)
    assert record["ok"] is True
    assert record["scores"] == _perfect(task)


def test_annotate_one_never_raises(tasks):
    for call in (
        lambda *a, **k: {"choices": []},
        lambda *a, **k: {"choices": [{"message": {"content": ""}}]},
        lambda *a, **k: {"choices": [{"message": {"content": "随便一段中文"}}]},
    ):
        record = annotate_one(tasks[0], "答案", model="m", call=call)
        assert record["ok"] is False
        assert record["reason"]


# --------------------------------------------------------------------------------------
# 3. 双标独立性
# --------------------------------------------------------------------------------------


def test_the_two_annotators_call_independently(tasks):
    """A / B 各一次调用，**不共享 messages**：B 拿到的 prompt 里没有 A 的分数。"""
    seen_a: list = []
    seen_b: list = []
    rows = run_double_annotation(
        [(tasks[0], "答案")],
        call_a=_recording_call([_ok_response(_perfect(tasks[0]))], seen=seen_a),
        call_b=_recording_call([_ok_response(_half(tasks[0]))], seen=seen_b),
        model_a="gpt-6-astra", model_b="deepseek-chat",
    )
    assert len(seen_a) == 1 and len(seen_b) == 1
    assert seen_a[0] is not seen_b[0]
    for prompt in seen_a + seen_b:
        blob = json.dumps(prompt, ensure_ascii=False)
        # 对方的分值不可能进 prompt。只查 `"c1": 1.0` 这种成对形态 ——
        # 指令里本来就有 `0 | 0.5 | 1` 的取值说明，按裸串查会误报。
        for point in tasks[0].rubric:
            cid = point["criterion_id"]
            assert f'"{cid}": 1.0' not in blob
            assert f'"{cid}": 0.5' not in blob

    row = rows[0]
    assert row["a"]["model"] == "gpt-6-astra"
    assert row["b"]["model"] == "deepseek-chat"
    assert row["a"]["scores"] == _perfect(tasks[0])
    assert row["b"]["scores"] == _half(tasks[0])


def test_the_two_annotators_use_two_different_channels():
    """两个标注者必须真的走两个服务 —— 同厂商两个模型是同一个脑子，不算独立。"""
    a, b = ANNOTATORS["a"], ANNOTATORS["b"]
    assert a["name"] != b["name"]
    assert a["api_key_env"] != b["api_key_env"]
    assert a["base_url_env"]      # A 走 LLMAPI 自定义端点


def test_one_annotator_failing_does_not_touch_the_other(tasks):
    rows = run_double_annotation(
        [(tasks[0], "答案")],
        call_a=lambda *a, **k: make_error_response("A 挂了"),
        call_b=_recording_call([_ok_response(_perfect(tasks[0]))]),
        model_a="gpt-6-astra", model_b="deepseek-chat",
    )
    assert rows[0]["a"]["ok"] is False
    assert rows[0]["b"]["ok"] is True


# --------------------------------------------------------------------------------------
# 4. 一致率 / κ / 共识
# --------------------------------------------------------------------------------------


def test_point_pairs_sorted_by_criterion_id():
    a = {"c2": 1.0, "c1": 0.0}
    b = {"c2": 0.0, "c1": 0.0}
    assert point_pairs([(a, b)]) == [(0.0, 0.0), (1.0, 0.0)]


def test_consensus_takes_the_shared_value_when_they_agree():
    assert consensus_scores({"c1": 1.0}, {"c1": 1.0}) == {"c1": 1.0}


def test_consensus_snaps_the_midpoint_to_a_legal_label():
    """0 与 0.5 的中点是 0.25，合法档里没有 —— 必须吸档，否则 aggregate 会抛。"""
    merged = consensus_scores({"c1": 0.0, "c2": 1.0}, {"c1": 0.5, "c2": 0.0})
    assert set(merged.values()) <= set(LABELS)
    assert merged["c1"] == 0.0   # 0.25 平手取低档（行为确定）
    assert merged["c2"] == 0.5   # 1 与 0 的中点 0.5 本身就是合法档


def test_cohens_kappa_matches_a_hand_computed_case():
    # 6 对：A=[1,1,1,0,0,0] B=[1,1,1,0,0,0] ⇒ 一致率 1，期望一致率 0.5 ⇒ κ = 1.0
    pairs = [(1.0, 1.0)] * 3 + [(0.0, 0.0)] * 3
    assert cohens_kappa(pairs) == pytest.approx(1.0)
    # 全同档 ⇒ 期望一致率 = 1 ⇒ 公式 0/0
    assert cohens_kappa([(1.0, 1.0)] * 5) is None
    assert cohens_kappa([]) is None


def test_summarize_reports_agreement_and_kappa(tasks):
    task = tasks[0]
    rows = run_double_annotation(
        [(task, "答案")],
        call_a=_recording_call([_ok_response(_perfect(task))]),
        call_b=_recording_call([_ok_response(_perfect(task))]),
        model_a="gpt-6-astra", model_b="deepseek-chat",
    )
    stats = summarize(rows)
    assert stats["n_items"] == 1
    assert stats["n_both_ok"] == 1
    assert stats["n_points"] == len(task.rubric)
    assert stats["point_agreement"] == 1.0
    # 全给满分 ⇒ 单档 ⇒ κ 无定义（报 None 而不是 0）
    assert stats["cohens_kappa"] is None
    assert stats["n_consensus_by_midpoint"] == 0


def test_summarize_excludes_degraded_rows_from_agreement(tasks):
    """降级条**不进一致率** —— 把它当 0 分会把「没干活」记成「他判 0」。"""
    task = tasks[0]
    rows = run_double_annotation(
        [(task, "答案")],
        call_a=_recording_call([_ok_response(_perfect(task))]),
        call_b=lambda *a, **k: make_error_response("B 挂了"),
        model_a="gpt-6-astra", model_b="deepseek-chat",
    )
    stats = summarize(rows)
    assert stats["n_items"] == 1
    assert stats["n_both_ok"] == 0
    assert stats["n_points"] == 0
    assert stats["point_agreement"] is None      # 没数据 ≠ 一致率 0
    assert stats["n_degraded_b"] == 1
    assert stats["n_degraded_a"] == 0
    assert any("B 挂了" in r for r in stats["degradation_reasons"]["b"])


def test_aggregate_g_matches_the_generation_time_aggregation(tasks):
    """共识 G 必须用**生成期那一份**加权 + 关键点封顶口径，否则比的是两套规则。"""
    from zh_agent_bench.tasks.qa import aggregate_rubric

    task = tasks[0]
    scores = _perfect(task)
    assert aggregate_g(task.rubric, scores) == pytest.approx(
        aggregate_rubric(list(task.rubric), scores)[0]
    )


def test_aggregate_g_returns_none_on_bad_input(tasks):
    """一条坏样本返回 None 而不是抛 —— 不该掀掉整份报告。"""
    assert aggregate_g(tasks[0].rubric, {"不存在的点": 1.0}) is None
    assert aggregate_g([], {}) is None


# --------------------------------------------------------------------------------------
# 5. judge vs 共识
# --------------------------------------------------------------------------------------


def test_judge_vs_consensus_compares_g_and_counts_pass_flips(tasks):
    task = tasks[0]
    rows = run_double_annotation(
        [(task, "答案")],
        call_a=_recording_call([_ok_response(_perfect(task))]),
        call_b=_recording_call([_ok_response(_perfect(task))]),
        model_a="gpt-6-astra", model_b="deepseek-chat",
    )
    aligned = judge_vs_consensus(rows, [{"task_id": task.id, "G": 1.0}])
    assert aligned["n"] == 1
    assert aligned["agreement"] == 1.0
    assert aligned["mean_abs_error"] == pytest.approx(0.0)
    assert aligned["n_pass_flip"] == 0
    assert aligned["n_divergence"] == 0

    # judge 判 0.5：仍过线（≥0.8？不 —— 0.5 < 0.8）⇒ 与共识（1.0）翻转
    flipped = judge_vs_consensus(rows, [{"task_id": task.id, "G": 0.5}])
    assert flipped["n_pass_flip"] == 1
    assert flipped["pass_flip_rate"] == 1.0
    assert flipped["mean_abs_error"] == pytest.approx(0.5)


def test_judge_vs_consensus_lists_large_divergences(tasks):
    task = tasks[0]
    rows = run_double_annotation(
        [(task, "答案")],
        call_a=_recording_call([_ok_response(_half(task))]),
        call_b=_recording_call([_ok_response(_half(task))]),
        model_a="gpt-6-astra", model_b="deepseek-chat",
    )
    out = judge_vs_consensus(rows, [{"task_id": task.id, "G": 1.0}])
    assert out["n_divergence"] == 1
    assert out["divergences"][0]["task_id"] == task.id
    assert out["divergences"][0]["judge_G"] == 1.0


def test_judge_vs_consensus_skips_incomparable_rows(tasks):
    """judge 没这一条 / 共识聚合不出来 ⇒ 跳过，不按 0 算。"""
    task = tasks[0]
    rows = run_double_annotation(
        [(task, "答案")],
        call_a=_recording_call([_ok_response(_perfect(task))]),
        call_b=_recording_call([_ok_response(_perfect(task))]),
        model_a="gpt-6-astra", model_b="deepseek-chat",
    )
    out = judge_vs_consensus(rows, [{"task_id": "别人的任务", "G": 1.0}])
    assert out["n"] == 0
    assert out["agreement"] is None
    assert out["mean_abs_error"] is None


# --------------------------------------------------------------------------------------
# 6. 接线：records 对齐、产物、报告
# --------------------------------------------------------------------------------------


def _write_records(path: Path, tasks: list[Task], *, with_answer: bool = True) -> None:
    lines = []
    for task in tasks:
        row: dict[str, Any] = {"task_id": task.id, "G": 1.0, "correct": True}
        if with_answer:
            row["answer"] = (task.ground_truth or {}).get("example_answer")
        lines.append(json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_answer_records_pairs_tasks_with_answers(tmp_path, monkeypatch):
    """全 60 条对上 ⇒ 返回 (task, answer) 配对，且条数 = 正式集条数。

    这里**必须用真正式集**（`data_root=None`）：这个函数的全部价值就在
    「产物与正式集同源」这道校验上，用临时目录伪造数据集反而测不到它。
    """
    from zh_agent_bench.calibrate import load_formal_split

    real = load_formal_split("trial")
    records = tmp_path / "records.jsonl"
    _write_records(records, real)

    pairs = load_answer_records("trial", records)
    assert len(pairs) == len(real) == 60
    assert [t.id for t, _ in pairs] == [t.id for t in real]
    assert all(isinstance(a, str) and a for _, a in pairs)


def test_load_answer_records_rejects_a_records_file_from_another_snapshot(tasks, tmp_path):
    """只写了 4 条却声称是 trial ⇒ 对不上必须报错，不是静默跳过。

    静默跳过会让一致率看着正常而实际样本对不上，是双标里最致命的错。
    """
    records = tmp_path / "records.jsonl"
    _write_records(records, tasks)
    with pytest.raises(ValueError, match="不同源"):
        load_answer_records("trial", records)


def test_load_answer_records_honours_limit(tmp_path):
    from zh_agent_bench.calibrate import load_formal_split

    records = tmp_path / "records.jsonl"
    _write_records(records, load_formal_split("trial"))
    assert len(load_answer_records("trial", records, limit=3)) == 3


def test_load_answer_records_requires_the_answer_field(tasks, tmp_path):
    records = tmp_path / "records.jsonl"
    _write_records(records, tasks, with_answer=False)
    with pytest.raises(ValueError, match="--save-answers"):
        load_answer_records("trial", records, data_root=tmp_path)


def test_load_answer_records_missing_file_mentions_how_to_produce_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="--save-answers"):
        load_answer_records("trial", tmp_path / "没有这个.jsonl", data_root=tmp_path)


def test_write_outputs_and_report_are_timestamp_free(tasks, tmp_path):
    task = tasks[0]
    rows = run_double_annotation(
        [(task, "答案")],
        call_a=_recording_call([_ok_response(_perfect(task))]),
        call_b=_recording_call([_ok_response(_half(task))]),
        model_a="gpt-6-astra", model_b="deepseek-chat",
    )
    stats = summarize(rows, judge_records=[{"task_id": task.id, "G": 0.6}])
    path = write_outputs(
        rows, stats, tmp_path, split="trial",
        model_a="gpt-6-astra", model_b="deepseek-chat", judge_model="deepseek-chat",
    )
    text = path.read_text(encoding="utf-8")
    assert (tmp_path / "annotations.jsonl").is_file()
    assert path.name == "annotate.md"
    assert "标注者间一致率" in text
    assert "judge vs 共识" in text
    assert "降级计数" in text
    # 两次渲染逐字节一致（LLM 随机性只允许影响分数，不允许影响结构）
    again = render_report(stats, split="trial", model_a="gpt-6-astra",
                          model_b="deepseek-chat", judge_model="deepseek-chat")
    assert again == text


def test_main_without_keys_returns_3_instead_of_crashing(monkeypatch, tmp_path):
    """无 key 是「不能跑」而不是「跑挂了」—— 判据 4 明确不算失败。"""
    for name in ("LLMAPI_API_KEY", "DEEPSEEK_API_KEY", "LLMAPI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    assert annotate.main(["--split", "trial", "--records", str(tmp_path / "x.jsonl")]) == 3
