"""B 场景 judge 测试 —— rubric 逐点打分 + G/F/T 三指标。

判据 §3.1 的八个小项在这里一一对应：

===== ========================================================
①     mock judge 返回逐点 ``{criterion_id: score}`` → G 聚合正确、critical 封顶生效
②     judge 漏 ``criterion_id`` → 该点按 0 记并写明原因（不崩）
③     judge 给非法分值（非 0/0.5/1） → 该点按 0 记并写明原因（不崩）
④     judge LLM 失败 / 抛异常 → 不适用分数 + 原因，不崩
⑤     F 的 claim-证据解析（支持 / 矛盾 / 证据不足）
⑥     F 空答案 = **不适用**（不是满分）
⑦     T 从轨迹机械判（合法率）
⑧     关键证据覆盖率计算正确
===== ========================================================

**全程不联网**：judge 的 LLM 调用被 ``monkeypatch`` 成脚本化假响应。
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from zh_agent_bench import judge as judge_mod
from zh_agent_bench import llm_client
from zh_agent_bench.judge import (
    CONTRADICTED,
    INSUFFICIENT,
    PASS_THRESHOLD_G,
    SUPPORTED,
    build_judge_prompt,
    judge,
    judge_task,
)
from zh_agent_bench.tasks import qa
from zh_agent_bench.tasks.schema import (
    AnswerKind,
    AnswerSpec,
    Category,
    Comparison,
    Normalization,
    Split,
    Task,
)

DOC_A = "kb-t-001"
DOC_B = "kb-t-002"

DOCS = [
    {
        "doc_id": DOC_A,
        "title": "差旅费用报销管理办法",
        "text": "本办法自2026年3月1日起执行。飞机票一律 Economy 舱，折扣舱需附部门负责人书面说明。",
    },
    {
        "doc_id": DOC_B,
        "title": "澜舟数据平台版本发布节奏",
        "text": "澜舟数据平台采用双周发布：奇数周发缺陷修复版本，偶数周发功能版本。",
    },
]


# --------------------------------------------------------------------------------------
# 测试脚手架
# --------------------------------------------------------------------------------------


def _point(cid: str, weight: float, *, critical: bool = False, evidence=(DOC_A,)) -> dict:
    """造一个与 :func:`qa._p` 同形但更瘦的评分点（字段集由 validate_rubric 保证一致）。"""
    return {
        "criterion_id": cid,
        "requirement": f"要点 {cid}",
        "weight": weight,
        "evidence_ids": list(evidence),
        "partial_rule": "只答一半 → 0.5" if critical else "缺任一要素 → 0",
        "accepted_alternatives": [f"{cid} 的等价说法"],
        "critical": critical,
    }


#: 默认 rubric 的四个 criterion_id（``qa.MIN_POINTS`` 是 4，两点会被 validate_rubric 拒收）。
C1, C2, C3, C4 = "t-0001-c1", "t-0001-c2", "t-0001-c3", "t-0001-c4"
ALL_IDS = (C1, C2, C3, C4)

#: 简写名 → criterion_id。写测试时 ``_all(C2=0)`` 比 ``_all(**{"t-0001-c2": 0})`` 好读，
#: 但 ``criterion_id`` 是 ``"t-0001-c2"`` 而简写是 ``C2`` —— 两边得有个映射，
#: 否则关键字会被当成新 criterion_id 塞进 scores，静默全满分。
_BY_NAME = {"C1": C1, "C2": C2, "C3": C3, "C4": C4}

#: 权重刻意不等 —— 四个点权重相同的话，「加权」和「数对几个」在数值上撞车，
#: 测不出程序到底走没走加权。
DEFAULT_RUBRIC = [
    _point(C1, 0.4),
    _point(C2, 0.3, critical=True, evidence=(DOC_B,)),
    _point(C3, 0.2),
    _point(C4, 0.1, evidence=(DOC_B,)),
]


def _all(**overrides) -> dict:
    """「四点全对」的 scores；``overrides`` 用简写名（``C2=0.0``）覆盖其中几点。"""
    payload = dict.fromkeys(ALL_IDS, 1.0)
    for name, value in overrides.items():
        payload[_BY_NAME[name]] = value
    return payload


def _task(rubric=None, *, docs=None, instruction="测试问题", validate=True) -> Task:
    """一条 B 场景任务。默认四个评分点（含一个 critical），权重和为 1。

    :param validate: 关掉可造「rubric 引了库里没有的文档」这种**故意不合法**的任务，
        用来验 judge 遇到坏数据是降级而不是抛。
    """
    rubric = rubric if rubric is not None else [dict(p) for p in DEFAULT_RUBRIC]
    if validate:
        qa.validate_rubric(rubric, {d["doc_id"] for d in (docs if docs is not None else DOCS)})
    return Task(
        id="t-0001",
        category=Category.RETRIEVAL,
        schema_version="1.1",
        instruction=instruction,
        initial_data={"任务类型": "单跳", "knowledge": docs if docs is not None else DOCS},
        tool_names=(qa.QA_TOOL_NAME,),
        ground_truth={"example_answer": "这是示例，不是判分依据"},
        answer_spec=AnswerSpec(
            kind=AnswerKind.STRING,
            comparison=Comparison.JUDGE,
            normalization=Normalization.NONE,
        ),
        split=Split.TRAIN,
        rubric=tuple(rubric),
    )


class ScriptedChat:
    """替换 :func:`llm_client.chat` 的假 judge。

    记录每次调用的 messages/tools/model，方便断言「prompt 逐条渲染」与「temperature=0」。
    """

    def __init__(self, *replies, error: str | None = None, raises: bool = False) -> None:
        self.replies = list(replies)
        self.error = error
        self.raises = raises
        self.calls: list[dict] = []

    def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.raises:
            raise ConnectionResetError("mock: judge 炸了")
        if self.error is not None:
            return llm_client.make_error(self.error, "mock 注入的失败")
        content = self.replies.pop(0) if self.replies else "{}"
        return {"choices": [{"message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}]}


def _reply(scores=None, claims=None, notes="ok") -> str:
    return json.dumps(
        {"scores": scores or {}, "claims": claims or [], "notes": notes},
        ensure_ascii=False,
    )


@pytest.fixture
def fake_chat(monkeypatch):
    """装上脚本化 judge，并交还装它的钩子（``set_reply`` / ``set_error`` / ``set_raises``）。"""

    def install(*replies, error=None, raises=False):
        scripted = ScriptedChat(*replies, error=error, raises=raises)
        monkeypatch.setattr(llm_client, "chat", scripted)
        return scripted

    return install


def _traj(*calls):
    return list(calls)


def _call(tool=qa.QA_TOOL_NAME, arguments=None, hits=(), illegal=False):
    """造一条轨迹记录，形状照 harness ``_ToolBox``：失败时带 ``error`` 键。

    ``illegal=True`` 时 ``result`` 是错误哨兵、**不带 hits** —— 与真实工具行为一致
    （契约错误时结果是 error，压根没有检索结果）。
    """
    entry = {
        "tool": tool,
        "arguments": arguments if arguments is not None else {"query": "舱位"},
        "result": {"hits": [{"doc_id": d, "title": "", "text": "", "score": 1} for d in hits]},
    }
    if illegal:
        entry["error"] = {"code": "invalid_input", "message": "query 为空"}
        entry["result"] = entry["error"]
    return entry


# --------------------------------------------------------------------------------------
# 分流：A 场景四算子必须一字不受影响
# --------------------------------------------------------------------------------------


def test_judge_specs_without_task_take_the_a_scene_path(monkeypatch):
    """不带 ``task`` 走 A 场景 → 四算子分支，**一次都不调 LLM**。

    没有这条守卫，B 场景的 import 一坏，A 场景评测也会跟着炸。
    """
    called = []
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: called.append(1))
    spec = AnswerSpec(kind=AnswerKind.STRING, comparison=Comparison.EXACT,
                      normalization=Normalization.NONE)
    r = judge("客户服务部", {"answer": "客户服务部"}, spec)
    assert r["correct"] is True
    assert r["reason"] == "逐字相等"
    assert called == []


def test_a_scene_result_keeps_its_original_keys(monkeypatch):
    """A 场景返回结构原封不动 —— 不多不少 G/F/T 键。"""
    spec = AnswerSpec(kind=AnswerKind.STRING, comparison=Comparison.EXACT,
                      normalization=Normalization.NONE)
    r = judge("x", {"answer": "x"}, spec, task=_task())
    assert set(r) == {"correct", "reason", "normalized_answer", "normalized_truth"}


# --------------------------------------------------------------------------------------
# ① 逐点打分 → G 聚合；critical 封顶生效
# --------------------------------------------------------------------------------------


#: 一条「rubric 全对 + claim 全支持」的干净回复 —— 想要 ``correct is True`` 就用它。
CLEAN_CLAIM = {"claim": "默认经济舱", "verdict": SUPPORTED, "evidence_id": DOC_A}


def test_g_is_the_weighted_coverage_of_the_rubric(fake_chat):
    fake_chat(_reply(_all(), claims=[CLEAN_CLAIM]))
    r = judge_task(_task(), "答全了")
    assert r["G"] == pytest.approx(1.0)
    assert r["scores"] == dict.fromkeys(ALL_IDS, 1.0)
    assert r["capped"] is False
    assert r["correct"] is True


def test_g_is_a_weighted_average_not_a_point_count(fake_chat):
    """只答中最轻的普通点（w=0.1）⇒ G=0.1，而不是「四点中答对一点 ⇒ 0.25」。"""
    fake_chat(_reply(_all(C1=0.0, C2=0.0, C3=0.0)))
    r = judge_task(_task(), "只答了最轻的一条")
    assert r["G"] == pytest.approx(0.1)


def test_half_credit_is_carried_through_to_g(fake_chat):
    """关键点 0.5 ⇒ G = 0.3×0.5 + 0.4+0.2+0.1 = 0.85。"""
    fake_chat(_reply(_all(C2=0.5)))
    assert judge_task(_task(), "x")["G"] == pytest.approx(0.85)


def test_a_missed_critical_point_caps_g(fake_chat):
    """普通点全对（原始 0.7）、关键点得 0 ⇒ 封到 CRITICAL_CAP=0.5。"""
    fake_chat(_reply(_all(C2=0.0)))
    r = judge_task(_task(), "漏了关键点")
    assert r["G"] == pytest.approx(qa.CRITICAL_CAP)
    assert r["capped"] is True
    assert r["correct"] is False


def test_capping_never_raises_a_score(fake_chat):
    """关键点得 0 但普通点也低时，封顶不抬分。"""
    fake_chat(_reply(_all(C1=0.0, C2=0.0, C3=0.0, C4=0.0)))
    r = judge_task(_task(), "什么都没答到")
    assert r["G"] == 0.0
    assert r["capped"] is True


def test_a_half_credited_critical_point_does_not_cap(fake_chat):
    """封顶只在关键点得 0 时触发 —— 给半分说明「说到了但没全对」，不算关键错误。"""
    fake_chat(_reply(_all(C2=0.5)))
    r = judge_task(_task(), "关键点只答一半")
    assert r["capped"] is False
    assert r["G"] == pytest.approx(0.85)


def test_g_below_the_frozen_threshold_is_not_correct(fake_chat):
    """阈值来自代码常量，不因 judge 打分而浮动。"""
    fake_chat(_reply(_all(C2=0.0, C1=0.5)))
    r = judge_task(_task(), "差一口气")
    assert r["G"] < PASS_THRESHOLD_G
    assert r["correct"] is False


# --------------------------------------------------------------------------------------
# ② ③ judge 返回残缺 / 非法：不崩，按 0 记并写明原因
# --------------------------------------------------------------------------------------


def test_a_missing_criterion_scores_zero_with_a_reason(fake_chat):
    fake_chat(_reply(_all(C2=None, C3=None, C4=None)))  # 三个点没给
    r = judge_task(_task(), "只答了前一条")
    assert r["scores"][C2] == 0.0
    assert C2 in r["reason"]
    assert "judge 未给出分值" in r["reason"]


def test_an_illegal_score_scores_zero_with_a_reason(fake_chat):
    """0.7 不在白名单 ⇒ 按 0 记。静默接受会让分数悄悄漂掉。"""
    fake_chat(_reply(_all(C1=0.7)))
    r = judge_task(_task(), "x")
    assert r["scores"][C1] == 0.0
    assert "不在 0/0.5/1 内" in r["reason"]
    assert r["G"] == pytest.approx(0.6)


@pytest.mark.parametrize("bad", ["满分", None, [1.0], {"s": 1}, True])
def test_unparseable_scores_never_crash(fake_chat, bad):
    fake_chat(_reply(dict.fromkeys(ALL_IDS, bad)))
    r = judge_task(_task(), "x")
    assert r["scores"] == dict.fromkeys(ALL_IDS, 0.0)
    assert r["reason"]


def test_a_non_dict_scores_block_is_rejected_whole(fake_chat):
    fake_chat(json.dumps({"scores": [1, 0, 1], "claims": []}, ensure_ascii=False))
    r = judge_task(_task(), "x")
    assert r["G"] == 0.0
    assert "未返回 scores 对象" in r["reason"]


def test_scores_keyed_by_position_one_is_not_silently_mapped(fake_chat):
    """judge 按序号（1/2/3/4）而非 criterion_id 打分 ⇒ 不猜、不映射，全 0 + 原因。"""
    fake_chat(_reply({"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0}))
    r = judge_task(_task(), "x")
    assert r["G"] == 0.0
    assert "judge 未给出分值" in r["reason"]


# --------------------------------------------------------------------------------------
# ④ judge LLM 失败 / 网络异常 / 返回垃圾：降级不崩
# --------------------------------------------------------------------------------------


def test_an_llm_error_sentinel_degrades_to_zero(fake_chat):
    fake_chat(error="llm_network_error")
    r = judge_task(_task(), "x")
    assert r["G"] == 0.0
    assert r["correct"] is False
    assert "judge 调用失败" in r["reason"]


def test_an_llm_exception_degrades_to_zero(fake_chat):
    """llm_client 自己不抛，但 mock 会抛 —— 判分器这一层还得兜住。"""
    fake_chat(raises=True)
    r = judge_task(_task(), "x")
    assert r["G"] == 0.0
    assert "judge 调用失败" in r["reason"]


def test_a_response_without_choices_degrades(fake_chat, monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: {"id": "x", "object": "chat.completion"})
    r = judge_task(_task(), "x")
    assert r["G"] == 0.0
    assert "choices" in r["reason"]


@pytest.mark.parametrize("content", ["", "   ", "抱歉，我无法完成评分", "[1, 2, 3]", "{\"scores\": "])
def test_unparseable_model_output_degrades_without_crashing(monkeypatch, content):
    monkeypatch.setattr(
        llm_client, "chat",
        lambda *a, **k: {"choices": [{"message": {"content": content}}]},
    )
    r = judge_task(_task(), "x")
    assert r["scores"] == dict.fromkeys(ALL_IDS, 0.0)
    assert r["reason"]


def test_a_model_wrapped_in_a_code_fence_is_still_parsed(fake_chat):
    """模型爱把 JSON 包进 ```json … ```，直接 loads 会把一份正确打分扔掉。"""
    fenced = "```json\n" + _reply(_all()) + "\n```"
    fake_chat(fenced)
    assert judge_task(_task(), "x")["G"] == pytest.approx(1.0)


def test_degradation_leaves_f_and_t_untouched_by_the_llm(fake_chat):
    """judge 挂了 F 自然无从谈起（不适用），但 T 是机械算的 —— 不该跟着塌。"""
    fake_chat(error="llm_no_api_key")
    r = judge_task(_task(), "x", trajectory=_traj(_call(hits=(DOC_A,))))
    assert r["F"] is None
    assert r["T"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# ⑤ F：claim-证据解析
# --------------------------------------------------------------------------------------


def test_f_counts_the_supported_share_of_claims(fake_chat):
    fake_chat(_reply(
        _all(),
        claims=[
            {"claim": "飞机票默认经济舱", "verdict": SUPPORTED, "evidence_id": DOC_A},
            {"claim": "冻结期是 1 月", "verdict": CONTRADICTED, "evidence_id": DOC_B},
            {"claim": "公司食堂 24 小时营业", "verdict": INSUFFICIENT, "evidence_id": ""},
        ],
    ))
    r = judge_task(_task(), "答了三条")
    assert r["F"] == pytest.approx(1 / 3)
    assert r["n_contradicted"] == 1
    assert r["n_unsupported"] == 1
    assert r["correct"] is False, "有矛盾/无依据断言 ⇒ 不算成功"


def test_f_is_one_when_every_claim_is_supported(fake_chat):
    fake_chat(_reply(
        _all(),
        claims=[
            {"claim": "默认经济舱", "verdict": SUPPORTED, "evidence_id": DOC_A},
            {"claim": "双周发布", "verdict": SUPPORTED, "evidence_id": DOC_B},
        ],
    ))
    r = judge_task(_task(), "答了")
    assert r["F"] == pytest.approx(1.0)
    assert r["correct"] is True


def test_an_unknown_verdict_falls_back_to_insufficient(fake_chat):
    """认不出的判定归最保守的一档，而不是悄悄当成「支持」。"""
    fake_chat(_reply(
        _all(),
        claims=[{"claim": "x", "verdict": "大概对", "evidence_id": ""}],
    ))
    r = judge_task(_task(), "x")
    assert r["claims"][0]["verdict"] == INSUFFICIENT
    assert r["F"] == 0.0
    assert "不合法" in r["reason"]


def test_malformed_claims_are_dropped_with_a_reason(fake_chat):
    fake_chat(_reply(
        _all(),
        claims=[
            "不是对象",
            {"verdict": SUPPORTED, "evidence_id": ""},   # 缺 claim 文本
            {"claim": "  ", "verdict": SUPPORTED, "evidence_id": ""},
            {"claim": "真断言", "verdict": SUPPORTED},
        ],
    ))
    r = judge_task(_task(), "x")
    assert [c["claim"] for c in r["claims"]] == ["真断言"]
    assert r["claims"][0]["evidence_id"] == ""
    assert "已丢弃" in r["reason"]


def test_a_non_list_claims_block_yields_no_claims(fake_chat):
    fake_chat(_reply(_all(), claims="支持"))
    r = judge_task(_task(), "x")
    assert r["claims"] == []
    assert r["F"] is None


# --------------------------------------------------------------------------------------
# ⑤-b TASK-011：F 的 claim 抽取口径 + 声明型放宽 unsupported
#
# 病根（TASK-010 实测）：声明型两类一致率塌到 25%/75%，5 条是 ``G=1.0`` 满分被判错。
# judge 把三类**不可核**句子抽成待证 claim，再拿检索证据去证 ⇒ ``unsupported ≥ 1``。
# --------------------------------------------------------------------------------------


def _task_of_type(task_type: str) -> Task:
    """同 :func:`_task`，只换 ``initial_data["任务类型"]`` —— 它决定 unsupported 的严格程度。"""
    task = _task()
    data = dict(task.initial_data)
    data["任务类型"] = task_type
    return dataclasses.replace(task, initial_data=data)


def test_judge_prompt_tells_the_model_which_sentences_are_not_claims(fake_chat):
    """prompt 必须**逐条列明**该排除的三类句子，否则模型无从判断。

    只写「只收可核事实断言」不够 —— 模型会把「库里的文档没有提到 X」当成可核断言
    （TASK-010 里它就是这么做的）。判据句（「这句话错了世界会不会跟着变」）给模型
    一把尺子，三条清单是拿尺子去量的具体对象，两者都要在。
    """
    scripted = fake_chat(_reply(_all(), claims=[]))
    judge_task(_task(), "答案")
    system = scripted.calls[0]["messages"][0]["content"]
    assert "行动建议" in system
    assert "元陈述" in system
    assert "元话语" in system
    assert "世界会不会跟着变" in system


def test_declarative_tasks_tolerate_unsupported_but_never_contradiction(fake_chat):
    """声明型：有「证据不足」的断言不算错，但有「矛盾」的断言照样算错。

    声明型正确答案本身就包含「库里没有 X / 我答不了」这类句子，用 ``unsupported == 0``
    卡它等于**用正确答案的固有特征否决正确答案**（TASK-010 实测把 G=1.0 全毙了）。
    **矛盾任何类型都不放过** —— 与证据原文直接冲突在哪种任务里都是硬伤。
    """
    for task_type in ("证据不足", "不可回答"):
        fake_chat(_reply(_all(), claims=[
            {"claim": "库里没有任何文档给出额度表", "verdict": INSUFFICIENT, "evidence_id": ""},
        ]))
        r = judge_task(_task_of_type(task_type), "答了")
        assert r["G"] == pytest.approx(1.0)
        assert r["n_unsupported"] == 1
        assert r["correct"] is True, f"{task_type}: 声明型不该被 unsupported 否决"

        fake_chat(_reply(_all(), claims=[
            {"claim": "住宿费每晚上限 800 元", "verdict": CONTRADICTED, "evidence_id": DOC_A},
        ]))
        r = judge_task(_task_of_type(task_type), "答了")
        assert r["n_contradicted"] == 1
        assert r["correct"] is False, f"{task_type}: 矛盾任何类型都要拦住"


def test_factual_tasks_keep_requiring_zero_unsupported(fake_chat):
    """事实型**不放宽**：那里每个数字都该有出处，无依据断言是实打实的幻觉。"""
    fake_chat(_reply(_all(), claims=[
        {"claim": "住宿费每晚上限 800 元", "verdict": INSUFFICIENT, "evidence_id": ""},
    ]))
    r = judge_task(_task_of_type("单跳"), "答了")
    assert r["n_unsupported"] == 1
    assert r["correct"] is False


def test_task_type_must_stay_in_sync_with_the_generator():
    """``judge.DECLARATIVE_TASK_TYPES`` 是 judge 侧的副本（不能 import qa，会成环）。

    两份字面量各写一份、由本测试钉住一致，比多一条模块依赖便宜。
    """
    assert judge_mod.DECLARATIVE_TASK_TYPES == {"证据不足", "不可回答"}
    assert judge_mod.DECLARATIVE_TASK_TYPES <= set(qa.TASK_TYPES)


def test_verdict_reports_which_task_type_it_used(fake_chat):
    """返回值带 ``task_type`` —— 判错时能一眼看出走的是严口径还是宽口径。"""
    fake_chat(_reply(_all(), claims=[]))
    assert judge_task(_task_of_type("多跳"), "答了")["task_type"] == "多跳"


# --------------------------------------------------------------------------------------
# ⑥ F 空答案 = 不适用（不是满分）
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("empty", [None, "", "   ", "\n\t ", [], {}])
def test_f_is_not_applicable_for_an_empty_answer(fake_chat, empty):
    """空答案 → F 为 None，**绝不是 1.0**。"""
    fake_chat(_reply(dict.fromkeys(ALL_IDS, 0.0), claims=[]))
    r = judge_task(_task(), empty)
    assert r["F"] is None
    assert r["F"] != 1.0
    assert "F 不适用" in r["reason"]


def test_a_full_marks_answer_with_empty_claims_is_not_a_free_pass(fake_chat):
    """rubric 全对但 F 无从判定 ⇒ 主榜仍不认成功：不能「只答 rubric 里的点」绕开忠实度。"""
    fake_chat(_reply(_all(), claims=[]))
    r = judge_task(_task(), "答了")
    assert r["G"] == pytest.approx(1.0)
    assert r["F"] is None
    assert r["correct"] is False


def test_an_empty_answer_never_scores_correct(fake_chat):
    fake_chat(_reply(dict.fromkeys(ALL_IDS, 0.0), claims=[]))
    assert judge_task(_task(), None)["correct"] is False


# --------------------------------------------------------------------------------------
# ⑦ T：从轨迹机械判
# --------------------------------------------------------------------------------------


def test_t_is_the_per_call_legality_rate(fake_chat):
    fake_chat(_reply(_all()))
    traj = _traj(_call(hits=(DOC_A,)), _call(arguments={"query": ""}, illegal=True))
    r = judge_task(_task(), "答了", trajectory=traj)
    assert r["T"] == pytest.approx(0.5)
    assert r["diagnostics"] == {"n_calls": 2, "n_illegal_calls": 1}
    assert r["correct"] is False, "工具协议不合规 ⇒ 主榜不算成功"


def test_t_is_one_when_no_call_is_illegal(fake_chat):
    fake_chat(_reply(_all()))
    traj = _traj(_call(hits=(DOC_A,)), _call(hits=(DOC_B,)))
    assert judge_task(_task(), "答了", trajectory=traj)["T"] == pytest.approx(1.0)


@pytest.mark.parametrize("traj", [None, [], (), "不是列表", {}, 42])
def test_t_is_not_applicable_without_a_trajectory(fake_chat, traj):
    """没调用工具 ≠ 协议违规 ⇒ T 不适用，既不自动满分也不自动 0。"""
    fake_chat(_reply(_all()))
    assert judge_task(_task(), "答了", trajectory=traj)["T"] is None


def test_t_counts_an_entry_with_an_error_key_as_illegal(fake_chat):
    fake_chat(_reply(_all()))
    traj = _traj(_call(illegal=True), _call(illegal=True), _call(hits=(DOC_A,)))
    assert judge_task(_task(), "答了", trajectory=traj)["T"] == pytest.approx(1 / 3)


def test_t_is_mechanical_and_does_not_consult_the_llm(fake_chat):
    """T 与 judge 无关：judge 打满分也救不了非法的工具调用。"""
    fake_chat(_reply(_all()))
    r = judge_task(_task(), "答了", trajectory=_traj(_call(illegal=True)))
    assert r["G"] == pytest.approx(1.0)
    assert r["T"] == 0.0
    assert r["correct"] is False


# --------------------------------------------------------------------------------------
# ⑧ 关键证据覆盖率（独立诊断，不混进 T）
# --------------------------------------------------------------------------------------


def test_evidence_coverage_is_the_share_of_required_docs_retrieved(fake_chat):
    """rubric 要 DOC_A + DOC_B，只检索到 DOC_A ⇒ 0.5。"""
    fake_chat(_reply(_all()))
    r = judge_task(_task(), "答了", trajectory=_traj(_call(hits=(DOC_A,))))
    assert r["evidence_coverage"] == pytest.approx(0.5)


def test_evidence_coverage_is_one_when_all_required_docs_were_retrieved(fake_chat):
    fake_chat(_reply(_all()))
    r = judge_task(_task(), "答了", trajectory=_traj(_call(hits=(DOC_A,)), _call(hits=(DOC_B,))))
    assert r["evidence_coverage"] == pytest.approx(1.0)


def test_evidence_coverage_is_zero_when_nothing_was_retrieved(fake_chat):
    fake_chat(_reply(_all()))
    r = judge_task(_task(), "答了", trajectory=_traj(_call(hits=())))
    assert r["evidence_coverage"] == 0.0


def test_evidence_coverage_ignores_irretrieved_documents(fake_chat):
    """命中一篇 rubric 没要的文档，不该算进覆盖率分子。"""
    fake_chat(_reply(_all()))
    r = judge_task(_task(), "答了", trajectory=_traj(_call(hits=(DOC_A, "kb-无关-999"))))
    assert r["evidence_coverage"] == pytest.approx(0.5)


def test_evidence_coverage_is_not_applicable_when_no_point_cites_evidence(fake_chat):
    rubric = [_point(cid, 0.25, evidence=()) for cid in ALL_IDS]
    fake_chat(_reply(_all()))
    r = judge_task(_task(rubric), "答了", trajectory=_traj(_call(hits=(DOC_A,))))
    assert r["evidence_coverage"] is None


def test_evidence_coverage_is_independent_of_t(fake_chat):
    """一次非法调用 + 一次合法检索拿全证据 ⇒ T=0.5、覆盖率 1.0，**两个数各自独立**。"""
    fake_chat(_reply(_all()))
    traj = _traj(_call(illegal=True), _call(hits=(DOC_A, DOC_B)))
    r = judge_task(_task(), "答了", trajectory=traj)
    assert r["T"] == pytest.approx(0.5)
    assert r["evidence_coverage"] == pytest.approx(1.0)


def test_an_illegal_call_returns_no_hits_to_count_as_retrieved(fake_chat):
    """非法调用时工具吐的是 error 哨兵、没有 hits —— 检索到 0 篇，不是「错误也算命中」。"""
    fake_chat(_reply(_all()))
    r = judge_task(_task(), "答了", trajectory=_traj(_call(illegal=True)))
    assert r["evidence_coverage"] == 0.0


def test_evidence_coverage_survives_a_broken_trajectory(fake_chat):
    """轨迹里没有 result / hits 不是列表 ⇒ 不崩，覆盖率算 0。"""
    fake_chat(_reply(_all()))
    traj = [{"tool": "x", "arguments": {}, "result": {"hits": "不是列表"}}, "垃圾", {"result": None}]
    r = judge_task(_task(), "答了", trajectory=traj)
    assert r["evidence_coverage"] == 0.0
    assert r["T"] == pytest.approx(1.0)


def test_prompt_marks_the_critical_point(fake_chat):
    """关键点必须在 prompt 里标出来 —— 封顶是**程序**算的，但 judge 得知道哪个是关键点。

    只给「权重 0.3」的话，模型没有任何理由对这个点比对别个点更严格；而丢关键点是主榜
    一票否决的量级。这条钉的是提示语本身进没进 prompt：撤掉它，``_collect_scores`` 照样
    正常收分，靠别的用例抓不到（反向验证 R9 实测过一轮才补上）。
    """
    scripted = fake_chat(_reply(_all()))
    judge_task(_task(), "答了")
    user = scripted.calls[0]["messages"][-1]["content"]
    assert "关键点" in user


# --------------------------------------------------------------------------------------
# prompt 渲染（判据 3）
# --------------------------------------------------------------------------------------


def test_prompt_renders_every_rubric_point_as_its_own_section(fake_chat):
    scripted = fake_chat(_reply(_all()))
    judge_task(_task(), "答了")
    user = scripted.calls[0]["messages"][-1]["content"]
    for cid in ALL_IDS:
        assert f"criterion_id = {cid}" in user
    assert user.count("### 评分点") == len(ALL_IDS), "每条 rubric 必须独立成段"


def test_prompt_carries_the_full_rule_and_the_evidence_source_text(fake_chat):
    scripted = fake_chat(_reply(_all()))
    judge_task(_task(), "答了")
    user = scripted.calls[0]["messages"][-1]["content"]
    assert "半分规则：" in user
    assert "等价说法" in user  # accepted_alternatives
    assert "证据原文" in user
    for doc in DOCS:
        assert doc["doc_id"] in user
        assert doc["text"] in user, "证据原文不许省略 —— judge 只能凭原文判忠实"


def test_prompt_never_asks_for_a_total_score(fake_chat):
    """判据 3 的硬要求：逐条返回，**不许整段塞「给个总分」**。"""
    scripted = fake_chat(_reply(_all()))
    judge_task(_task(), "答了")
    system = scripted.calls[0]["messages"][0]["content"]
    assert "不要给总分" in system
    assert '"scores"' in scripted.calls[0]["messages"][-1]["content"]


def test_prompt_does_not_leak_the_example_answer(fake_chat):
    """示例答案是给人看的；送进 prompt 就是 P0-1 点名的「复述即满分」。"""
    scripted = fake_chat(_reply(_all()))
    judge_task(_task(), "答了")
    whole = json.dumps(scripted.calls[0]["messages"], ensure_ascii=False)
    assert "这是示例，不是判分依据" not in whole


def test_prompt_marks_the_empty_answer_explicitly(fake_chat):
    scripted = fake_chat(_reply(dict.fromkeys(ALL_IDS, 0.0)))
    judge_task(_task(), None)
    user = scripted.calls[0]["messages"][-1]["content"]
    assert "（空）" in user


def test_prompt_notes_a_missing_evidence_document_instead_of_inventing_it(fake_chat):
    """知识库里缺 DOC_B 时 prompt 写明「未提供」，**不许把别的文档原文顶上**。"""
    scripted = fake_chat(_reply(dict.fromkeys(ALL_IDS, 0.0)))
    # validate=False：这条 rubric 引了库里没有的 DOC_B，本就是坏数据（生成期会被拒），
    # 这里要验的正是 judge 碰到坏数据时的**降级行为**，不是数据合不合法。
    judge_task(_task(docs=[DOCS[0]], validate=False), "x")
    user = scripted.calls[0]["messages"][-1]["content"]
    assert f"[{DOC_B}]（本任务未提供该文档原文）" in user
    assert DOCS[0]["text"] in user, "能给的原文仍要给"


def test_prompt_says_so_when_a_point_cites_no_evidence(fake_chat):
    rubric = [_point(cid, 0.25, evidence=() if cid != C4 else (DOC_B,)) for cid in ALL_IDS]
    scripted = fake_chat(_reply(_all()))
    judge_task(_task(rubric), "x")
    assert "本评分点不依赖具体文档" in scripted.calls[0]["messages"][-1]["content"]


def test_prompt_is_deterministic(fake_chat):
    """同输入两次渲染逐字节一致（确定性铁律）。"""
    task, answer = _task(), "答了"
    assert build_judge_prompt(task, answer) == build_judge_prompt(task, answer)


def test_prompt_has_no_literal_newline_escape(fake_chat):
    """prompt 是发给模型的正文，不许混进 ``\\n`` 这种「源码里的转义」——
    模型会当成两个字符读进去，章节标题就断行了。"""
    scripted = fake_chat(_reply(_all()))
    judge_task(_task(), "答了")
    whole = "".join(m["content"] for m in scripted.calls[0]["messages"])
    assert "\\n" not in whole


def test_prompt_renders_a_non_string_answer_as_json(fake_chat):
    scripted = fake_chat(_reply(dict.fromkeys(ALL_IDS, 0.0)))
    judge_task(_task(), {"文本": "答了", "引用": [DOC_A]})
    user = scripted.calls[0]["messages"][-1]["content"]
    assert "答了" in user and DOC_A in user


def test_judge_requests_zero_temperature_and_records_the_model(fake_chat):
    scripted = fake_chat(_reply(_all()))
    r = judge_task(_task(), "答了")
    assert scripted.calls[0]["temperature"] == judge_mod.JUDGE_TEMPERATURE
    assert scripted.calls[0]["temperature"] == 0.0
    assert r["model"] == judge_mod.JUDGE_MODEL, "judge 的 model id 必须记下来备归因"


# --------------------------------------------------------------------------------------
# 结构完整性 / 健壮性
# --------------------------------------------------------------------------------------


def test_the_result_carries_all_three_metrics_even_on_a_clean_run(fake_chat):
    fake_chat(_reply(
        _all(),
        claims=[{"claim": "默认经济舱", "verdict": SUPPORTED, "evidence_id": DOC_A}],
    ))
    r = judge_task(_task(), "答了")
    for key in ("correct", "G", "F", "T", "scores", "capped", "reason", "claims",
                "evidence_coverage", "model", "n_contradicted", "n_unsupported"):
        assert key in r, f"§2.5 要求返回 {key}"
    assert json.dumps(r, ensure_ascii=False), "结果必须可序列化"


def test_a_task_without_a_rubric_degrades_instead_of_raising(fake_chat):
    fake_chat(_reply())
    task = _task()
    bare = Task(
        id="t-0002", category=Category.RETRIEVAL, schema_version="1.1",
        instruction="x", initial_data={}, tool_names=(),
        ground_truth={},
        answer_spec=task.answer_spec, split=Split.TRAIN, rubric=None,
    )
    r = judge_task(bare, "答了")
    assert r["G"] == 0.0
    assert r["reason"]


def test_a_broken_rubric_degrades_instead_of_raising(fake_chat):
    """rubric 自己不合法（键不齐）⇒ aggregate_rubric 会抛 —— 判分器把它降级成原因。"""
    fake_chat(_reply(_all()))
    task = _task()
    broken = Task(
        id="t-0003", category=Category.RETRIEVAL, schema_version="1.1",
        instruction="x", initial_data={"knowledge": DOCS}, tool_names=(),
        ground_truth={}, answer_spec=task.answer_spec, split=Split.TRAIN,
        rubric=({"criterion_id": "t-0003-c1", "weight": 2.0},),  # 权重不归一、无 requirement
    )
    r = judge_task(broken, "答了")
    assert r["G"] == 0.0
    assert "rubric 聚合失败" in r["reason"]


def test_the_whole_judge_is_deterministic_under_a_scripted_model(fake_chat):
    """同输入 + 同一脚本 ⇒ 两次结果完全相等（分数可复现是 M1 的核心承诺）。"""
    replies = [_reply(_all(C2=0.0),
                      claims=[{"claim": "x", "verdict": CONTRADICTED, "evidence_id": ""}])]
    scripted = fake_chat(*replies)
    task, traj = _task(), _traj(_call(hits=(DOC_A,)), _call(illegal=True))
    first = judge_task(task, "答了", trajectory=traj)
    scripted.replies = list(replies)
    assert judge_task(task, "答了", trajectory=traj) == first


def test_judge_is_reachable_through_the_a_scene_entry_point_too(fake_chat):
    """经 :func:`judge` 直接调（harness 的走法）与 :func:`judge_task` 结果一致。"""
    scripted = fake_chat(_reply(_all()))
    task = _task()
    direct = judge("答了", task.ground_truth, task.answer_spec,
                   task=task, trajectory=_traj(_call(hits=(DOC_A,))))
    scripted.replies = [_reply(_all())]
    assert direct == judge_task(task, "答了", trajectory=_traj(_call(hits=(DOC_A,))))


def test_judge_module_still_imports_only_stdlib_plus_local():
    """运行时零第三方依赖是 M1 的承诺，本模块不能顺手引进一个包。"""
    assert judge_mod.__doc__
    import inspect
    tree = inspect.getsource(judge_mod)
    for banned in ("import requests", "import numpy", "import pandas", "import openai"):
        assert banned not in tree
