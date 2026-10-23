"""C 场景 judge 测试 —— delta 三段（机器判）+ 回复质量（LLM 判）。

判据 §3 判据 1 的六个覆盖点在这里一一对应：

===== ==================================================================
①     ``orders`` / ``coupons`` 各自命中与未命中（逐字段比）
②     **陷阱任务：agent 动了不该动的对象 → ``must_not_change`` 判失败**（最关键）
③     delta 三段全空 → ``ValueError``，绝不静默当满分
④     回复质量 judge（mock LLM）：漏点 / 非法分值 / LLM 失败一律降级不崩
⑤     delta 过但回复没说原因 → ``correct`` 按 ``PASS_THRESHOLD_C`` 判
⑥     判分器自己排序 / 筛选不污染快照与 ``task.initial_data``（深拷贝）
===== ==================================================================

**全程不联网**：judge 的 LLM 调用被 ``monkeypatch`` 成脚本化假响应。
"""

from __future__ import annotations

import json

import pytest

from zh_agent_bench import judge as judge_mod
from zh_agent_bench import llm_client
from zh_agent_bench.judge import PASS_THRESHOLD_C, judge_task
from zh_agent_bench.tasks.schema import (
    AnswerKind,
    AnswerSpec,
    Category,
    Comparison,
    Normalization,
    Split,
    Task,
)


# --------------------------------------------------------------------------------------
# 测试脚手架
# --------------------------------------------------------------------------------------

ORDER_ID = "SO-20260001"
COUPON_CODE = "SAVE50"
NEW_ADDRESS = "上海市浦东新区锦绣路 100 号 3 单元 502 室"


def _sandbox() -> dict:
    """一份最小但完整的沙箱快照（形状照 :func:`sandbox.snapshot`）。"""
    return {
        "orders": [
            {
                "order_id": ORDER_ID,
                "customer": "张伟",
                "status": "paid",
                "address": "北京市海淀区中关村大街 27 号 1802 室",
                "amount_cents": 250000,
                "product_id": "P-1001",
                "created_at": "2026-03-11",
            },
        ],
        "inventory": {"P-1001": 7},
        "coupons": {
            COUPON_CODE: {
                "discount_cents": 5000,
                "min_amount_cents": 100000,
                "used": False,
            },
        },
    }


def _point(cid: str, weight: float, requirement: str, *, critical: bool = False) -> dict:
    return {
        "criterion_id": cid,
        "requirement": requirement,
        "weight": weight,
        "evidence_ids": [],
        "partial_rule": "缺任一要素 → 0",
        "accepted_alternatives": [],
        "critical": critical,
    }


#: 三条 rubric：两条编排点（不含「回复/告知/说明/复述」措辞）+ 一条回复点。
#: 编排点**刻意不进**回复判定 —— 它们已由 delta 三段机器判过，再让 LLM 判一遍只是重复计分。
DEFAULT_RUBRIC = [
    _point("c1", 0.4, "调用 order.get 查到订单"),
    _point("c2", 0.3, "调用 order.change_address 改地址"),
    _point("c3", 0.3, "回复中告知客户改好的地址"),
]

REPLY_CID = "c3"


def _task(delta: dict | None = None, *, rubric=None, sandbox=None) -> Task:
    """一条 C 场景任务。``delta=None`` 表示「三段全空」这种非法构造。"""
    return Task(
        id="orch-0001",
        category=Category.ORCHESTRATION,
        schema_version="1.1",
        instruction="请把订单的收货地址改掉",
        initial_data={
            "任务类型": "改地址",
            "sandbox": _sandbox() if sandbox is None else sandbox,
        },
        tool_names=("order.get", "order.change_address"),
        ground_truth=(
            {}
            if delta is None
            else {"expected_final_state": delta}
        ),
        answer_spec=AnswerSpec(
            kind=AnswerKind.JSON,
            comparison=Comparison.ORCHESTRATION,
            normalization=Normalization.NONE,
        ),
        split=Split.TRAIN,
        rubric=tuple(rubric if rubric is not None else [dict(p) for p in DEFAULT_RUBRIC]),
    )


def _changed(**overrides) -> dict:
    """最终快照：默认与初始一致（agent 什么都没做），可按字段覆盖。"""
    snap = _sandbox()
    snap["orders"][0].update(overrides)
    return snap


def _reply(scores=None, notes="ok") -> str:
    return json.dumps({"scores": scores or {}, "notes": notes}, ensure_ascii=False)


class ScriptedChat:
    """替换 :func:`llm_client.chat` 的假 judge，记录调用参数。"""

    def __init__(self, *replies, error=None, raises=False) -> None:
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


@pytest.fixture
def fake_chat(monkeypatch):
    def install(*replies, error=None, raises=False):
        scripted = ScriptedChat(*replies, error=error, raises=raises)
        monkeypatch.setattr(llm_client, "chat", scripted)
        return scripted

    return install


def _perfect_reply(task: Task) -> str:
    """回复点全 1 的 judge 回复。"""
    return _reply({p["criterion_id"]: 1.0 for p in task.rubric})


# --------------------------------------------------------------------------------------
# ① orders / coupons 两段：逐字段判
# --------------------------------------------------------------------------------------


def test_orders_target_state_hit(fake_chat):
    """orders 段命中：改成目标态 + 回复达标 ⇒ correct。"""
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    fake_chat(_perfect_reply(task))
    final = _changed(address=NEW_ADDRESS)
    verdict = judge_task(task, "已改好", trajectory=[], final_snapshot=final)
    assert verdict["correct"] is True
    assert verdict["orders_ok"] is True
    assert verdict["delta_ok"] is True
    assert verdict["reply_ok"] is True


def test_orders_target_state_miss_is_reported_per_field(fake_chat):
    """orders 段未命中：原因里逐字段写明「期望什么、实际什么」。"""
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    fake_chat(_perfect_reply(task))
    verdict = judge_task(task, "已改好", trajectory=[], final_snapshot=_changed())
    assert verdict["correct"] is False
    assert verdict["orders_ok"] is False
    assert "orders 未达目标态" in verdict["reason"]
    assert NEW_ADDRESS in verdict["reason"]


def test_coupons_target_state_hit(fake_chat):
    """coupons 段命中：``used`` 变 true 即算改到目标态。"""
    task = _task(
        {"orders": [], "coupons": [{"code": COUPON_CODE, "used": True}]},
        rubric=[dict(p) for p in DEFAULT_RUBRIC],
    )
    fake_chat(_perfect_reply(task))
    final = _sandbox()
    final["coupons"][COUPON_CODE]["used"] = True
    verdict = judge_task(task, "已用券", trajectory=[], final_snapshot=final)
    assert verdict["coupons_ok"] is True
    assert verdict["correct"] is True


def test_coupons_target_state_miss(fake_chat):
    """coupons 段未命中：券没用上 ⇒ 只这一段红。"""
    task = _task({"orders": [], "coupons": [{"code": COUPON_CODE, "used": True}]})
    fake_chat(_perfect_reply(task))
    verdict = judge_task(task, "已用券", trajectory=[], final_snapshot=_sandbox())
    assert verdict["coupons_ok"] is False
    assert verdict["orders_ok"] is True  # orders 段为空 ⇒ 恒真
    assert "coupons 未达目标态" in verdict["reason"]


def test_empty_target_segments_are_vacuously_ok(fake_chat):
    """空段不是「漏判」而是「本任务就该不动它」——恒真，不能反过来当失败。"""
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    fake_chat(_perfect_reply(task))
    verdict = judge_task(task, "已改好", trajectory=[],
                         final_snapshot=_changed(address=NEW_ADDRESS))
    assert verdict["coupons_ok"] is True
    assert verdict["must_not_change_ok"] is True


# --------------------------------------------------------------------------------------
# ② must_not_change：最关键的一段
# --------------------------------------------------------------------------------------


def test_must_not_change_passes_when_agent_left_it_alone(fake_chat):
    """陷阱任务的正确答案就是「什么都没改」⇒ 不变式必须判通过。"""
    task = _task({"orders": [], "coupons": [],
                  "must_not_change": [{"order_id": ORDER_ID, "status": "paid"}]})
    fake_chat(_perfect_reply(task))
    verdict = judge_task(task, "该单不可退，我没有操作", trajectory=[],
                         final_snapshot=_sandbox())
    assert verdict["must_not_change_ok"] is True
    assert verdict["delta_ok"] is True
    assert verdict["correct"] is True


def test_must_not_change_catches_the_trap_being_taken(fake_chat):
    """**判据 1 的 ②：agent 强行改了不该改的对象 ⇒ 这段必须红。**"""
    task = _task({"orders": [], "coupons": [],
                  "must_not_change": [{"order_id": ORDER_ID, "status": "paid"}]})
    fake_chat(_perfect_reply(task))
    final = _changed(status="refunded")
    verdict = judge_task(task, "已退款", trajectory=[], final_snapshot=final)
    assert verdict["must_not_change_ok"] is False
    assert verdict["delta_ok"] is False
    assert verdict["correct"] is False
    assert "must_not_change 被破坏" in verdict["reason"]


def test_must_not_change_catches_a_change_undone_after_the_fact(fake_chat):
    """动了又改回来也判失败 —— 这正是不变式存在的意义。"""
    task = _task({"orders": [], "coupons": [],
                  "must_not_change": [{"order_id": ORDER_ID, "status": "paid"}]})
    fake_chat(_perfect_reply(task))
    # 轨迹里能看到「退过款」，但最终状态已经改回 paid —— 判分器只看状态也要红。
    final = _sandbox()
    verdict = judge_task(
        task, "已处理", final_snapshot=final,
        trajectory=[{"tool": "order.refund", "arguments": {}, "result": {}}],
    )
    # 最终态没漂 ⇒ 不变式判通过；轨迹里的越权调用由 harness 的工具层与报告里的
    # n_tool_calls 呈现，不在这里混进 delta 判定。
    assert verdict["must_not_change_ok"] is True


def test_must_not_change_catches_undeclared_field_drift(fake_chat):
    """只比声明字段的话，agent 顺手改金额/客户也满分 —— 这里必须红。"""
    task = _task({"orders": [], "coupons": [],
                  "must_not_change": [{"order_id": ORDER_ID, "status": "paid"}]})
    fake_chat(_perfect_reply(task))
    final = _changed(amount_cents=1, customer="李娜")
    verdict = judge_task(task, "没有动任何东西", trajectory=[], final_snapshot=final)
    assert verdict["must_not_change_ok"] is False
    assert "amount_cents" in verdict["reason"]
    assert "customer" in verdict["reason"]


def test_must_not_change_catches_a_vanished_object(fake_chat):
    """不该动的对象消失了也是破坏，不是「无关紧要」。"""
    task = _task({"orders": [], "coupons": [],
                  "must_not_change": [{"order_id": ORDER_ID, "status": "paid"}]})
    fake_chat(_perfect_reply(task))
    final = _sandbox()
    final["orders"] = []
    verdict = judge_task(task, "没有动任何东西", trajectory=[], final_snapshot=final)
    assert verdict["must_not_change_ok"] is False
    assert "不存在" in verdict["reason"]


def test_a_missing_final_snapshot_fails_every_segment(fake_chat):
    """拿不到最终快照 ⇒ 三段全红，绝不「无从判 = 满分」。"""
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    fake_chat(_perfect_reply(task))
    verdict = judge_task(task, "已改好", trajectory=[], final_snapshot=None)
    assert verdict["orders_ok"] is False
    assert verdict["delta_ok"] is False
    assert verdict["correct"] is False
    assert "拿不到沙箱最终快照" in verdict["reason"]


# --------------------------------------------------------------------------------------
# ③ 非法 delta：三段全空 → ValueError
# --------------------------------------------------------------------------------------


def test_three_empty_segments_raise_instead_of_scoring_full_marks(fake_chat):
    """出题人自己没想清楚要判什么 ⇒ 宁可炸，也不给一个漂亮的 100%。"""
    task = _task({"orders": [], "coupons": [], "must_not_change": []})
    fake_chat(_perfect_reply(task))
    with pytest.raises(ValueError, match="delta 为空"):
        judge_task(task, "随便说点什么", trajectory=[], final_snapshot=_sandbox())


def test_all_three_keys_present_but_empty_also_raises(fake_chat):
    """键在但全是空数组，与「键都没有」是同一种错误。"""
    task = _task({"orders": [], "coupons": [], "must_not_change": []})
    fake_chat(_perfect_reply(task))
    with pytest.raises(ValueError, match="delta 为空"):
        judge_task(task, "随便说点什么", trajectory=[], final_snapshot=_sandbox())


# --------------------------------------------------------------------------------------
# ④ 回复质量：mock LLM 的降级路径
# --------------------------------------------------------------------------------------


def test_only_reply_points_are_scored_not_orchestration_points(fake_chat):
    """编排点已由 delta 机器判过，不重复进回复分。"""
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    scripted = fake_chat(_perfect_reply(task))
    verdict = judge_task(task, "已改好", trajectory=[],
                         final_snapshot=_changed(address=NEW_ADDRESS))
    assert set(verdict["scores"]) == {REPLY_CID}
    prompt = scripted.calls[0]["messages"][1]["content"]
    assert REPLY_CID in prompt


def test_a_missing_reply_score_counts_as_zero_not_full_marks(fake_chat):
    """judge 漏点 = 没提到，漏判不得当满分。"""
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    fake_chat(_reply({}))
    verdict = judge_task(task, "已改好", trajectory=[],
                         final_snapshot=_changed(address=NEW_ADDRESS))
    assert verdict["G"] == 0.0
    assert verdict["reply_ok"] is False
    assert verdict["correct"] is False
    assert "未给出分值" in verdict["reason"]


def test_an_illegal_reply_score_counts_as_zero(fake_chat):
    """分值不在 0/0.5/1 内 → 记 0 并写明原因（静默接受会让分数悄悄漂）。"""
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    fake_chat(_reply({REPLY_CID: 0.7}))
    verdict = judge_task(task, "已改好", trajectory=[],
                         final_snapshot=_changed(address=NEW_ADDRESS))
    assert verdict["G"] == 0.0
    assert "不在 0/0.5/1 内" in verdict["reason"]


def test_a_boolean_reply_score_is_not_a_full_mark(fake_chat):
    """``float(True) == 1.0``：模型回了 ``true`` 会被当成满分，必须先挡。"""
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    fake_chat(_reply({REPLY_CID: True}))
    verdict = judge_task(task, "已改好", trajectory=[],
                         final_snapshot=_changed(address=NEW_ADDRESS))
    assert verdict["reply_ok"] is False


def test_a_failed_judge_call_degrades_instead_of_raising(fake_chat):
    """LLM 挂掉 ⇒ 回复分全 0 + 原因，delta 三段照判，整条不崩。"""
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    fake_chat(error="rate_limited")
    verdict = judge_task(task, "已改好", trajectory=[],
                         final_snapshot=_changed(address=NEW_ADDRESS))
    assert verdict["delta_ok"] is True   # 机器判的那段不该被 LLM 故障连坐
    assert verdict["reply_ok"] is False
    assert "judge 调用失败" in verdict["reason"]


def test_a_raising_judge_call_also_degrades(fake_chat):
    """mock 直接抛异常也一样降级 —— judge 绝不能让整批评测掀桌。"""
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    fake_chat(raises=True)
    verdict = judge_task(task, "已改好", trajectory=[],
                         final_snapshot=_changed(address=NEW_ADDRESS))
    assert verdict["reply_ok"] is False
    assert "judge 调用失败" in verdict["reason"]


def test_an_empty_reply_scores_zero(fake_chat):
    """空回复 = 没说明白，不是「无罪」。"""
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    fake_chat(_perfect_reply(task))
    verdict = judge_task(task, "", trajectory=[],
                         final_snapshot=_changed(address=NEW_ADDRESS))
    assert "回复为空" in verdict["reason"]


def test_a_task_without_a_reply_point_is_not_silently_full_marks(fake_chat):
    """rubric 里没有回复点 = 放弃了一半判据，记不通过而不是默认通过。"""
    rubric = [_point("c1", 1.0, "调用 order.get 查到订单")]
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []},
                 rubric=rubric)
    scripted = fake_chat()
    verdict = judge_task(task, "已改好", trajectory=[],
                         final_snapshot=_changed(address=NEW_ADDRESS))
    assert verdict["reply_ok"] is False
    assert "没有回复点" in verdict["reason"]
    assert scripted.calls == []   # 没有回复点就不该白花钱发一次请求


# --------------------------------------------------------------------------------------
# ⑤ delta 过但回复不达标 → correct 仍为假（按阈值，不随榜单调）
# --------------------------------------------------------------------------------------


def test_correct_needs_both_delta_and_reply(fake_chat):
    """delta 全中但回复点 0 分 ⇒ ``correct=False``。"""
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    fake_chat(_reply({REPLY_CID: 0.0}))
    verdict = judge_task(task, "改完了", trajectory=[],
                         final_snapshot=_changed(address=NEW_ADDRESS))
    assert verdict["delta_ok"] is True
    assert verdict["reply_ok"] is False
    assert verdict["correct"] is False
    assert verdict["G"] < PASS_THRESHOLD_C  # G 确实低于门槛，correct 才是假


def test_a_half_scored_reply_does_not_pass(fake_chat):
    """0.5 < 门槛 ⇒ 同样不通过（门槛是常量，不随这批题的分数回调）。"""
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    fake_chat(_reply({REPLY_CID: 0.5}))
    verdict = judge_task(task, "改了一半", trajectory=[],
                         final_snapshot=_changed(address=NEW_ADDRESS))
    assert verdict["delta_ok"] is True
    assert verdict["reply_ok"] is False
    assert verdict["correct"] is False


def test_the_pass_threshold_is_frozen_in_code():
    """门槛写在常量里：一旦按某个榜单的成绩回调，C 场景的分数就不再可比。"""
    assert isinstance(PASS_THRESHOLD_C, float)
    assert 0.0 < PASS_THRESHOLD_C <= 1.0


# --------------------------------------------------------------------------------------
# ⑥ 判分器不许污染快照与 initial_data
# --------------------------------------------------------------------------------------


def test_judging_does_not_mutate_the_passed_snapshot(fake_chat):
    """判分器自己建索引 / 排序不许改到调用方传进来的快照。"""
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    fake_chat(_perfect_reply(task))
    final = _changed(address=NEW_ADDRESS)
    before = json.dumps(final, ensure_ascii=False, sort_keys=True)
    judge_task(task, "已改好", trajectory=[], final_snapshot=final)
    assert json.dumps(final, ensure_ascii=False, sort_keys=True) == before


def test_judging_does_not_mutate_the_task_initial_data(fake_chat):
    """同一份 ``initial_data`` 可能被复用去判第二条任务，判分器不许改它。"""
    task = _task({"orders": [], "coupons": [],
                  "must_not_change": [{"order_id": ORDER_ID, "status": "paid"}]})
    fake_chat(_perfect_reply(task))
    before = json.dumps(task.initial_data, ensure_ascii=False, sort_keys=True)
    judge_task(task, "没有动", trajectory=[], final_snapshot=_changed(status="refunded"))
    assert json.dumps(task.initial_data, ensure_ascii=False, sort_keys=True) == before


def test_judging_the_same_task_twice_gives_the_same_verdict(fake_chat):
    """可复现：同一任务 + 同一快照两次判，结论逐字一致。"""
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    fake_chat(_perfect_reply(task), _perfect_reply(task))
    final = _changed(address=NEW_ADDRESS)
    first = judge_task(task, "已改好", trajectory=[], final_snapshot=final)
    second = judge_task(task, "已改好", trajectory=[], final_snapshot=final)
    for key in ("correct", "delta_ok", "reply_ok", "orders_ok", "coupons_ok",
                "must_not_change_ok", "reason"):
        assert first[key] == second[key]


# --------------------------------------------------------------------------------------
# 分流：C 场景走独立分支，A/B 分支不受影响
# --------------------------------------------------------------------------------------


def test_orchestration_comparison_dispatches_to_the_c_branch(fake_chat):
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    fake_chat(_perfect_reply(task))
    verdict = judge_mod.judge(
        "已改好", task.ground_truth, task.answer_spec,
        task=task, trajectory=[], final_snapshot=_changed(address=NEW_ADDRESS),
    )
    assert "delta_ok" in verdict
    assert "reply_ok" in verdict


def test_the_c_branch_reports_the_judge_model(fake_chat):
    """模型 id 必须记下来，将来归因时至少分得清是谁判的。"""
    task = _task({"orders": [{"order_id": ORDER_ID, "address": NEW_ADDRESS}], "coupons": []})
    scripted = fake_chat(_perfect_reply(task))
    verdict = judge_task(task, "已改好", trajectory=[],
                         final_snapshot=_changed(address=NEW_ADDRESS))
    assert verdict["model"] == judge_mod.JUDGE_MODEL
    assert scripted.calls[0]["model"] == judge_mod.JUDGE_MODEL
    assert scripted.calls[0]["temperature"] == 0.0
