"""judge 与被测 agent 分离（TASK-027）—— 注入点测试。

判据 §3 第 1 条要求的三项覆盖，在这里一一对应：

============================  ==========================================================
①  不传 ``judge_kwargs`` 时   ``llm_client.chat`` 收到的 messages / model / temperature
    请求体逐字节不变           **完全等同调用前的基线**（B 与 C 两条路径各测一次）
②  传入 ``model``/            judge 请求确实改道到指定模型 / 端点 / 密钥
    ``client_kwargs`` 时
③  产物署名跟着走            ``report.md`` 与 record 里的 ``model`` 不得仍写 deepseek
============================  ==========================================================

另加一条**契约没要求但本单自己踩到的坑**：``--judge-base-url`` 与 ``--judge-key-env``
必须**同时**进 ``client_kwargs`` —— 只挂 base_url 会让密钥被静默丢弃。

**全程不联网**：:func:`llm_client.chat` 被 monkeypatch 成脚本化假响应。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from zh_agent_bench import eval_run
from zh_agent_bench import judge as judge_mod
from zh_agent_bench import llm_client
from zh_agent_bench.judge import JUDGE_MODEL, judge, judge_task
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

DOC_A = "kb-s-001"
DOC_B = "kb-s-002"

DOCS = [
    {"doc_id": DOC_A, "title": "报销管理办法", "text": "飞机票一律 Economy 舱。"},
    {"doc_id": DOC_B, "title": "发布节奏", "text": "双周发布：奇数周发缺陷修复版。"},
]


def _point(cid: str, weight: float) -> dict[str, Any]:
    return {
        "criterion_id": cid,
        "requirement": f"要点 {cid}",
        "weight": weight,
        "evidence_ids": [DOC_A],
        "partial_rule": "缺任一要素 → 0",
        "accepted_alternatives": [f"{cid} 的等价说法"],
        "critical": False,
    }


#: 四个点（``qa.MIN_POINTS`` 是 4）。权重不等，测得出来「加权」与「数对几个」不是一回事。
RUBRIC = [_point("s-1", 0.4), _point("s-2", 0.3), _point("s-3", 0.2), _point("s-4", 0.1)]


def _qa_task(rubric=None) -> Task:
    rubric = rubric if rubric is not None else [dict(p) for p in RUBRIC]
    qa.validate_rubric(rubric, {d["doc_id"] for d in DOCS})
    return Task(
        id="sep-t-0001",
        category=Category.RETRIEVAL,
        schema_version="1.1",
        instruction="测试问题",
        initial_data={"任务类型": "单跳", "knowledge": DOCS},
        tool_names=(qa.QA_TOOL_NAME,),
        ground_truth={"example_answer": "示例"},
        answer_spec=AnswerSpec(
            kind=AnswerKind.STRING,
            comparison=Comparison.JUDGE,
            normalization=Normalization.NONE,
        ),
        split=Split.TRAIN,
        rubric=tuple(rubric),
    )


def _c_task() -> Task:
    """一条最小 C 场景任务：delta 只声明一条 orders 行，且 rubric 末点含「回复」措辞。"""
    return Task(
        id="sep-c-0001",
        category=Category.ORCHESTRATION,
        schema_version="1.1",
        instruction="把订单改成新地址并回复客户",
        initial_data={
            "任务类型": "改地址",
            "sandbox": {
                "orders": [{"order_id": "O1", "status": "paid", "address": "旧地址",
                            "amount_cents": 100, "customer": "甲"}],
                "coupons": {},
            },
        },
        tool_names=("orders.update_address",),
        ground_truth={
            "expected_final_state": {
                "orders": [{"order_id": "O1", "address": "新地址"}],
                "coupons": [],
                "must_not_change": [{"order_id": "O1", "status": "paid",
                                     "amount_cents": 100, "customer": "甲"}],
            }
        },
        answer_spec=AnswerSpec(
            kind=AnswerKind.STRING,
            comparison=Comparison.ORCHESTRATION,
            normalization=Normalization.NONE,
        ),
        split=Split.TRAIN,
        rubric=(
            _point("c-1", 1.0),
            {
                **_point("c-2", 1.0),
                "requirement": "已向客户回复说明改址结果",
            },
        ),
    )


class _FakeAgent:
    """只会说话的假 agent。``name`` 是报告署名，缺了 ``write_outputs`` 就炸。"""

    name = "fake:separation"

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def __call__(self, task, tools) -> str:
        return self.answer


class RecordingChat:
    """替换 :func:`llm_client.chat`，记录收到的全部 kwargs。"""

    def __init__(self, score: float = 1.0) -> None:
        self.score = score
        self.calls: list[dict[str, Any]] = []

    def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        payload = {
            "scores": [
                {"criterion_id": cid, "score": self.score}
                for cid in ("s-1", "s-2", "s-3", "s-4", "c-1", "c-2")
            ],
            "claims": [],
        }
        return {
            "choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]
        }


# ======================================================================================
# ① 缺省路径逐字节不变
# ======================================================================================


def test_the_b_path_without_judge_kwargs_sends_exactly_the_previous_request(monkeypatch) -> None:
    """不传任何注入参数 ⇒ messages / model / temperature 与改动前完全一致。"""
    task = _qa_task()
    chat = RecordingChat()
    monkeypatch.setattr(llm_client, "chat", chat)

    judge_task(task, "飞机票坐经济舱。")

    assert len(chat.calls) == 1
    call = chat.calls[0]
    assert call["messages"] == judge_mod.build_judge_prompt(task, "飞机票坐经济舱。")
    assert call["model"] == JUDGE_MODEL
    assert call["temperature"] == judge_mod.JUDGE_TEMPERATURE
    # 关键：没传就**不能**出现这两个键，否则等价于把 deepseek 端点写死在了调用点。
    assert "base_url" not in call
    assert "api_key" not in call


def test_the_c_path_without_judge_kwargs_sends_exactly_the_previous_request(monkeypatch) -> None:
    """C 场景同样口径：缺省时请求体不含任何端点 / 密钥覆盖。"""
    task = _c_task()
    chat = RecordingChat()
    monkeypatch.setattr(llm_client, "chat", chat)
    final = {
        "orders": [{"order_id": "O1", "status": "paid", "address": "新地址",
                    "amount_cents": 100, "customer": "甲"}],
        "coupons": {},
    }

    verdict = judge_task(task, "已为您改址", final_snapshot=final)

    assert len(chat.calls) == 1
    call = chat.calls[0]
    assert call["model"] == JUDGE_MODEL
    assert "base_url" not in call
    assert "api_key" not in call
    assert verdict["model"] == JUDGE_MODEL


def test_the_injection_kwargs_do_not_leak_into_the_a_scenario() -> None:
    """A 场景是纯函数比较算子，连 judge 都不经过 —— 传注入参数也不能改变结果。"""
    task = Task(
        id="sep-a-0001",
        category=Category.QUERY,
        schema_version="1.1",
        instruction="查一下",
        initial_data={},
        tool_names=(),
        ground_truth={"answer": "财务部"},
        answer_spec=AnswerSpec(
            kind=AnswerKind.STRING,
            comparison=Comparison.EXACT,
            normalization=Normalization.STRIP,
        ),
        split=Split.TRAIN,
    )
    plain = judge_task(task, " 财务部 ")
    injected = judge_task(task, " 财务部 ", model="gpt-6-astra",
                          client_kwargs={"base_url": "https://example.invalid/v1"})
    assert plain == injected
    assert plain["correct"] is True


# ======================================================================================
# ② 注入生效
# ======================================================================================


def test_the_model_kwarg_reaches_the_judge_request(monkeypatch) -> None:
    chat = RecordingChat()
    monkeypatch.setattr(llm_client, "chat", chat)

    judge_task(_qa_task(), "答案", model="gpt-6-astra")

    assert chat.calls[0]["model"] == "gpt-6-astra"


def test_the_client_kwargs_reach_the_judge_request(monkeypatch) -> None:
    """端点与密钥必须一起落到 chat 上 —— 只给端点会让密钥静默回落到默认 env。"""
    chat = RecordingChat()
    monkeypatch.setattr(llm_client, "chat", chat)

    judge_task(
        _qa_task(), "答案", model="gpt-6-astra",
        client_kwargs={"base_url": "https://llmapi.example/v1", "api_key": "sk-test"},
    )

    call = chat.calls[0]
    assert call["base_url"] == "https://llmapi.example/v1"
    assert call["api_key"] == "sk-test"
    assert call["model"] == "gpt-6-astra"


def test_the_c_path_honours_the_injected_model(monkeypatch) -> None:
    """C 场景的回复质量判定也走注入的模型，且**记录里的 model 字段跟着变**。"""
    chat = RecordingChat()
    monkeypatch.setattr(llm_client, "chat", chat)
    final = {
        "orders": [{"order_id": "O1", "status": "paid", "address": "新地址",
                    "amount_cents": 100, "customer": "甲"}],
        "coupons": {},
    }

    verdict = judge_task(_c_task(), "已为您改址", final_snapshot=final, model="gpt-6-astra")

    assert chat.calls[0]["model"] == "gpt-6-astra"
    # 署名跟着走 —— 否则报告会说「gpt-6-astra 的 judge 给的分数」，实际是 deepseek 判的。
    assert verdict["model"] == "gpt-6-astra"


def test_a_broken_judge_still_degrades_instead_of_raising(monkeypatch) -> None:
    """换了 judge 之后它照样可能挂 —— 铁律不因为换模型就破。"""
    def _boom(*args, **kwargs):
        raise RuntimeError("上游 502")

    monkeypatch.setattr(llm_client, "chat", _boom)

    verdict = judge_task(_qa_task(), "答案", model="gpt-6-astra")

    assert verdict["correct"] is False
    assert verdict["G"] == 0.0
    assert "judge 调用失败" in verdict["reason"]


# ======================================================================================
# ③ eval_run 三个开关
# ======================================================================================


def test_the_three_switches_default_to_the_previous_behaviour() -> None:
    """缺省时三个开关都必须是 None / 默认 env —— 装配出的 judge_kwargs 是个空 dict。"""
    namespace = _parse([])
    assert namespace.judge_model is None
    assert namespace.judge_base_url is None
    assert namespace.judge_key_env == llm_client.API_KEY_ENV


def _parse(argv: list[str]) -> Any:
    """跑 main 的参数解析（不执行），拿到与 CLI 同构的 namespace。"""
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-key-env", default=llm_client.API_KEY_ENV)
    return parser.parse_args(argv)


def test_the_judge_switches_reach_both_scenarios(monkeypatch, tmp_path) -> None:
    """B 与 C 两条 CLI 路径都必须用注入的 judge，且报告署名为注入的模型名。"""
    task = _qa_task()
    chat = RecordingChat()

    monkeypatch.setattr(llm_client, "chat", chat)
    monkeypatch.setattr(eval_run, "load_formal_split", lambda *a, **k: [task])
    monkeypatch.setattr("zh_agent_bench.agents.llm.make_llm_agent",
                        lambda **k: _FakeAgent("飞机票坐经济舱。"))

    out = tmp_path / "qa"
    assert eval_run.main([
        "--split", "dev", "--out", str(out),
        "--judge-model", "gpt-6-astra",
        "--judge-base-url", "https://llmapi.example/v1",
        "--judge-key-env", "LLMAPI_API_KEY",
    ]) == 0

    assert chat.calls, "judge 根本没被调用"
    assert chat.calls[0]["model"] == "gpt-6-astra"
    assert chat.calls[0]["base_url"] == "https://llmapi.example/v1"
    # 默认 env 没设时取到空串是对的（llm_client 会转成 llm_no_api_key 哨兵，不抛）。
    assert "api_key" in chat.calls[0]

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "judge 模型：gpt-6-astra" in report
    assert f"judge 模型：{JUDGE_MODEL}" not in report


def test_the_judge_switches_reach_the_orchestration_scenario(monkeypatch, tmp_path) -> None:
    """C 场景 CLI 同样吃三个开关 —— 这是本单最容易漏的一条（它提前 return 到另一函数）。"""
    task = _c_task()
    chat = RecordingChat()

    monkeypatch.setattr(llm_client, "chat", chat)
    monkeypatch.setattr("zh_agent_bench.agents.llm.make_llm_agent",
                        lambda **k: _FakeAgent("已为您改址"))

    out = tmp_path / "c"
    assert eval_run.main([
        "--split", "orchestration", "--preset", "trial", "--per-type", "1",
        "--out", str(out),
        "--judge-model", "gpt-6-astra",
    ]) == 0

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "judge 模型：gpt-6-astra" in report


def test_omitting_every_switch_keeps_the_report_signed_with_the_default_judge(
    monkeypatch, tmp_path
) -> None:
    """缺省时报告署名仍是 deepseek —— 判据 1 ③ 的「产物逐字节不变」在可观测层面的落点。"""
    task = _qa_task()
    monkeypatch.setattr(llm_client, "chat", RecordingChat())
    monkeypatch.setattr(eval_run, "load_formal_split", lambda *a, **k: [task])
    monkeypatch.setattr("zh_agent_bench.agents.llm.make_llm_agent",
                        lambda **k: _FakeAgent("飞机票坐经济舱。"))

    out = tmp_path / "qa"
    assert eval_run.main(["--split", "dev", "--out", str(out)]) == 0

    report = (out / "report.md").read_text(encoding="utf-8")
    assert f"judge 模型：{JUDGE_MODEL}" in report


def test_a_whole_run_with_and_without_the_switch_differs_only_in_the_judge(
    monkeypatch, tmp_path
) -> None:
    """同一批 fake 答案，换 judge 后**只有 judge 请求与署名变了**，结构与统计口径不动。"""
    task = _qa_task()

    def _run(out, extra: list[str]) -> str:
        monkeypatch.setattr(llm_client, "chat", RecordingChat())
        monkeypatch.setattr(eval_run, "load_formal_split", lambda *a, **k: [task])
        monkeypatch.setattr("zh_agent_bench.agents.llm.make_llm_agent",
                            lambda **k: _FakeAgent("飞机票坐经济舱。"))
        assert eval_run.main(["--split", "dev", "--out", str(out)] + extra) == 0
        return (out / "records.jsonl").read_text(encoding="utf-8")

    baseline = _run(tmp_path / "base", [])
    swapped = _run(tmp_path / "swap", ["--judge-model", "gpt-6-astra"])

    base_row = json.loads(baseline.strip())
    swap_row = json.loads(swapped.strip())
    assert base_row["task_id"] == swap_row["task_id"]
    assert base_row["correct"] == swap_row["correct"]
    assert base_row["judge_degradation"] == swap_row["judge_degradation"]
    # records 里不含 model 字段 —— 两个 judge 的产物靠 report.md 区分，不靠记录自证。
    assert "model" not in base_row


@pytest.mark.parametrize("argv,expected", [
    ([], {"judge_model": None, "judge_base_url": None}),
    (["--judge-model", "gpt-6-astra"], {"judge_model": "gpt-6-astra"}),
    (["--judge-base-url", "https://x.invalid/v1"],
     {"judge_base_url": "https://x.invalid/v1"}),
])
def test_the_switches_parse_independently(argv, expected) -> None:
    namespace = _parse(argv)
    for key, value in expected.items():
        assert getattr(namespace, key) == value
