"""LLM agent 与 llm_client 的测试 —— **全程不联网**。

测试策略：把 :func:`llm_client.chat` 换成预先排好的「剧本」（一个假 chat），
让每个用例精确控制模型每一步说什么。真实网络不可测（会 flaky、还要 key），
而 function-calling 循环的正确性恰恰是「第 N 步返回什么、循环怎么接」——
那是纯逻辑，用假响应就能钉死。

覆盖 TASK-005 判据 1 的六项：
① 完整循环（tool_call → final answer）+ tool_calls 进轨迹
② 工具调用经 check_contract，坏参数不崩
③ max_steps 到顶返回 None
④ 按 answer_spec.kind 解析答案
⑤ 网络/HTTP 异常返回哨兵，不崩
⑥ api_key 不泄漏进任何输出
"""

from __future__ import annotations

import json

import pytest

from zh_agent_bench import llm_client
from zh_agent_bench.agents import llm as llm_agent_mod
from zh_agent_bench.agents.llm import LlmAgent, make_llm_agent
from zh_agent_bench.harness import _ToolBox, run_trajectory
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
# 造任务 / 造响应
# --------------------------------------------------------------------------------------


#: 每种 answer kind 对应的合法 comparison（schema 层会拒非法组合）。
_LEGAL = {
    AnswerKind.STRING: Comparison.EXACT,
    AnswerKind.NUMBER: Comparison.DECIMAL_EXACT,
    AnswerKind.ORDERED_LIST: Comparison.ORDERED_LIST_EXACT,
    AnswerKind.JSON: Comparison.JSON_CANONICAL,
}

#: 每种 kind 的规范答案真值。与 make_task 的默认 initial_data 配套。
_TRUTH = {
    AnswerKind.STRING: "用户运营部",
    AnswerKind.NUMBER: "1.50",
    AnswerKind.ORDERED_LIST: ["周涛", "朱婷", "刘洋"],
    AnswerKind.JSON: {"表名": "商品销售表", "rows": []},
}


def make_task(
    category: Category = Category.QUERY,
    kind: AnswerKind = AnswerKind.STRING,
    tool_names: tuple[str, ...] = ("table.query",),
    instruction: str = "查一下『蔡雯』的部门是什么",
    initial_data: dict | None = None,
    truth: object = None,
) -> Task:
    """造一条最小可用的 Task。answer_spec 的 comparison 必须与 kind 匹配（schema 会拒）。

    真值默认按 kind 取 :data:`_TRUTH`；需要判错的用例传 ``truth`` 覆盖。
    """
    data = initial_data if initial_data is not None else {
        "表名": "员工信息表",
        "rows": [{"姓名": "蔡雯", "部门": "用户运营部", "城市": "南宁"}],
    }
    return Task(
        id=f"{category.value}-9999",
        category=category,
        schema_version="1.0",
        instruction=instruction,
        initial_data=data,
        tool_names=tool_names,
        ground_truth={"answer": _TRUTH[kind] if truth is None else truth},
        answer_spec=AnswerSpec(kind=kind, comparison=_LEGAL[kind], normalization=Normalization.STRIP),
        split=Split.HELDOUT,
    )


def tool_call_response(tool_id: str, arguments: dict | str, call_id: str = "c1") -> dict:
    """造一条「模型要调工具」的响应。

    传**项目工具 ID**（带点），内部转成 OpenAI 形式的函数名（下划线）——
    这正是真实 provider 回给我们的样子，测试因此顺带钉住「模型回下划线名能正确还原调工具」。
    """
    return {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": llm_agent_mod._to_openai_name(tool_id),
                        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                    },
                }],
            },
        }]
    }


def text_response(text: str, finish_reason: str = "stop") -> dict:
    """造一条「模型给最终答案」的响应。"""
    return {
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": text},
        }]
    }


class ScriptedChat:
    """按剧本逐次返回的假 chat。

    记录每次调用的 messages/tools，供断言「回填内容对不对」用。
    """

    def __init__(self, script: list[dict]) -> None:
        self.script = script
        self.calls: list[dict] = []

    def __call__(self, messages, *, tools=None, model=None, base_url=None,
                 api_key=None, timeout=None, **kwargs):
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
        })
        if not self.script:
            raise AssertionError("剧本已耗尽：模型还在被调用（多半是循环没停）")
        return self.script.pop(0)


@pytest.fixture
def fake_chat(monkeypatch):
    """装上假 chat 的 monkeypatch 夹具。返回一个 set_script 风格的构造器。"""
    def install(script: list[dict]) -> ScriptedChat:
        scripted = ScriptedChat(script)
        monkeypatch.setattr(llm_client, "chat", scripted)
        return scripted
    return install


# --------------------------------------------------------------------------------------
# ① 完整循环 + tool_calls 进轨迹
# --------------------------------------------------------------------------------------


def test_full_loop_tool_call_then_answer(fake_chat):
    """模型先调工具、拿到结果后给答案；答案正确，且这次工具调用进了轨迹。"""
    task = make_task()
    scripted = fake_chat([
        tool_call_response("table.query", {"table": task.initial_data["rows"], "filters": {"姓名": "蔡雯"}}),
        text_response("用户运营部"),
    ])
    agent = make_llm_agent(api_key="k")
    box = _ToolBox(task.tool_names)
    result = run_trajectory(task, agent)

    assert result["correct"] is True, result["reason"]
    assert result["final_answer"] == "用户运营部"
    # 轨迹必须留有工具调用记录 —— 后续中文 judge 的 tool-call accuracy 靠它
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "table.query"
    assert len(scripted.calls) == 2


def test_tool_result_is_fed_back_as_role_tool(fake_chat):
    """工具结果必须以 role=tool 且带 tool_call_id 回填，否则 provider 会拒收。"""
    task = make_task()
    scripted = fake_chat([
        tool_call_response("table.query", {"table": task.initial_data["rows"], "filters": {"姓名": "蔡雯"}}),
        text_response("用户运营部"),
    ])
    agent = make_llm_agent(api_key="k")
    agent(task, _ToolBox(task.tool_names))

    second_request = scripted.calls[1]["messages"]
    tool_messages = [m for m in second_request if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "c1"
    # 工具返回值是结构化结果，不是拼好的答案（findings.md §三）
    payload = json.loads(tool_messages[0]["content"])
    assert payload["rows"][0]["部门"] == "用户运营部"


def test_tool_schemas_expose_only_task_tools(fake_chat):
    """只把本任务声明的工具转成 function schema —— 给全量工具就等于泄题。"""
    task = make_task(tool_names=("table.stats",))
    scripted = fake_chat([text_response("0")])
    agent = make_llm_agent(api_key="k")
    agent(task, _ToolBox(task.tool_names))

    names = [t["function"]["name"] for t in scripted.calls[0]["tools"]]
    assert names == ["table_stats"]  # 发给 provider 的名字是下划线形式


def test_openai_function_names_are_legal():
    """OpenAI 要求 function name 匹配 ``^[a-zA-Z0-9_-]+$``。

    工具 ID 带点，直接发过去会被 provider 拒收（实测 HTTP 400）。四个工具全扫一遍，
    保证将来新增带点的工具 ID 也不会漏掉这层转换。
    """
    import re

    legal = re.compile(r"^[a-zA-Z0-9_-]+$")
    for tool_id in ("table.query", "table.stats", "table.organize", "format.csv_to_json"):
        task = make_task(tool_names=(tool_id,))
        name = llm_agent_mod.build_tool_schemas(task)[0]["function"]["name"]
        assert legal.match(name), f"{tool_id} → {name!r} 不符合 OpenAI 命名规范"
        # 靠查表（不是字符串反向替换）还原成工具 ID —— 见 _from_openai_name 的注释
        assert llm_agent_mod._from_openai_name(name, (tool_id,)) == tool_id


def test_from_openai_name_only_accepts_declared_tools():
    """还原只认本任务声明的工具 —— 模型凭空编个名字也调不到别的工具。"""
    # 反向替换的老写法会在这里错把下划线换成点（format.csv.to.json），必须认不出来
    assert llm_agent_mod._from_openai_name("format_csv_to_json", ()) is None
    assert llm_agent_mod._from_openai_name("format_csv_to_json", ("table.query",)) is None
    assert llm_agent_mod._from_openai_name("format_csv_to_json", ("format.csv_to_json",)) \
        == "format.csv_to_json"


def test_tool_schema_only_uses_openai_legal_types():
    """转出来的 function schema 里不允许出现 OpenAI 非法的 type。

    项目 schema 用的 ``{"type": "any"}``（如 table.stats 的 result、format 的空字段约定）
    不是 OpenAI 合法类型，必须降级成不带 type 的空 schema。四个工具全扫一遍，
    保证将来新增工具带 any 类型时也不会漏掉这条转换。
    """
    legal = {"object", "array", "string", "number", "integer", "boolean", "null"}
    for tool_id in ("table.query", "table.stats", "table.organize", "format.csv_to_json"):
        task = make_task(tool_names=(tool_id,))
        schema = llm_agent_mod.build_tool_schemas(task)[0]["function"]["parameters"]
        assert schema["type"] == "object"
        for sub in schema["properties"].values():
            assert "any" not in json.dumps(sub, ensure_ascii=False)
            if "type" in sub:
                assert sub["type"] in legal, f"{tool_id} 的 {sub} 含非法 type"


def test_any_type_field_is_downgraded(monkeypatch):
    """存在 {"type": "any"} 字段时，转换后该字段不再带 type（语义等价于「任意」）。"""
    from zh_agent_bench.tools import registry

    fake_spec = registry.ToolSpec(
        id="fake.any",
        version="1.0",
        fn=lambda p: {},
        input_schema={
            "type": "object",
            "required": ["must"],
            "properties": {
                "must": {"type": "string"},
                "whatever": {"type": "any"},
            },
        },
        output_schema={"type": "object", "required": [], "properties": {}},
    )
    monkeypatch.setattr(registry, "get_tool", lambda tid: fake_spec)

    task = make_task(tool_names=("fake.any",))
    params = llm_agent_mod.build_tool_schemas(task)[0]["function"]["parameters"]
    assert params["required"] == ["must"]
    assert params["properties"]["must"] == {"type": "string"}
    assert params["properties"]["whatever"] == {}  # any -> 无 type


# --------------------------------------------------------------------------------------
# ② 工具调用经 check_contract，坏参数不崩
# --------------------------------------------------------------------------------------


def test_bad_arguments_do_not_crash(fake_chat):
    """参数坏掉（缺必填列）时回填错误文本给模型，agent 自己能纠正并答对。"""
    task = make_task(category=Category.STATS, kind=AnswerKind.NUMBER, tool_names=("table.stats",))
    scripted = fake_chat([
        # 缺 column / op → check_contract 抛 ValueError → 转成 role=tool 的错误文本
        tool_call_response("table.stats", {"table": []}),
        tool_call_response("table.stats", {"table": [1, 2], "column": "销售额", "op": "avg"}),
        text_response("0.02"),
    ])
    agent = make_llm_agent(api_key="k")
    box = _ToolBox(task.tool_names)
    result = run_trajectory(task, agent)

    # 契约报错没有掀桌，agent 重试后仍然答了
    assert result["final_answer"] == "0.02"
    error_feedback = [
        json.loads(m["content"]) for m in scripted.calls[1]["messages"] if m["role"] == "tool"
    ]
    assert "error" in error_feedback[0]
    assert error_feedback[0]["error"]["code"] == "bad_tool_call"


def test_non_json_arguments_do_not_crash(fake_chat):
    """模型吐出非 JSON 的 arguments 字符串 → 判错回填，不抛。"""
    task = make_task()
    scripted = fake_chat([
        tool_call_response("table.query", "这不是 JSON"),
        text_response("用户运营部"),
    ])
    agent = make_llm_agent(api_key="k")
    box = _ToolBox(task.tool_names)
    result = run_trajectory(task, agent)

    assert result["correct"] is True
    feedback = [json.loads(m["content"]) for m in scripted.calls[1]["messages"] if m["role"] == "tool"]
    assert "error" in feedback[0]


def test_unknown_tool_does_not_crash(fake_chat):
    """模型调了本任务没有的工具 → 报 unknown_tool，不抛。"""
    task = make_task(tool_names=("table.query",))
    scripted = fake_chat([
        tool_call_response("table.organize", {"table": [], "sort_by": "销售额"}),
        text_response("用户运营部"),
    ])
    agent = make_llm_agent(api_key="k")
    result = run_trajectory(task, agent)

    assert result["correct"] is True
    feedback = [json.loads(m["content"]) for m in scripted.calls[1]["messages"] if m["role"] == "tool"]
    assert feedback[0]["error"]["code"] == "unknown_tool"


def test_tool_error_is_passthrough_so_the_model_sees_the_real_reason(fake_chat):
    """工具「想返回 error」时 agent 不掀桌、能把**原始 code** 回给模型，并纠正参数重试后答对。

    TASK-007 之前这条路径走的是 ``bad_tool_call``：工具吐的 ``{"error": {"code": "unknown_column"}}``
    不含 ``result``，被 :func:`check_contract` 的 output 校验拦下抛 ``ValueError``，
    agent 在 ``except`` 里把它改写成 ``bad_tool_call`` + 一句校验器的抱怨。**真正的错误原因
    当场被换掉了** —— RESULT-006 的 ``stats-0004`` 里模型说「工具持续报同样的错误」，
    没人看得到那个错误是什么，就丢在这一步。

    现在 error 分支由契约认账（output_schema 的 anyOf），返回值原样透传给模型：
    模型拿得到 ``unknown_column``，才知道该改的是**列名**而不是把同一个调用再发一遍。
    """
    task = make_task(category=Category.STATS, kind=AnswerKind.NUMBER, tool_names=("table.stats",))
    table = [{"销售额": 100}, {"销售额": 200}]
    scripted = fake_chat([
        tool_call_response("table.stats", {"table": table, "column": "不存在", "op": "avg"}),
        tool_call_response("table.stats", {"table": table, "column": "销售额", "op": "avg"}),
        text_response("1.50"),
    ])
    agent = make_llm_agent(api_key="k")
    result = run_trajectory(task, agent)

    assert result["correct"] is True, "透传 error 后 agent 仍该能纠正参数并答对"
    feedback = [json.loads(m["content"]) for m in scripted.calls[1]["messages"] if m["role"] == "tool"]
    assert feedback[0]["error"]["code"] == "unknown_column", (
        "工具的原始错误码必须透传给模型；被改写成 bad_tool_call 就等于把原因吃掉"
    )
    # 同一格也要留在轨迹里（落盘分析靠它），不是只活在回给模型的那段文本里。
    assert result["tool_calls"][0]["error"]["code"] == "unknown_column"


# --------------------------------------------------------------------------------------
# ③ max_steps 到顶
# --------------------------------------------------------------------------------------


def test_max_steps_exhausted_returns_none(fake_chat):
    """一直要调工具、从不给答案 → max_steps 到顶返回 None，judge 判错而不崩。"""
    task = make_task()
    rows = task.initial_data["rows"]
    fake_chat([
        tool_call_response("table.query", {"table": rows, "filters": {"姓名": "蔡雯"}}, call_id=f"c{i}")
        for i in range(5)
    ])
    agent = LlmAgent(api_key="k", max_steps=3)
    result = run_trajectory(task, agent)

    assert result["final_answer"] is None
    assert result["correct"] is False
    assert len(result["tool_calls"]) == 3  # 3 步 = 3 次工具调用


def test_truncated_response_is_not_an_answer(fake_chat):
    """finish_reason=length（回复被截断）→ 视为没答案，继续要下一步。"""
    task = make_task()
    fake_chat([
        text_response('{"部门": "用户运营部"', finish_reason="length"),
        text_response("用户运营部"),
    ])
    agent = make_llm_agent(api_key="k")
    assert run_trajectory(task, agent)["correct"] is True


def test_empty_response_then_none(fake_chat):
    """模型空回（无 content 无 tool_calls）→ 循环继续，步数耗尽后返回 None。"""
    task = make_task()
    empty = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": ""}}]}
    fake_chat([empty, empty])
    agent = LlmAgent(api_key="k", max_steps=2)
    result = run_trajectory(task, agent)
    assert result["final_answer"] is None
    assert result["correct"] is False


# --------------------------------------------------------------------------------------
# ④ 按 answer_spec.kind 解析答案
# --------------------------------------------------------------------------------------


def test_ordered_list_answer_parsed_to_list(fake_chat):
    """organize：JSON 数组文本 → Python list，ordered_list_exact 才判得过。"""
    task = make_task(category=Category.ORGANIZE, kind=AnswerKind.ORDERED_LIST,
                     tool_names=("table.organize",))
    fake_chat([text_response('["周涛", "朱婷", "刘洋"]')])
    agent = make_llm_agent(api_key="k")
    answer = agent(task, _ToolBox(task.tool_names))
    assert answer == ["周涛", "朱婷", "刘洋"]


def test_json_answer_parsed_to_dict(fake_chat):
    """format：JSON 对象文本 → dict。"""
    task = make_task(category=Category.FORMAT, kind=AnswerKind.JSON,
                     tool_names=("format.csv_to_json",), instruction="把这份 CSV 转成 JSON",
                     initial_data={"csv_text": "a,b\n1,2\n"})
    fake_chat([text_response('{"表名": "商品销售表", "rows": []}')])
    agent = make_llm_agent(api_key="k")
    answer = agent(task, _ToolBox(task.tool_names))
    assert answer == {"表名": "商品销售表", "rows": []}


def test_unparseable_json_answer_returns_text_not_raise(fake_chat):
    """解析失败不抛异常：原样返回文本，交给 judge 判错。"""
    task = make_task(category=Category.FORMAT, kind=AnswerKind.JSON,
                     tool_names=("format.csv_to_json",))
    fake_chat([text_response("这不是 JSON")])
    agent = make_llm_agent(api_key="k")
    answer = agent(task, _ToolBox(task.tool_names))
    assert answer == "这不是 JSON"  # 未抛异常，judge 会判 json 解析失败


def test_code_fence_is_stripped(fake_chat):
    """模型爱裹 ```json 围栏；剥掉后再解析，不让格式习惯决定 JSON 类任务的对错。"""
    task = make_task(category=Category.ORGANIZE, kind=AnswerKind.ORDERED_LIST,
                     tool_names=("table.organize",))
    fake_chat([text_response('```json\n["周涛", "朱婷"]\n```')])
    agent = make_llm_agent(api_key="k")
    assert agent(task, _ToolBox(task.tool_names)) == ["周涛", "朱婷"]


def test_number_answer_stays_string(fake_chat):
    """number 类答案保持字符串 —— judge 对 float 判错是有意的（findings.md §二.8）。"""
    task = make_task(category=Category.STATS, kind=AnswerKind.NUMBER, tool_names=("table.stats",))
    fake_chat([text_response("83337.90")])
    agent = make_llm_agent(api_key="k")
    assert agent(task, _ToolBox(task.tool_names)) == "83337.90"


# --------------------------------------------------------------------------------------
# ⑤ 网络 / HTTP 异常 → 哨兵，不崩
# --------------------------------------------------------------------------------------


def test_network_error_returns_sentinel_and_agent_gives_none(monkeypatch):
    """urlopen 抛网络异常 → chat 返回哨兵，agent 返回 None，harness 不崩。"""
    import urllib.error

    def boom(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", boom)
    response = llm_client.chat([{"role": "user", "content": "hi"}], api_key="k")
    assert "error" in response
    assert response["error"]["code"] == "llm_network_error"

    task = make_task()
    result = run_trajectory(task, make_llm_agent(api_key="k"))
    assert result["correct"] is False
    assert result["final_answer"] is None


def test_http_error_returns_sentinel(monkeypatch):
    """HTTP 非 2xx → 哨兵，带状态码。"""
    import io
    import urllib.error

    def boom(request, timeout=None):
        raise urllib.error.HTTPError(
            url="https://api.deepseek.com/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error": "invalid key"}'),
        )

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", boom)
    response = llm_client.chat([{"role": "user", "content": "hi"}], api_key="wrong")
    assert response["error"]["code"] == "llm_http_error"
    assert "401" in response["error"]["message"]


def test_non_json_response_returns_sentinel(monkeypatch):
    """代理返回 HTML 错误页 → 哨兵，不抛 JSONDecodeError。"""
    class FakeResponse:
        def read(self):
            return b"<html>502 Bad Gateway</html>"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", lambda r, timeout=None: FakeResponse())
    response = llm_client.chat([{"role": "user", "content": "hi"}], api_key="k")
    assert response["error"]["code"] == "llm_bad_response"


def test_malformed_choices_do_not_crash(monkeypatch):
    """choices 为空 / 结构不对 → first_message 返回 None，agent 返回 None。"""
    for payload in ({"choices": []}, {"choices": [{}]}, {"nonsense": 1}, []):
        monkeypatch.setattr(llm_client, "chat", lambda *a, **k: payload)
        assert llm_client.first_message(payload) is None
        assert run_trajectory(make_task(), make_llm_agent(api_key="k"))["correct"] is False


def test_missing_api_key_returns_sentinel(monkeypatch):
    """没有 key → 哨兵提示，不发请求。"""
    monkeypatch.delenv(llm_client.API_KEY_ENV, raising=False)
    response = llm_client.chat([{"role": "user", "content": "hi"}])
    assert response["error"]["code"] == "llm_no_api_key"


def test_chat_passes_model_and_base_url(monkeypatch):
    """--model / --base-url 参数确实透传到请求层。"""
    captured = {}

    class FakeResponse:
        def read(self):
            return json.dumps(text_response("ok")).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def capture(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", capture)
    llm_client.chat(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
        model="my-model", base_url="https://example.com/v1/", api_key="k", timeout=30,
    )
    assert captured["url"] == "https://example.com/v1/chat/completions"
    assert captured["body"]["model"] == "my-model"
    assert captured["body"]["tool_choice"] == "auto"
    assert captured["timeout"] == 30
    assert captured["headers"]["Authorization"] == "Bearer k"


# --------------------------------------------------------------------------------------
# ⑥ 密钥不泄漏
# --------------------------------------------------------------------------------------


SECRET = "sk-secret-DO-NOT-LEAK-1234567890"


def _all_text(value) -> str:
    """把任意结构拍平成一段文本，供「这里有没有泄密钥」断言。"""
    return json.dumps(value, ensure_ascii=False, default=str)


def test_api_key_never_leaks_into_any_output(monkeypatch):
    """端到端扫一遍：网络错误、HTTP 错误、正常响应、榜单、轨迹、agent 状态，
    任何一处文本里都不该出现密钥。"""
    import urllib.error

    def boom(request, timeout=None):
        raise urllib.error.HTTPError(
            url="https://api.deepseek.com/v1/chat/completions",
            code=401, msg="Unauthorized", hdrs=None, fp=__import__("io").BytesIO(b"nope"),
        )

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", boom)
    agent = make_llm_agent(api_key=SECRET)
    result = run_trajectory(make_task(), agent)

    for blob in (result, _all_text(result), _all_text(agent.name), agent.name, str(result["reason"])):
        assert SECRET not in _all_text(blob)

    error_text = _all_text(llm_client.chat([{"role": "user", "content": "x"}], api_key=SECRET))
    assert SECRET not in error_text


def test_agent_name_has_no_base_url_or_key():
    """榜单上的 agent 名只含 model —— 不含端点，更不含密钥。"""
    agent = make_llm_agent(model="deepseek-chat", base_url="https://example.com/v1", api_key=SECRET)
    assert agent.name == "llm-agent:deepseek-chat"
    assert "example.com" not in agent.name
    assert SECRET not in agent.name


def test_api_key_goes_only_into_authorization_header(monkeypatch):
    """密钥只出现在 Authorization 头里，不进 URL、不进请求体。"""
    captured = {}

    class FakeResponse:
        def read(self):
            return b'{"choices": []}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def capture(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = request.data.decode("utf-8")
        captured["auth"] = request.headers.get("Authorization")
        return FakeResponse()

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", capture)
    llm_client.chat([{"role": "user", "content": "x"}], api_key=SECRET)
    assert captured["auth"] == f"Bearer {SECRET}"
    assert SECRET not in captured["url"]
    assert SECRET not in captured["body"]


def test_error_message_keeps_host_only(monkeypatch):
    """报错里只留 host —— 万一调用方把密钥拼进了 url 路径也不会被回显。"""
    import urllib.error

    def boom(request, timeout=None):
        raise urllib.error.HTTPError(
            url="https://example.com/v1", code=500, msg="x", hdrs=None,
            fp=__import__("io").BytesIO(b""),
        )

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", boom)
    # 即使 base_url 里含密钥，报错也只暴露 host
    response = llm_client.chat([], api_key="k", base_url="https://example.com/sk-leak")
    assert "sk-leak" not in response["error"]["message"]


# --------------------------------------------------------------------------------------
# prompt 构造
# --------------------------------------------------------------------------------------


def test_system_prompt_carries_the_money_rule():
    """金额铁律必须进 prompt —— judge 对 float 判错是有意的（findings.md §五.2）。"""
    task = make_task(category=Category.STATS, kind=AnswerKind.NUMBER, tool_names=("table.stats",))
    prompt = llm_agent_mod.build_system_prompt(task)
    assert "十进制字符串" in prompt
    assert "禁止输出浮点数" in prompt
    assert "只输出答案本体" in prompt


def test_user_prompt_contains_instruction_and_data():
    task = make_task()
    prompt = llm_agent_mod.build_user_prompt(task)
    assert "查一下『蔡雯』的部门是什么" in prompt
    assert "用户运营部" in prompt


def test_parse_answer_strips_fence_only_when_paired():
    """半截围栏当普通文本处理（judge 自然判错），不做启发式猜测。"""
    assert llm_agent_mod.parse_answer("```json\n[1]", AnswerKind.JSON) == "```json\n[1]"
    assert llm_agent_mod.parse_answer("```json\n[1]\n```", AnswerKind.JSON) == [1]
    # 无围栏的纯文本原样返回
    assert llm_agent_mod.parse_answer("  hi  ", AnswerKind.STRING) == "hi"


# --------------------------------------------------------------------------------------
# 同一实例跑多条任务
# --------------------------------------------------------------------------------------


def test_agent_instance_is_reusable(fake_chat):
    """同一实例连跑两条任务互不串味（消息历史不跨任务累积）。"""
    scripted = fake_chat([text_response("甲"), text_response("乙")])
    agent = make_llm_agent(api_key="k")
    assert agent(make_task(), _ToolBox(make_task().tool_names)) == "甲"
    assert agent(make_task(), _ToolBox(make_task().tool_names)) == "乙"
    # 第二次请求的消息里不应有第一次的 user 内容残留成两条
    assert len(scripted.calls[1]["messages"]) == 2


# --------------------------------------------------------------------------------------
# TASK-021：C 场景 schema 从 box 取（修 TASK-020 打回的根因）
#
# 根因：`build_tool_schemas` 只拿 task、硬查全局 `TOOL_REGISTRY`，而 C 场景的沙箱工具
# （order.get 等）不在那张表里 → `ValueError: 未知工具 ID: 'order.get'`，真 key 跑分 8 条全挂。
# 下面第一条测试就是这个 bug 的原样复现 —— **它必须能在没有 box 时才红**，
# 否则打回一次还会再炸一次。
# --------------------------------------------------------------------------------------


def _orchestration_task(tool_names: tuple[str, ...] = ("order.get", "order.refund")) -> Task:
    """一条最小可用的 C 场景任务（answer_spec 走 ORCHESTRATION 比较）。"""
    from zh_agent_bench import sandbox as sb
    from zh_agent_bench.tasks.schema import Comparison

    return Task(
        id="orch-9999", category=Category.ORCHESTRATION, schema_version="1.1",
        instruction="处理这笔订单",
        initial_data={"任务类型": "退款判断",
                      "sandbox": sb.snapshot(sb.make_sandbox(42))},
        tool_names=tool_names,
        ground_truth={"expected_final_state": {"orders": [], "coupons": []}},
        answer_spec=AnswerSpec(kind=AnswerKind.JSON, comparison=Comparison.ORCHESTRATION,
                               normalization=Normalization.NONE),
        split=Split.TRAIN, rubric=(),
    )


def _orchestration_box(tool_names: tuple[str, ...]):
    from zh_agent_bench import sandbox as sb
    from zh_agent_bench.harness import _SandboxToolBox

    return _SandboxToolBox(tool_names, sb.tool_functions(sb.make_sandbox(42)))


def test_c_scene_schemas_do_not_raise_unknown_tool_id():
    """C 场景建 schema **不许抛** —— 这正是 TASK-020 打回时 8 条全挂的那一句。

    注意这里显式传 box：不传就退回全局注册表路径、照样抛 ``未知工具 ID``。
    """
    task = _orchestration_task()
    schemas = llm_agent_mod.build_tool_schemas(task, _orchestration_box(task.tool_names))
    assert [s["function"]["name"] for s in schemas] == ["order_get", "order_refund"]


def test_c_scene_schema_parameters_come_from_the_sandbox_truth_source():
    """parameters 与 sandbox.TOOL_SPECS 同形（description 也带过去了，不是干巴巴的占位）。"""
    from zh_agent_bench.sandbox import TOOL_SPECS

    task = _orchestration_task(("order.refund",))
    schema = llm_agent_mod.build_tool_schemas(task, _orchestration_box(task.tool_names))[0]
    params = schema["function"]["parameters"]
    assert params["required"] == TOOL_SPECS["order.refund"]["required"]
    assert set(params["properties"]) == set(TOOL_SPECS["order.refund"]["properties"])
    assert schema["function"]["description"]


def test_c_scene_openai_names_are_legal():
    """沙箱工具名也带点（``order.get``）→ 必须同样走点换下划线的转换，否则 HTTP 400。"""
    import re

    legal = re.compile(r"^[a-zA-Z0-9_-]+$")
    for tool_id in ("order.list", "order.get", "order.change_address",
                    "order.refund", "inventory.check", "coupon.use"):
        task = _orchestration_task((tool_id,))
        schema = llm_agent_mod.build_tool_schemas(task, _orchestration_box((tool_id,)))[0]
        name = schema["function"]["name"]
        assert legal.match(name), f"{tool_id} → {name!r} 违反 OpenAI 命名规范"
        # 查表还原（不是反向替换）：工具 ID 本身不含下划线，但保持与 A/B 同一条路径
        assert llm_agent_mod._from_openai_name(name, (tool_id,)) == tool_id


def test_a_and_b_scenes_still_go_through_the_global_registry(monkeypatch):
    """**A/B 行为逐字不变**：``_ToolBox`` 没有 specs()，必须仍走 ``get_tool``。"""
    from zh_agent_bench.tools import registry

    seen: list[str] = []
    real_get_tool = registry.get_tool

    def _tracking(tid):
        seen.append(tid)
        return real_get_tool(tid)

    monkeypatch.setattr(registry, "get_tool", _tracking)

    task = make_task(tool_names=("table.stats",))
    schemas = llm_agent_mod.build_tool_schemas(task, _ToolBox(task.tool_names))
    assert seen == ["table.stats"], "A/B 路径没走注册表 —— A/B 行为变了"
    assert schemas[0]["function"]["name"] == "table_stats"


def test_a_and_b_scenes_ignore_a_box_without_specs():
    """A/B 的 ``_ToolBox`` 没有 specs()，即使显式传进去也必须仍走注册表。"""
    task = make_task(tool_names=("table.stats",))
    assert not hasattr(_ToolBox(task.tool_names), "specs")
    schemas = llm_agent_mod.build_tool_schemas(task, _ToolBox(task.tool_names))
    assert [s["function"]["name"] for s in schemas] == ["table_stats"]


def test_box_schemas_win_over_the_registry_when_both_are_possible(monkeypatch):
    """两条路都在时**优先 box** —— 这是本单的修法本身，必须钉住优先级不被后来人改反。"""
    from zh_agent_bench.tools import registry

    def _explode(tid):
        raise AssertionError(f"C 场景不该碰全局注册表，却查了 {tid}")

    monkeypatch.setattr(registry, "get_tool", _explode)

    task = _orchestration_task(("order.get",))
    schemas = llm_agent_mod.build_tool_schemas(task, _orchestration_box(task.tool_names))
    assert [s["function"]["name"] for s in schemas] == ["order_get"]


def test_the_c_scene_agent_loop_runs_end_to_end_with_a_fake_chat(fake_chat):
    """假 chat 跑完整循环：模型调沙箱工具 → 拿结果 → 出回复。轨迹里有这次调用。

    这是**能离线跑的最接近真跑的形状** —— 真 key 跑（判据 3）之前先在这里确认循环通。
    """
    task = _orchestration_task(("order.get", "order.refund"))
    scripted = fake_chat([
        tool_call_response("order.get", {"order_id": "SO-20260001"}),
        text_response("该订单不能退款"),
    ])
    box = _orchestration_box(task.tool_names)
    result = make_llm_agent(api_key="k")(task, box)

    assert result == "该订单不能退款"
    assert [c["tool"] for c in box.calls] == ["order.get"]
    assert [t["function"]["name"] for t in scripted.calls[0]["tools"]] == \
        ["order_get", "order_refund"]


def test_a_c_scene_tool_error_is_fed_back_to_the_model(fake_chat):
    """沙箱的 unknown_id / invalid_state 是给模型看的可纠正信号，不是崩溃。"""
    task = _orchestration_task(("order.refund",))
    scripted = fake_chat([
        tool_call_response("order.refund", {"order_id": "不存在的单", "reason": "不想要了"}),
        tool_call_response("order.refund", {"order_id": "SO-20260001", "reason": "不想要了"}),
        text_response("已退"),
    ])
    box = _orchestration_box(task.tool_names)
    make_llm_agent(api_key="k")(task, box)

    assert len(box.calls) == 2
    assert box.calls[0]["error"]["code"] == "unknown_id"
    # 错误是以 role=tool 的 JSON 喂回去的。注意不能直接断言 scripted.calls[1]：
    # ScriptedChat 存的是 messages 列表的**引用**，而循环在原地 append，
    # 所以每条记录看到的是同一个增长中的 list —— 要按 role 找那条 tool 消息。
    tool_messages = [m for m in scripted.calls[0]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 2
    assert json.loads(tool_messages[0]["content"])["error"]["code"] == "unknown_id"
    assert "error" not in json.loads(tool_messages[1]["content"])
