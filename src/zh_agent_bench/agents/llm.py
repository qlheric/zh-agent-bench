"""LLM agent —— OpenAI 兼容的 function-calling 循环。

与 :mod:`reference` 协议一致（``solve(task, tools) -> answer``），但决策交给模型：
读中文指令 → 自己选工具 → 自己拼参数 → 自己给答案。榜单分数由此才第一次**不是 100%**，
而那才是「评 agent」的真实含义。

**必须用真 function-calling，不能把工具塞进 prompt 让它手写 JSON。**
后续中文 judge 要评 tool-call accuracy，看的是「agent 主动挑了哪个工具、传了什么参」；
手写 JSON 的话这个信号就退化成了字符串解析技巧，评的不是工具调用能力。

三条设计约束：

1. **工具只经 ``tools[name](payload)`` 调用**（该代理内部走 :func:`check_contract`，
   findings.md §四.5）。契约漂移当场暴露，轨迹里也会留下这次调用。
2. **绝不让 LLM 的失败变成评测的失败**。网络错、返回结构怪、模型死循环，
   一律转成「答不出来」（``None``）交回 judge 判错，而不是抛出去挂掉整批。
3. **答案类型由 agent 自己解析成 Python 值**再交给 judge：``ordered_list`` / ``json``
   走 :func:`json.loads`。解析失败就原样返回文本让 judge 判错 —— 判分器接得住字符串，
   这里不需要再多一层容错。
"""

from __future__ import annotations

import json
from typing import Any

from .. import llm_client
from ..tasks.schema import AnswerKind, Task

#: harness 用这个名字标注「本榜由 LLM agent 产生」。
LLM_AGENT_NAME = "llm-agent"

#: 默认最大步数（一次 LLM 回复 = 一步）。8 步够 4–5 次工具调用 + 一次收尾，
#: 又不至于让模型在工具链上无限打转把评测拖成天价账单。
DEFAULT_MAX_STEPS = 8


def _answer_rule(kind: AnswerKind) -> str:
    """按 ``answer_spec.kind`` 给出最终答案的输出格式指令。

    这是 prompt 里唯一「按类别变化」的部分：判分是按 kind 分四种算子的，
    告诉模型「输出 JSON 数组」却不告诉它这是哪种任务，模型大概率给你套一层 markdown 代码块。
    """
    return {
        AnswerKind.STRING: (
            "直接输出这个值本身（不要引号、不要解释、不要单位）。"
            "如果值是空字段或工具返回 null，就原样输出空字符串。"
        ),
        AnswerKind.NUMBER: (
            "直接输出一个十进制字符串，保留两位小数（例如 83337.90）。"
            "**禁止输出浮点数、科学计数法或计算过程** —— 判分按十进制精确相等，"
            "浮点尾巴会被判错。"
        ),
        AnswerKind.ORDERED_LIST: (
            "输出一个 JSON 数组，元素是字符串，**顺序必须与结果一致**，不要加 markdown 代码块。"
            '例如 ["张三", "李四"]'
        ),
        AnswerKind.JSON: (
            "输出一个 JSON 对象，键序无所谓、但数组元素的顺序必须与结果一致，不要加 markdown 代码块。"
            '例如 {"表名": "销售明细表", "rows": [{"姓名": "张三"}]}'
        ),
    }[kind]


def build_system_prompt(task: Task) -> str:
    """构造中文 system prompt。

    要点固定为四条：只用给定工具 / 按 schema 传参 / 金额铁律 / 只输出答案本体。
    第四条最容易被忽略但最影响分数 —— 模型爱解释，一句「销售额最高的员工是王伟，销售额为
    50000 元」在 ``exact`` 判分下必然判错。
    """
    return (
        "你是一个解决中文办公数据任务的 agent。\n"
        "\n"
        "规则：\n"
        "1. 你只能调用下面提供的工具，每个任务的工具互不相同；"
        "工具的参数必须严格符合它的 schema。\n"
        "2. 需要的数据必须**通过调用工具取得**，不要凭空猜测，也不要自己心算数据。\n"
        "3. 金额一律以整数分存储，统计结果的金额请输出十进制字符串（保留两位小数），"
        "禁止输出浮点数或科学计数法。\n"
        "4. 工具返回 error 时读它的 code 与 message，修正参数后重试，最多一次。\n"
        "\n"
        "最终答案的输出格式（极其重要）：\n"
        f"- 任务类别：{task.category.value}\n"
        f"- {task.answer_spec.kind.value}：{_answer_rule(task.answer_spec.kind)}\n"
        "- **只输出答案本体**，不要加解释、前缀、引号或 markdown 代码块。"
        "judge 会把你的输出按上面写的类别逐字比对，多余的字一律判错。\n"
    )


def build_user_prompt(task: Task) -> str:
    """构造 user 消息：指令 + 初始数据。

    ``initial_data`` 原样 JSON 序列化（含中文与 ``\\n`` 转义），不在这里做格式化 ——
    任何「帮模型收拾一下数据」的操作都会把工具本该做的活儿偷偷替它干了，
    评出来的就不是 agent 的能力了。
    """
    return (
        f"任务指令：{task.instruction}\n"
        f"\n"
        f"初始数据：\n"
        f"{json.dumps(task.initial_data, ensure_ascii=False, indent=2)}\n"
    )


#: OpenAI 规范要求 function name 匹配 ``^[a-zA-Z0-9_-]+$``，而工具 ID 里带点
#: （``format.csv_to_json``）—— 直接发过去会被 provider 拒收（实测 HTTP 400）。
#:
#: **改的是发给 provider 的函数名，不是工具 ID**：工具 ID 是 ``tools/registry.py``
#: 冻结的契约，改它等于换工具（findings.md §三）。
#:
#: 还原**不做字符串反向替换**，而是拿本任务声明的工具 ID 逐一比对 —— 工具 ID 里
#: 本来就含下划线（``csv_to_json``），反向替换会把它错换成点。


def _to_openai_name(tool_id: str) -> str:
    """``format.csv_to_json`` → ``format_csv_to_json``。"""
    return tool_id.replace(".", "_")


def _from_openai_name(name: str, tool_names: tuple[str, ...]) -> str | None:
    """把 provider 回的函数名还原成工具 ID。认不出来返回 ``None``。

    查表而非反向替换：本项目的工具 ID 本身含下划线（``format.csv_to_json``），
    字符串反向替换会把它错换成点。查表天然只认本任务声明的工具 ——
    模型就算凭空编一个名字，也调不到任务外的工具。
    """
    for tool_id in tool_names:
        if _to_openai_name(tool_id) == name:
            return tool_id
    return None


def build_tool_schemas(task: Task, tools: Any = None) -> list[dict[str, Any]]:
    """把任务声明的工具转成 OpenAI function schema。

    **优先问 box 要 schema**（``tools.specs()``），拿不到才走全局 ``TOOL_REGISTRY``。
    这不是「兼容两种 box」的礼貌写法，而是修一个真炸过的问题：TASK-020 打回的根因就是
    这里硬查全局注册表，而 C 场景的沙箱工具（``order.get`` 等）压根不在那张表里，
    真 key 跑分 8 条全挂在 ``ValueError: 未知工具 ID: 'order.get'``。
    —— 而 mock 测试全绿，因为假 agent 直接吃 box、从不建 schema。**mock 绕过的路径，
    生产时就是雷。**

    两条路径都只吃 schema，转换逻辑完全一样：
    1. **函数名**：``format.csv_to_json`` → ``format_csv_to_json``（点不是 OpenAI 合法字符，
       直接发过去会拿到 HTTP 400 —— 实测踩到的坑）；
    2. **类型降级**：``{"type": "any"}`` 在 OpenAI 规范里不是合法类型
       （合法值是 object/array/string/number/integer/boolean/null），
       降级成不带 ``type`` 的空 schema，语义等价于「任意」。

    :param tools: harness 的工具视图。给 ``_SandboxToolBox``（C 场景）时走 box 路径，
        给 :class:`harness._ToolBox`（A/B）或 ``None`` 时走注册表路径 —— 后者**行为逐字不变**。
    """
    from ..tools.registry import get_tool  # 局部导入：tools 包较重，测试里 import 本模块不必拖上它

    box_specs = getattr(tools, "specs", None)
    if callable(box_specs):
        return [
            {
                "type": "function",
                "function": {
                    "name": _to_openai_name(spec["name"]),
                    "description": spec["description"],
                    "parameters": _normalize_parameters(spec["parameters"]),
                },
            }
            for spec in box_specs()
        ]

    schemas: list[dict[str, Any]] = []
    for tool_id in task.tool_names:
        spec = get_tool(tool_id)
        schemas.append({
            "type": "function",
            "function": {
                "name": _to_openai_name(tool_id),
                "description": f"中文办公数据工具 {tool_id}",
                "parameters": _normalize_parameters(spec.input_schema),
            },
        })
    return schemas


def _normalize_parameters(input_schema: dict[str, Any]) -> dict[str, Any]:
    """项目 input_schema → OpenAI ``parameters``。

    ``required`` 只列必填项；可选参数（table.organize 的 order/limit/tiebreak、
    order.list 的 customer）留给模型自己判断，schema 里标 required 反而会诱导它瞎填。
    """
    return {
        "type": "object",
        "properties": {
            name: ({} if sub.get("type") == "any" else dict(sub))
            for name, sub in input_schema.get("properties", {}).items()
        },
        "required": list(input_schema.get("required", [])),
    }


def _invoke_tool(openai_name: str, call: dict[str, Any], tools: dict,
                 tool_names: tuple[str, ...]) -> str:
    """执行一次 tool_call 并回填文本。**不抛异常**。

    :param openai_name: 模型回的函数名（**已转成 OpenAI 形式的**，点换成了下划线），
        这里用 :func:`_from_openai_name` 拿本任务声明的工具还原成工具 ID。
    :param tool_names: 本任务 :attr:`Task.tool_names`，还原函数名的查表范围。

    参数解析（模型给的是 JSON 字符串）、未知工具、契约校验失败、结构不对 ——
    全部转成一段给模型看的错误文本，让它自己纠正。执行异常也要接住：
    LLM 调错工具不该让整批评测崩掉，那会把「agent 答错」和「评测跑不完」混成同一件事。
    """
    tool_id = _from_openai_name(openai_name, tool_names)
    try:
        raw_args = call.get("arguments") or "{}"
        payload = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        if not isinstance(payload, dict):
            return json.dumps(
                llm_client.make_error("invalid_arguments", "arguments 必须是 JSON 对象"),
                ensure_ascii=False,
            )
        if tool_id is None or tool_id not in tools:
            # tools 是 harness 的 _ToolBox（不是 dict），所以这里不能 sorted(tools)，
            # 也不知道别的工具名。报「本任务未提供该工具」即可 —— agent 侧只需知道这条路走不通。
            return json.dumps(
                llm_client.make_error("unknown_tool", f"本任务未提供工具 {openai_name!r}"),
                ensure_ascii=False,
            )
        # 关键路径：经 harness 的工具代理执行，内部走 check_contract（findings.md §四.5），
        # 并把这次调用记进轨迹 —— 后续中文 judge 的 tool-call accuracy 就靠这条记录。
        return json.dumps(tools[tool_id](payload), ensure_ascii=False)
    except ValueError as exc:  # json.JSONDecodeError 是 ValueError 子类；check_contract 也抛 ValueError
        return json.dumps(llm_client.make_error("bad_tool_call", str(exc)), ensure_ascii=False)
    except Exception as exc:  # 兜底：任何工具侧的意外都不该掀桌
        return json.dumps(
            llm_client.make_error("tool_call_failed", f"{type(exc).__name__}: {exc}"),
            ensure_ascii=False,
        )


def _strip_code_fence(text: str) -> str:
    """剥掉模型爱加的 ```json ... ``` 围栏。

    这不是给模型放水：围栏里包着的值照样要判分，只是不让「有没有加围栏」这个
    与任务能力无关的格式习惯决定 JSON 类任务的对错 —— 否则 organize/format 两类
    会被围栏问题整类拖死，榜单测的就不是 agent 会不会干活了。
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    # 围栏必须成对，否则当普通文本（judge 自然判错，不在这里猜）。
    if len(lines) < 2 or not lines[-1].strip().startswith("```"):
        return stripped
    return "\n".join(lines[1:-1]).strip()


def parse_answer(text: str, kind: AnswerKind) -> Any:
    """把模型文本答案按 ``kind`` 解析成 judge 吃得下的 Python 值。

    ``ordered_list`` / ``json`` 走 :func:`json.loads`，失败**原样返回文本**：
    judge 的 ``json_canonical`` 自己也会解析字符串，解析不了就判错，语义一致，
    且判错理由（judge 生成的）比这里自己编一句更准。
    """
    text = _strip_code_fence(text)
    if kind is AnswerKind.JSON or kind is AnswerKind.ORDERED_LIST:
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text


def _answer_from_choice(choice: dict[str, Any]) -> str | None:
    """取最终答案文本。``finish_reason == "length"`` 视为**没有答案**（截断的 JSON 判不了）。

    不能只按「有没有 content」判：被 ``max_tokens`` 截断的回复会留下半截 content，
    那正是最容易产出「半个 JSON」的情况 —— 让它去当答案，判错理由会误导成
    「内容不对」而不是「回复被截断」，把轨迹分析带偏。
    """
    if choice.get("finish_reason") == "length":
        return None
    message = choice.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return None


class LlmAgent:
    """function-calling 循环。可复用同一实例跑多条任务（配置不变）。

    刻意做成有状态对象而不是闭包：榜单要按 agent 标注来源，实例上带着 model/base_url，
    打印榜单时能说清「这榜是谁跑的」，而不只是「llm-agent」。
    """

    def __init__(
        self,
        *,
        model: str = llm_client.DEFAULT_MODEL,
        base_url: str = llm_client.DEFAULT_BASE_URL,
        api_key: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        timeout: int = llm_client.DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_steps = max_steps
        self.timeout = timeout

    @property
    def name(self) -> str:
        """harness 榜单用的 agent 名。只含 model，不含 base_url / 密钥。"""
        return f"{LLM_AGENT_NAME}:{self.model}"

    def __call__(self, task: Task, tools: dict) -> Any:
        """Agent 协议：``(task, tools) -> final_answer``。

        返回 ``None`` 表示「答不出来」（LLM 出错或步数用尽）。harness 交给 judge 判错，
        不抛 —— 判分器接得住 None。
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(task)},
            {"role": "user", "content": build_user_prompt(task)},
        ]
        schemas = build_tool_schemas(task, tools)

        for _ in range(self.max_steps):
            response = llm_client.chat(
                messages,
                tools=schemas,
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
            choice = llm_client.first_choice(response)
            if choice is None:
                # 哨兵（网络/HTTP/key/结构异常）或 choices 为空。转成 None，judge 判错。
                return None
            message = choice.get("message")
            if not isinstance(message, dict):
                return None

            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                messages.append(message)
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    call_id = call.get("id") or f"call_{len(messages)}"
                    function = call.get("function")
                    if not isinstance(function, dict):
                        continue
                    name = function.get("name") or ""
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _invoke_tool(name, function, tools, task.tool_names),
                    })
                continue  # 回到模型，让它看结果

            text = _answer_from_choice(choice)
            if text is not None:
                return parse_answer(text, task.answer_spec.kind)
            # 既无 tool_calls 也无 content（模型空回 / 被截断）：再给一步机会，
            # 步数用尽则落到循环外的 return None。

        return None  # max_steps 到顶仍无答案 → judge 判错，不崩


def make_llm_agent(
    model: str = llm_client.DEFAULT_MODEL,
    base_url: str = llm_client.DEFAULT_BASE_URL,
    api_key: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    timeout: int = llm_client.DEFAULT_TIMEOUT,
) -> LlmAgent:
    """工厂：按配置造一个满足 harness ``Agent`` 协议的 LLM agent。

    与 ``reference.solve`` 同签名，所以 :func:`harness.evaluate` 能一视同仁地跑。
    """
    return LlmAgent(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_steps=max_steps,
        timeout=timeout,
    )
