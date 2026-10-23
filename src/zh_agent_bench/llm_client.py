"""OpenAI 兼容 chat completions 客户端 —— **仅 stdlib urllib**，不新增任何依赖。

为什么不用 requests：M1 的全部承诺是「clone 即跑、运行时零第三方依赖」。为了调一次
HTTP 就引入依赖，等于把 M1 建立起来的那条承诺（榜单不装任何包就能跑）撕开一道口子。
``urllib.request`` 做一次 POST 足够。

两条铁律（比功能更重要）：

1. **密钥绝不泄漏**。``api_key`` 只进请求头，绝不进 URL、返回值、异常消息或日志。
   错误信息里连 base_url 都只保留 host —— 万一有人把 key 拼在 url 里也不会被回显。
2. **本模块绝不抛异常**。网络不通 / HTTP 非 2xx / 超时 / 返回非 JSON，一律转成
   与工具错误同形状的哨兵 ``{"error": {"code": ..., "message": ...}}``：
   agent 侧识别 ``error`` 键就转成「答不出来」，harness 侧 judge 照常判错。
   让一批跑挂的评测把整批评测挂掉，是评测工具最不该犯的错。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

#: 默认 provider（OpenAI 兼容端点，换 qwen / moonshot 只改 base_url + model）。
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

#: 单次请求超时上限（秒）。评测是离线批处理，宁可慢也不要挂死。
DEFAULT_TIMEOUT = 60

#: 单次回复上限。工具调用只需几十 token 的函数名+参数，答案也不会超过一条 JSON 行。
MAX_TOKENS = 2048

#: 默认 API key 的环境变量名（CLI 也用它兜底）。
API_KEY_ENV = "DEEPSEEK_API_KEY"


def make_error(code: str, message: str) -> dict[str, Any]:
    """与 ``tools/_errors.py`` 同形状的错误哨兵。

    刻意不 import ``tools._errors.ErrorCode`` —— 那个枚举是**工具契约版本**的一部分，
    LLM 传输层的错误不属于契约，改它就该 bump 版本。这里只借形状。
    """
    return {"error": {"code": code, "message": message}}


def _host_only(url: str) -> str:
    """从 url 里只留 host 部分用于报错。防御：调用方可能把密钥拼在路径里。"""
    parts = url.split("/")
    return parts[2] if len(parts) > 2 else "<url>"


def chat(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: int = MAX_TOKENS,
    temperature: float | None = None,
) -> dict[str, Any]:
    """发一次 chat completions，返回解析后的 JSON（含 ``choices[0].message``）。

    :param messages: OpenAI 格式的消息列表。
    :param tools: OpenAI function-calling 的 ``tools`` 数组；``None`` 表示不带工具。
    :param temperature: **``None``（默认）= 不发这个字段**，由服务端用各模型默认值。
        judge 传 ``0.0`` 求尽量确定；agent 侧保持 ``None`` 不受影响（少发一个字段，
        就少一个「不同 provider 对缺省值处理不同」的变量）。
    :returns: 服务端原始 JSON，或 :func:`make_error` 的哨兵。**任何情况下都不抛异常。**
    """
    # 显式参数优先于 env；两者都没有才报「没 key」。
    key = api_key or os.environ.get(API_KEY_ENV, "")
    if not key:
        return make_error("llm_no_api_key", f"未提供 API key（--api-key 或环境变量 {API_KEY_ENV}）")

    url = base_url.rstrip("/") + "/chat/completions"
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    request = urllib.request.Request(
        url,
        # ensure_ascii=False：中文指令以 UTF-8 原样发出，转义成 \uXXXX 既臃肿又难排查。
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # HTTPError 是 URLError 的子类，必须先接它；body 里常有服务端的具体理由。
        detail = exc.read().decode("utf-8", "replace")[:300]
        return make_error("llm_http_error", f"HTTP {exc.code} @ {_host_only(url)}: {detail}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # 网络层异常（含 socket.timeout、ConnectionReset）—— 一律降级成哨兵。
        return make_error("llm_network_error", f"{type(exc).__name__} @ {_host_only(url)}: {exc}")

    try:
        payload = json.loads(raw)
    except ValueError:
        # 代理返回 HTML 错误页之类。截断原文，别把整个页面塞进榜单。
        return make_error("llm_bad_response", f"响应不是合法 JSON（前 200 字）: {raw[:200]}")
    if not isinstance(payload, dict):
        return make_error("llm_bad_response", f"响应顶层不是对象，而是 {type(payload).__name__}")
    return payload


def first_choice(response: dict[str, Any]) -> dict[str, Any] | None:
    """从响应里取 ``choices[0]``。取不到返回 ``None``（由调用方判错处理）。

    抽出来是因为「哨兵 / 结构异常 / 正常」三种情况的判定要在 agent 侧做一次，
    不该散在两个模块里各判一遍。

    返回 **choice 而非 message**：``finish_reason`` 挂在 choice 上，不在 message 上，
    而「回复是不是被截断」恰恰是「这段 content 能不能当答案」的关键判据。
    """
    if not isinstance(response, dict) or "error" in response:
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    return first if isinstance(first, dict) else None


def first_message(response: dict[str, Any]) -> dict[str, Any] | None:
    """从响应里取 ``choices[0].message``。取不到返回 ``None``。"""
    choice = first_choice(response)
    if choice is None:
        return None
    message = choice.get("message")
    return message if isinstance(message, dict) else None
