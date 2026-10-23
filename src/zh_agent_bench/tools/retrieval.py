"""检索工具：``knowledge.search`` —— B 场景（知识检索 QA）的唯一工具。

A 场景的表工具是「给结构化数据、算确定值」；B 场景反过来：**问题在中文文档里，
知识库可能压根没有答案**。所以这个工具的契约与表工具有两处刻意的不同：

- **不预制答案**：返回的是带 ``score`` 的文档切片，不是问题的答案。
  「这些文档够不够回答问题」是 agent 的判断，不是工具的判断（brainstorm §3.3
  「检索证据覆盖率」是诊断指标，不该混进判分）。
- **允许零命中**：查不到就是 ``{"hits": []}``，**不是 error**。零命中是合法结果，
  agent 据此回答「证据不足」——那正是 B 场景要考的能力之一。error 只留给人没把
  参数用对（空 query、类型错）的情况。

**确定性铁律**（与 ``table.py`` 同款）：同输入 → 同输出，逐字节一致；无随机、不读时钟、
不联网、不改入参、不依赖字典迭代序（排序显式给全序 tiebreak）。

**分词不引依赖**：中文用**字符 bigram**（``销售额`` → ``销售``/``售额``），
西文按整词切。相比单字，bigram 显著降低误召回；相比 jieba，零依赖且没有版本漂移，
对「同输入同输出」这条判据更友好。代价是召回不如词典分词精确 —— 对检索工具够用，
真要提升质量是换分词器的事，不该在这里先造轮子。
"""

from __future__ import annotations

import re
from typing import Any

from ._errors import ErrorCode, make_error

#: ``top_k`` 的默认值与硬上限。默认 5：够覆盖一个事实的常见分布，又不至于把
#: 整库塞进上下文。上限 10 是防「agent 要 1000 条」把上下文打爆 —— 检索是找证据，
#: 不是导出全表。
DEFAULT_TOP_K = 5
MAX_TOP_K = 10

#: 连续的 CJK 字符（含扩展区）。按**码点区间**判定，不引 `unicodedata`（慢且脆）。
_CJK = r"㐀-䶿一-鿿豈-﫿"

#: 一个「词」的切法：连续 ASCII 字母数字，或连续 CJK。数字与字母连写（``Q3报表``）算一个词。
_WORD_RE = re.compile(rf"[0-9a-zA-Z]+|[{_CJK}]+")


def _tokens(text: str) -> frozenset[str]:
    """把文本切成可检索的 token 集合（**集合** —— 同一 token 重复出现不加分）。

    CJK 串 → 全部字符 bigram **加** 单字；西文/数字串 → 整词小写
    （``AI``/``ai``/``Ai`` 应命中同一份文档）。

    **为什么单字也留**：只出 bigram 时，单字查询（``差``）与任何长 CJK 串都无交集 ——
    ``差旅报销标准`` 出的是 ``差旅/旅报/报销/销标/标准``，没有 ``差``，
    于是「查一个单字」永远零命中，agent 会以为库里没有这条。
    单字召回噪音大，但打分是 token 交集计数：bigram 命中仍比单字多，排序不会被带偏。
    真正靠单字命中的文档，恰好说明它除了这个字没什么可匹配的 —— 把它排前面并不离谱。
    """
    out: set[str] = set()
    for run in _WORD_RE.findall(text):
        if not run:
            continue
        if run[0].isascii():
            out.add(run.lower())
        else:
            out.update(run)  # 单字
            if len(run) > 1:
                out.update(run[i:i + 2] for i in range(len(run) - 1))  # bigram
    return frozenset(out)


def _score(query_tokens: frozenset[str], title_tokens: frozenset[str], body_tokens: frozenset[str]) -> int:
    """打一个整数分。标题命中算 2 分、正文命中算 1 分。

    标题权重更高：中文文档里标题承载的信息密度确实高于正文。
    刻意**不用** TF-IDF 也不加长度归一 —— 那要引入语料统计，跨调用状态会让
    「同输入同输出」这条判据失效。整数分够排序用。
    """
    return sum(2 for t in query_tokens if t in title_tokens) + sum(
        1 for t in query_tokens if t in body_tokens
    )


def knowledge_search(payload: dict[str, Any]) -> dict[str, Any]:
    """在 ``knowledge`` 里按 ``query`` 检索，返回按相关度降序的 top_k 切片。

    :param payload: ``{"query": str, "knowledge": [{"doc_id","title","text"}, ...], "top_k": int?}``

    :returns: ``{"hits": [{"doc_id","title","text","score"}...]}``；零命中时 ``hits`` 为空列表
        （**不是** error —— 「库里没有」是合法答案，见模块 docstring）。
    """
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return make_error(ErrorCode.INVALID_INPUT, "query 不能为空")

    knowledge = payload.get("knowledge", [])
    if not isinstance(knowledge, list):
        return make_error(ErrorCode.INVALID_INPUT, "knowledge 必须是列表")

    top_k = payload.get("top_k", DEFAULT_TOP_K)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        return make_error(ErrorCode.INVALID_INPUT, f"top_k 必须是正整数（1..{MAX_TOP_K}）")
    top_k = min(top_k, MAX_TOP_K)  # 超上限截断而非报错：agent 要多了是它的选择，不该掀桌

    query_tokens = _tokens(query)
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, doc in enumerate(knowledge):
        if not isinstance(doc, dict):
            continue  # 坏文档跳过而非中断：一条脏数据不该让整次检索失败
        title = str(doc.get("title", ""))
        text = str(doc.get("text", ""))
        score = _score(query_tokens, _tokens(title), _tokens(text))
        if score > 0:  # 零分不入榜：没有 token 命中的文档对 agent 是纯噪音
            scored.append((score, index, doc))

    # 全序 tiebreak（-score, 入库下标）：分数相同的文档按知识库原顺序返回，
    # 保证输出不依赖 dict/集合迭代序 —— 这是「逐字节一致」的前提。
    scored.sort(key=lambda item: (-item[0], item[1]))
    return {
        "hits": [
            {"doc_id": doc.get("doc_id"), "title": doc.get("title"), "text": doc.get("text"), "score": score}
            for score, _index, doc in scored[:top_k]
        ]
    }
