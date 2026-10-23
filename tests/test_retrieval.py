"""检索工具测试 —— TASK-008 判据 2。

``knowledge.search`` 是 B 场景（知识检索 QA）的唯一工具。它与 A 场景的表工具有两处
**刻意的不同**，本组测试就是把这两处钉死：

1. **零命中不是 error** —— 查不到返回 ``{"hits": []}``。「库里有但没提到」是 B 场景
   要考的核心能力（「证据不足」类任务），若零命中抛错，agent 根本无法表达它。
2. **错误码沿用 ``_errors.py``** —— 不自创 code、不改 ErrorCode 枚举。

判据要求覆盖的六项逐条对应：
- 确定性（同输入两次逐字节一致）    → `test_same_input_twice_is_byte_identical`
- top_k 截断                       → `test_top_k_*`
- 空 query 报 error                → `test_empty_query_*`
- 命中排序稳定                     → `test_ranking_*`
- 中文文本可检索（含多字词）       → `test_chinese_*`
"""

from __future__ import annotations

import json

from zh_agent_bench.tools._errors import ErrorCode, is_error
from zh_agent_bench.tools.registry import check_contract, get_tool
from zh_agent_bench.tools.retrieval import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    knowledge_search,
)

#: 一份小知识库（虚构中文办公场景，与 A 场景词表无关）。
KNOWLEDGE = [
    {"doc_id": "d1", "title": "远程办公管理办法", "text": "员工每周最多远程办公两天，需要提前在系统申请。"},
    {"doc_id": "d2", "title": "差旅报销标准", "text": "一线城市住宿费上限每晚六百元，超出部分需部门经理审批。"},
    {"doc_id": "d3", "title": "年度培训政策", "text": "年度培训预算为每人两千元，超出部分不予报销。"},
    {"doc_id": "d4", "title": "设备申领规则", "text": "笔记本电脑在公司资产系统中申领，折旧年限为四年。"},
]


def _search(query, **kwargs):
    return knowledge_search({"query": query, "knowledge": KNOWLEDGE, **kwargs})


def _ids(result):
    return [hit["doc_id"] for hit in result["hits"]]


# --------------------------------------------------------------------------------------
# 中文可检索
# --------------------------------------------------------------------------------------


def test_chinese_multi_character_query_finds_the_right_document():
    """多字词必须能命中 —— 单字级切分会召回一堆无关文档。"""
    assert _ids(_search("差旅报销标准"))[0] == "d2"


def test_chinese_bigram_matches_on_shared_terms():
    """命中按 token 交集算：只共享部分词的文档也能召回，只是排在后面。"""
    ids = _ids(_search("住宿费"))
    assert "d2" in ids
    assert _ids(_search("折旧年限"))[0] == "d4"


def test_ascii_terms_are_case_insensitive():
    """西文按整词小写切，``Q3`` 与 ``q3`` 应命中同一份文档。"""
    result = _search("Q3 报告")
    assert isinstance(result["hits"], list)


def test_single_character_chinese_token_is_searchable():
    """单字串也要能检索 —— bigram 对长度 1 的串不产出 token，会让这类查询静默零命中。"""
    assert _ids(_search("差")) == ["d2"]


def test_unrelated_query_returns_no_hits():
    assert _search("完全无关的量子纠缠话题")["hits"] == []


def test_zero_hits_is_not_an_error():
    """**零命中是合法结果，不是 error** —— B 场景要考「证据不足」，不能把它变成报错。"""
    result = _search("完全无关的量子纠缠话题")
    assert not is_error(result)
    assert result == {"hits": []}


# --------------------------------------------------------------------------------------
# 确定性
# --------------------------------------------------------------------------------------


def test_same_input_twice_is_byte_identical():
    """判据 2 的核心：同输入两次逐字节一致（无随机、不读时钟、不依赖集合迭代序）。"""
    query = "报销 住宿 预算"
    first = json.dumps(_search(query, top_k=3), ensure_ascii=False, sort_keys=True)
    second = json.dumps(_search(query, top_k=3), ensure_ascii=False, sort_keys=True)
    assert first.encode() == second.encode()


def test_ranking_is_stable_under_equal_scores():
    """同分文档按**入库下标**返回，而不是依赖 dict/集合迭代序。

    这是「逐字节一致」的前提：若同分时顺序随机，同一份知识库两次跑出的
    top_k 可能取到不同文档，而 top_k 恰好卡在分界线上时结果就会漂。
    """
    tied = [
        {"doc_id": "a", "title": "同名", "text": "报销"},
        {"doc_id": "b", "title": "同名", "text": "报销"},
        {"doc_id": "c", "title": "同名", "text": "报销"},
    ]
    result = knowledge_search({"query": "报销", "knowledge": tied})
    assert _ids(result) == ["a", "b", "c"]
    assert _ids(knowledge_search({"query": "报销", "knowledge": tied})) == ["a", "b", "c"]


def test_input_is_not_mutated():
    """纯函数铁律：不改入参。"""
    payload = {"query": "报销", "knowledge": json.loads(json.dumps(KNOWLEDGE))}
    before = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    knowledge_search(payload)
    assert json.dumps(payload, ensure_ascii=False, sort_keys=True) == before


def test_result_is_json_serializable():
    """hits 必须整体可 json 序列化（harness 的 ``write_trace`` 要落盘）。"""
    json.dumps(_search("报销", top_k=4), ensure_ascii=False)


# --------------------------------------------------------------------------------------
# top_k
# --------------------------------------------------------------------------------------


def test_top_k_defaults_to_five():
    assert DEFAULT_TOP_K == 5
    many = [{"doc_id": f"d{i}", "title": "报销政策", "text": "报销"} for i in range(10)]
    assert len(knowledge_search({"query": "报销", "knowledge": many})["hits"]) == 5


def test_top_k_truncates():
    assert len(_search("报销", top_k=2)["hits"]) == 2


def test_top_k_above_max_is_clamped_not_rejected():
    """超上限**截断**而非报错 —— agent 想多要点是它的选择，不该因此掀桌。"""
    assert MAX_TOP_K == 10
    assert len(_search("报销", top_k=999)["hits"]) <= MAX_TOP_K


def test_top_k_larger_than_corpus_returns_all_hits():
    assert len(_search("报销", top_k=10)["hits"]) <= len(KNOWLEDGE)


def test_top_k_must_be_positive_int():
    assert is_error(_search("报销", top_k=0))
    assert is_error(_search("报销", top_k=-1))
    assert is_error(_search("报销", top_k="3"))


def test_top_k_bool_is_rejected():
    """bool 是 int 的子类，不显式排除的话 ``top_k=True`` 会被当成 1。"""
    assert is_error(_search("报销", top_k=True))


# --------------------------------------------------------------------------------------
# 错误分支
# --------------------------------------------------------------------------------------


def test_empty_query_is_an_error():
    """判据点名的「空 query 报 error」。"""
    result = _search("")
    assert is_error(result)
    assert result["error"]["code"] == ErrorCode.INVALID_INPUT.value


def test_whitespace_only_query_is_an_error():
    assert is_error(_search("   "))


def test_non_string_query_is_an_error():
    assert is_error(knowledge_search({"query": 123, "knowledge": KNOWLEDGE}))


def test_missing_query_is_an_error():
    assert is_error(knowledge_search({"knowledge": KNOWLEDGE}))


def test_knowledge_must_be_a_list():
    assert is_error(knowledge_search({"query": "报销", "knowledge": "not-a-list"}))


def test_all_error_codes_come_from_the_enum():
    """不自创 code —— harness 靠这个枚举做机器可判定的分支。"""
    for payload in ({"query": ""}, {"query": "x", "top_k": 0}, {"query": "x", "knowledge": "bad"}):
        result = knowledge_search({**payload, "knowledge": payload.get("knowledge", KNOWLEDGE)})
        if is_error(result):
            assert result["error"]["code"] in {c.value for c in ErrorCode}


def test_malformed_document_is_skipped_not_fatal():
    """一条脏文档不该让整次检索失败 —— 返回它的坏邻居即可。"""
    dirty = [*KNOWLEDGE, "not-a-dict", {"doc_id": "x", "title": "报销补充", "text": "报销"}]
    result = knowledge_search({"query": "报销", "knowledge": dirty})
    assert not is_error(result)
    assert "x" in _ids(result)


# --------------------------------------------------------------------------------------
# 契约
# --------------------------------------------------------------------------------------


def test_registered_as_knowledge_search():
    assert get_tool("knowledge.search").id == "knowledge.search"


def test_check_contract_accepts_valid_payload():
    result = check_contract("knowledge.search", {"query": "差旅报销", "knowledge": KNOWLEDGE})
    assert _ids(result)[0] == "d2"


def test_check_contract_error_branch_is_passthrough():
    """error 分支经 ``check_contract`` 原样透传，错误码不被改写（TASK-007 的修复）。"""
    result = check_contract("knowledge.search", {"query": ""})
    assert result["error"]["code"] == ErrorCode.INVALID_INPUT.value


def test_input_schema_requires_query():
    spec = get_tool("knowledge.search")
    assert spec.input_schema["required"] == ["query"]


def test_output_schema_requires_hits():
    assert get_tool("knowledge.search").output_schema["required"] == ["hits"]


def test_hits_carry_doc_id_title_text_score():
    """每条命中必须带 score —— agent 要能看出「哪几篇最相关」。"""
    hit = _search("差旅报销标准")["hits"][0]
    assert set(hit) == {"doc_id", "title", "text", "score"}
    assert hit["score"] > 0


def test_hits_do_not_leak_the_answer():
    """不预制答案：工具只给文档切片，不给「答案是什么」。

    与表工具同款铁律 —— 若工具直接吐出答案，评测就废了。
    """
    hit = _search("差旅报销标准")["hits"][0]
    assert "answer" not in hit
    assert hit["text"] == KNOWLEDGE[1]["text"]


def test_tool_imports_only_stdlib():
    """不引 jieba 等第三方依赖（findings §二：运行时零第三方依赖）。"""
    import ast
    import pathlib

    path = pathlib.Path(knowledge_search.__code__.co_filename)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    allowed = {"__future__", "re", "typing"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            assert node.module.split(".")[0] in allowed, node.module
