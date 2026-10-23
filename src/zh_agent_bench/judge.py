"""客观判分器 —— M1 的命门。

照 ``tasks/schema.py`` 的 :class:`AnswerSpec` 实现**四种比较算子，不自创第五种**：

======================  ========================================================
comparison              语义
======================  ========================================================
``exact``               字符串逐字相等；``normalization == strip`` 时先 ``str.strip()``
``decimal_exact``       双方解析为 ``Decimal`` 后精确相等
``ordered_list_exact``  长度一致 + 按序逐元素相等
``json_canonical``      双方解析为 JSON 对象后 ``sort_keys=True`` 再比，**不比原文**
======================  ========================================================

上面四种是 **A 场景**（客观题）的全部，与 ``tasks/schema.py`` 的 :class:`Comparison` 一一对应。

**B 场景**（``Comparison.JUDGE``）走另一条路：rubric 逐条打分（:func:`_judge_by_rubric`）。
它不是「第五个比较算子」—— 它压根不做比较，是 LLM-as-judge + 程序聚合。三指标：

======  ==========================================================================
``G``   目标达成。``Σ wᵢ·sᵢ / Σ wᵢ``，逐点由 judge 给 0/0.5/1，**加权求和是程序干的**
``F``   忠实度。答案里每个实质 claim 判「支持/矛盾/证据不足」，支持者加权占比。
        **空答案 F = 不适用**，绝不是满分（没说话谈不上忠实）
``T``   工具协议合规。从轨迹机械算单次调用合法率（不调工具 ⇒ T 不适用）
======  ==========================================================================

**铁律：判分器绝不能抛异常。** agent 的答案可能是 ``None``、空字符串、乱码、类型完全不对。
判分器遇到烂答案只能返回 ``correct=False`` 加一句原因 —— 一崩就整批评测挂掉，
而那恰恰是最需要判分器工作的场景。所以本模块内所有解析都包在 try 里，
异常一律转成 ``reason``。这条对 LLM 侧同样成立：judge 挂掉、网络断、模型返回乱码，
都只降级成「不适用 + 原因」，绝不冒泡。
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any

from . import llm_client
from .tasks.schema import AnswerSpec, Comparison, Normalization


# --------------------------------------------------------------------------------------
# 规范化（各自独立，一个失败不影响另一个）
# --------------------------------------------------------------------------------------


def _normalize_text(value: Any, spec: AnswerSpec) -> str | None:
    """文本规范化。失败返回 None。"""
    if isinstance(value, str):
        return value.strip() if spec.normalization is Normalization.STRIP else value
    if value is None:
        return None
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        # agent 把答案写成数字也算合法（如 研发部 不可能是数字，但金额可能是）
        return str(value).strip() if spec.normalization is Normalization.STRIP else str(value)
    return None


def _normalize_decimal(value: Any) -> Decimal | None:
    """把 str/int/Decimal 统一成 Decimal。**float 也接**——agent 很可能算出浮点数，
    判分器要宽容地接住它再精确比较，而不是因为收到 float 就崩。"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        # 先转 str 再进 Decimal：Decimal(0.1) 是精确二进制值，
        # 而 str(0.1) 是「0.1」——后者才符合人写的十进制意图。
        return Decimal(str(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return Decimal(text)
        except InvalidOperation:
            return None
    return None


def _normalize_list(value: Any) -> list[Any] | None:
    """只接受 list/tuple。逐元素做 strip（有序列表答案都是短文本）。"""
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.append(item.strip() if isinstance(item, str) else item)
        return out
    return None


def _normalize_json(value: Any) -> Any:
    """把值变成「可 canonical 比较」的对象。失败返回哨兵对象（不是 None，
    因为 ``None`` 本身是合法的 JSON 值）。"""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return _Unparseable(value)
    return value


class _Unparseable:
    """解析失败的哨兵。与任何真实 JSON 值都不会相等，故必然判错。"""

    __slots__ = ("raw",)

    def __init__(self, raw: Any) -> None:
        self.raw = raw

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Unparseable) and self.raw == other.raw

    def __repr__(self) -> str:
        return f"_Unparseable({self.raw!r})"


# --------------------------------------------------------------------------------------
# 四个比较算子
# --------------------------------------------------------------------------------------


def _cmp_exact(answer: Any, truth: Any, spec: AnswerSpec) -> tuple[bool | None, str]:
    norm_answer = _normalize_text(answer, spec)
    if norm_answer is None:
        return None, f"答案不是可比较的文本（收到 {type(answer).__name__}）"
    norm_truth = _normalize_text(truth, spec)
    if norm_truth is None:
        return None, f"ground_truth 不是可比较的文本（收到 {type(truth).__name__}）"
    return norm_answer == norm_truth, "逐字相等" if norm_answer == norm_truth else "文本不相等"


def _cmp_decimal_exact(answer: Any, truth: Any, spec: AnswerSpec) -> tuple[bool | None, str]:
    norm_answer = _normalize_decimal(answer)
    if norm_answer is None:
        return None, f"答案无法解析为十进制数（收到 {answer!r}）"
    norm_truth = _normalize_decimal(truth)
    if norm_truth is None:
        return None, f"ground_truth 无法解析为十进制数（收到 {truth!r}）"
    ok = norm_answer == norm_truth
    return ok, ("十进制精确相等" if ok else f"数值不等：答案 {norm_answer} vs 真值 {norm_truth}")


def _cmp_ordered_list_exact(answer: Any, truth: Any, spec: AnswerSpec) -> tuple[bool | None, str]:
    norm_answer = _normalize_list(answer)
    if norm_answer is None:
        return None, f"答案不是列表（收到 {type(answer).__name__}）"
    norm_truth = _normalize_list(truth)
    if norm_truth is None:
        return None, f"ground_truth 不是列表（收到 {type(truth).__name__}）"
    if len(norm_answer) != len(norm_truth):
        return False, f"长度不等：答案 {len(norm_answer)} 项 vs 真值 {len(norm_truth)} 项"
    if norm_answer == norm_truth:
        return True, "按序逐元素相等"
    for i, (a, t) in enumerate(zip(norm_answer, norm_truth)):
        if a != t:
            return False, f"第 {i} 项不等：答案 {a!r} vs 真值 {t!r}"
    return False, "逐项相等但整体不等"


def _cmp_json_canonical(answer: Any, truth: Any, spec: AnswerSpec) -> tuple[bool | None, str]:
    norm_answer = _normalize_json(answer)
    norm_truth = _normalize_json(truth)
    if isinstance(norm_answer, _Unparseable):
        return False, f"答案无法解析为 JSON：{norm_answer.raw!r}"
    if isinstance(norm_truth, _Unparseable):
        return None, f"ground_truth 无法解析为 JSON：{norm_truth.raw!r}"
    try:
        # sort_keys=True：键序无关；ensure_ascii=False：中文不比转义形式
        a = json.dumps(norm_answer, sort_keys=True, ensure_ascii=False)
        t = json.dumps(norm_truth, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        return None, f"JSON 序列化失败：{exc}"
    return a == t, "JSON canonical 相等" if a == t else "JSON 内容不等"


_COMPARATORS = {
    Comparison.EXACT: _cmp_exact,
    Comparison.DECIMAL_EXACT: _cmp_decimal_exact,
    Comparison.ORDERED_LIST_EXACT: _cmp_ordered_list_exact,
    Comparison.JSON_CANONICAL: _cmp_json_canonical,
}


# --------------------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------------------


def judge(answer: Any, ground_truth: dict, answer_spec: AnswerSpec, *,
          task: Any = None, trajectory: Any = None, final_snapshot: Any = None,
          **judge_kwargs: Any) -> dict:
    """判一条答案。

    返回 ``{"correct": bool, "reason": str, "normalized_answer": Any, "normalized_truth": Any}``

    **任何情况下都不抛异常**：答案缺失/类型不对/无法解析，一律 ``correct=False``。
    唯一会 ``raise`` 的是「``answer_spec`` 自己不合法」（比较算子不在枚举里）——
    那是数据集的 bug，不是 agent 的问题，必须让调用方当场发现。

    :param task: **仅 B 场景需要**。``comparison is JUDGE`` 时 rubric 与知识库都在
        :class:`Task` 上（rubric 是逐条评分点、知识库是 judge 核证据的原文），两个参数
        ``ground_truth``/``answer_spec`` 里都没有。A 场景不传，行为与从前逐字相同。
    :param trajectory: **仅 B 场景需要**。harness 的 ``tool_calls`` 全量记录，
        供 :func:`_tool_compliance` 机械算 ``T``、:func:`_evidence_coverage` 算证据覆盖率。
    :param final_snapshot: **仅 C 场景需要**。agent 跑完后沙箱的 :func:`sandbox.snapshot`。
        C 场景的答案在**世界状态**里不在文本里，没有它就无判可判（见 :func:`_judge_orchestration`）。
    :param judge_kwargs: **judge 注入**（``model`` / ``client_kwargs`` 等），转发给 B/C
        两条 LLM 路径。**一个都不传时请求体、模型、端点与从前逐字节相同** ——
        A 场景根本不看它，那几个比较算子全是纯函数的性质。
    """
    if answer_spec.comparison is Comparison.ORCHESTRATION:
        return _judge_orchestration(task, answer, final_snapshot=final_snapshot,
                                    trajectory=trajectory, **judge_kwargs)
    if answer_spec.comparison is Comparison.JUDGE:
        return _judge_by_rubric(task, answer, trajectory=trajectory, **judge_kwargs)

    try:
        truth = ground_truth.get("answer") if isinstance(ground_truth, dict) else None
        comparator = _COMPARATORS.get(answer_spec.comparison)
        if comparator is None:
            raise ValueError(f"未实现的比较算子: {answer_spec.comparison!r}")

        verdict, reason = comparator(answer, truth, answer_spec)
        return {
            "correct": bool(verdict),
            "reason": reason,
            "normalized_answer": answer,
            "normalized_truth": truth,
        }
    except ValueError:
        raise
    except Exception as exc:  # 兜底：判分器绝不能因烂答案崩掉
        return {
            "correct": False,
            "reason": f"判分时发生意外错误：{type(exc).__name__}: {exc}",
            "normalized_answer": answer,
            "normalized_truth": None,
        }


def judge_task(task: Any, answer: Any, trajectory: Any = None, *,
               final_snapshot: Any = None, **judge_kwargs: Any) -> dict:
    """便捷封装：直接从 :class:`Task` 取 ground_truth 与 answer_spec 判分。

    :param judge_kwargs: 转发给 :func:`judge`，用于把 judge 换成别的模型 / 端点 / 密钥
        （``--judge-model`` 等开关走这条路）。**一个都不传时逐字节等同于从前。**
    """
    return judge(answer, task.ground_truth, task.answer_spec,
                 task=task, trajectory=trajectory, final_snapshot=final_snapshot,
                 **judge_kwargs)


# ======================================================================================
# B 场景：rubric 驱动的中文 LLM-as-judge
#
# 「judge 只做逐点判断、聚合由程序做」是这条路的全部意义（gpt-6-astra P0-1）：
# 让模型直接给一个总分，丢分点就永远查不出来了，而分数也没法复现。
# ======================================================================================

#: judge 用的模型（与被评 agent 分离 —— 同模型自评有偏置，偏差实验是后续单的事，
#: 但模型 id 必须记下来，将来要做归因时至少分得清是谁判的）。
JUDGE_MODEL = llm_client.DEFAULT_MODEL

#: judge 请求温度。0 = 尽量确定性（LLM 侧仍非严格确定，但把方差压到最小）。
JUDGE_TEMPERATURE = 0.0

#: 主榜口径的 G 门槛。**冻结在代码里**：阈值一旦按榜单成绩回调，分数就不再可比。
PASS_THRESHOLD_G = 0.8

#: claim 的三种判定。中文原词，判分报告直接可读，不用来回翻译。
SUPPORTED = "支持"
CONTRADICTED = "矛盾"
INSUFFICIENT = "证据不足"

#: judge 的三档取值白名单。不在白名单里的分值一律记 0 并写明原因 —— 「模型给了 0.7」
#: 这种输出静默接受会让分数悄悄漂掉，而 rubric 的 0/0.5/1 是可判性校验钉死的。
LEGAL_POINT_SCORES = (0.0, 0.5, 1.0)

#: **声明型**任务类型：正确答案本身包含「库里没有 X / 我答不了，先澄清 Y」。
#: 字符串取值来自 ``tasks/qa.py`` 的 ``TASK_TYPES``，**不 import 那个模块** ——
#: judge 是评测运行期热路径，qa 会拉进生成期依赖（且 qa 自己也 import 这里的
#: ``aggregate_rubric``，import 回去成环）。两边由 ``tests/test_judge_b.py`` 钉住一致。
DECLARATIVE_TASK_TYPES = frozenset({"证据不足", "不可回答"})


def _docs_by_id(task: Any) -> dict[str, dict[str, Any]]:
    """任务知识库按 ``doc_id`` 建索引。**取不到就返回空 dict**，不抛 ——
    judge 会在 prompt 里发现「证据原文为空」，那条路已由 :func:`_judge_by_rubric` 降级。"""
    try:
        initial = getattr(task, "initial_data", None) or {}
        docs = initial.get("knowledge") if isinstance(initial, dict) else None
        if not isinstance(docs, list):
            return {}
        return {d["doc_id"]: d for d in docs if isinstance(d, dict) and "doc_id" in d}
    except Exception:
        return {}


def _is_empty_answer(answer: Any) -> bool:
    """空答案判定。``None`` / ``""`` / 纯空白 / 空容器都算「没回答」。

    只有「没回答」这一种情况的 F 是**不适用**：一份空白答案对知识库的忠实度不是满分，
    恰恰相反 —— 它什么断言都没做，所以也就无从判它是否被证据支持。
    """
    if answer is None:
        return True
    if isinstance(answer, str):
        return not answer.strip()
    if isinstance(answer, (list, tuple, dict, set)):
        return len(answer) == 0
    return False


def build_judge_prompt(task: Any, answer: Any) -> list[dict[str, Any]]:
    """渲染 B 场景 judge 的 messages。**逐条 rubric 独立成段**，外加原文与判分说明。

    为什么不能整段塞进去让模型「打个总分」（P0-1）：总分不可回溯到丢分点、跨次不可复现、
    还天然奖励「把参考答案复述一遍」—— 复述与答对在总分上长得一模一样。

    证据原文**必须**进 prompt：judge 判「支持/矛盾/证据不足」的唯一依据就是原文，
    不给它原文它只能凭记忆编，那是拿模型的记忆当知识库。

    ``ground_truth["example_answer"]`` **不进 prompt**：它只是「一个合格答案长什么样」，
    拿它做判分依据就是 P0-1 点名要消灭的那种判法（复述即满分）。
    """
    rubric = getattr(task, "rubric", None) or []
    docs = _docs_by_id(task)

    sections: list[str] = []
    for i, point in enumerate(rubric, start=1):
        cid = point.get("criterion_id", f"#{i}")
        block = [
            f"### 评分点 {i}（criterion_id = {cid}）",
            f"要求：{point.get('requirement', '')}",
            f"权重：{point.get('weight', '')}"
            + ("（关键点：完全没答到时整题判失败，不得被其他点的高分抵消）" if point.get("critical") else ""),
            f"半分规则：{point.get('partial_rule', '')}",
        ]
        alts = point.get("accepted_alternatives") or []
        if alts:
            # 列出来是为了消歧义（模型会因为措辞不同就判 0），不是为了给「照抄即可满分」的路子。
            block.append("以下表述视为等价，同样给满分：" + "；".join(str(a) for a in alts))
        ev_ids = point.get("evidence_ids") or []
        if ev_ids:
            block.append("证据原文（判分依据，以原文为准）：")
            for doc_id in ev_ids:
                doc = docs.get(doc_id)
                if doc is None:
                    block.append(f"- [{doc_id}]（本任务未提供该文档原文）")
                else:
                    block.append(f"- [{doc_id}] {doc.get('title', '')}：{doc.get('text', '')}")
        else:
            block.append("证据原文：本评分点不依赖具体文档（针对「有没有混入无关内容」这类要点）。")
        sections.append("\n".join(block))

    answer_text = "" if _is_empty_answer(answer) else (
        answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
    )

    system = (
        "你是中文知识检索任务的评分员。给你一份评分细则和一份待评答案，"
        "你只负责逐条判断答案有没有做到每个评分点，不要给总分。\n"
        "判分原则：\n"
        "1. 只依据「证据原文」判断，不要用你自己的知识补充或纠正原文；\n"
        "2. 依据原文逐条打 0 / 0.5 / 1，0.5 的触发情形以该评分点的「半分规则」为准，"
        "没有写半分规则的评分点只能是 0 或 1；\n"
        "3. 措辞不同但意思等价，算答到；\n"
        "4. 答案里没有明确写出来的内容一律不得算答到（不替答案脑补）；\n"
        "5. **claims 只准收「可核事实断言」** —— 能拿证据原文判断真假的陈述。"
        "下面三类句子**一律不许抽成 claim**：\n"
        "   ① 行动建议/祈使句：「建议…」「应当…」「需要索取…」「请联系…」"
        "「可以先向…确认」。对下一步的提议，证据原文里本来就查不到；\n"
        "   ② 关于证据覆盖范围的元陈述：「库里的文档没有提到 X」"
        "「入库的是节选版」「只说明了…不包含…」。这是对**证据本身**的描述，"
        "不是对世界的断言，不要拿检索结果去证它；\n"
        "   ③ 元话语与客套：「这个问题现在还答不了」「我需要先确认两点」"
        "「请告诉我…我再去找」这类开场白、收尾话。\n"
        "   判据一句话：**这句话错了，世界会不会跟着变？** 不会 ⇒ 不是可核事实断言。\n"
        "   抽不到可核事实断言时 claims 给空数组，这是合法输出，**不扣分**。\n"
        "只输出一个 JSON 对象，不要输出任何解释文字或 Markdown 代码块标记。"
    )
    user = (
        f"## 问题\n{getattr(task, 'instruction', '')}\n\n"
        f"## 待评答案\n{answer_text or '（空）'}\n\n"
        f"## 评分细则（共 {len(sections)} 个评分点）\n\n"
        + "\n\n".join(sections)
        + "\n\n## 你要输出的 JSON 结构\n"
        '{"scores": {"<criterion_id>": 0 | 0.5 | 1, ...}, '
        '"claims": [{"claim": "<答案里的一句实质断言，原文照抄>", '
        '"verdict": "支持 | 矛盾 | 证据不足", "evidence_id": "<依据的 doc_id 或空串>"}], '
        '"notes": "<一句话总评>"}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _ask_judge(task: Any, answer: Any, **kwargs: Any) -> tuple[dict[str, Any] | None, str]:
    """发一次 judge 请求。返回 ``(payload, reason)``；失败时 ``payload=None``。

    :func:`llm_client.chat` 本身不抛异常（网络/HTTP/坏 JSON 都转成 ``{"error": ...}``），
    这里再兜一层：测试里 mock 的 ``chat`` 可能直接抛，两边都不能让判分崩掉。
    """
    try:
        response = llm_client.chat(
            build_judge_prompt(task, answer),
            model=kwargs.get("model") or JUDGE_MODEL,
            temperature=kwargs.get("temperature", JUDGE_TEMPERATURE),
            **kwargs.get("client_kwargs", {}),
        )
    except Exception as exc:  # mock 抛异常 / 未来的 SDK 变更 —— 一样只降级
        return None, f"judge 调用失败：{type(exc).__name__}: {exc}"
    if not isinstance(response, dict) or "error" in response:
        err = response.get("error") if isinstance(response, dict) else None
        detail = err.get("message") if isinstance(err, dict) else str(response)
        return None, f"judge 调用失败：{detail}"
    message = llm_client.first_message(response)
    if message is None:
        return None, "judge 返回没有 choices[0].message"
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return None, "judge 返回 content 为空"
    return _loads_strict(content), "judge 返回的不是合法 JSON"


def _loads_strict(content: str) -> dict[str, Any] | None:
    """从模型回复里抠出 JSON 对象。

    模型爱把 JSON 包在 ```json … ``` 里。直接 ``json.loads`` 会打在那三个反引号上，
    判分器就该把一份完全正确的打分扔了 —— 所以先剥围栏再剥「前面那段寒暄」。
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _collect_scores(
    rubric: list[dict[str, Any]], raw: Any, notes: list[str]
) -> dict[str, float]:
    """把 judge 的 ``scores`` 收敛成「每个 criterion_id 都有 0/0.5/1」。

    三种残缺一律补 0 **并写明原因**（判据 1 的 ②③）：judge 漏点 = 没提到（漏判不得当
    满分）；分值非法 = 不认（静默接受会让分数悄悄漂）；整块不是 dict = 整份作废。
    宁可漏判成 0，也不要猜。
    """
    scores: dict[str, float] = {}
    if not isinstance(raw, dict):
        notes.append("judge 未返回 scores 对象，全部评分点按 0 记")
        raw = {}
    for point in rubric:
        cid = point.get("criterion_id")
        value = raw.get(cid)
        # bool 必须先挡掉：``float(True) == 1.0``，模型回了个 ``true`` 会被当成满分。
        if isinstance(value, bool) or value is None:
            notes.append(f"{cid}: judge 未给出分值（收到 {value!r}），按 0 记")
            scores[cid] = 0.0
            continue
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            notes.append(f"{cid}: 分值 {value!r} 不是数字，按 0 记")
            scores[cid] = 0.0
            continue
        if number not in LEGAL_POINT_SCORES:
            notes.append(f"{cid}: 分值 {value!r} 不在 0/0.5/1 内，按 0 记")
            scores[cid] = 0.0
            continue
        scores[cid] = number
    return scores


def _collect_claims(raw: Any, notes: list[str]) -> list[dict[str, Any]]:
    """收敛 claim 列表。verdict 认中文三档，认不出的归「证据不足」（最保守的一档）。"""
    if not isinstance(raw, list):
        return []
    claims: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            notes.append(f"claim 格式非法（{type(item).__name__}），已丢弃")
            continue
        claim = item.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            notes.append("claim 缺 claim 文本，已丢弃")
            continue
        verdict = item.get("verdict")
        if verdict not in (SUPPORTED, CONTRADICTED, INSUFFICIENT):
            notes.append(f"claim {claim!r} 的 verdict {verdict!r} 不合法，按「{INSUFFICIENT}」记")
            verdict = INSUFFICIENT
        evidence_id = item.get("evidence_id")
        claims.append({
            "claim": claim,
            "verdict": verdict,
            "evidence_id": evidence_id if isinstance(evidence_id, str) else "",
        })
    return claims


def _faithfulness(claims: list[dict[str, Any]]) -> tuple[float | None, int, int]:
    """``F = 支持 claim 数 / claim 总数``。无 claim ⇒ ``None``（不适用），不是 1.0。

    没提取出任何 claim 是最常见的一种：答案全是空话、没有可核的实质断言。这时判 1.0
    等于「什么都能满分」，判 0 等于「说的每句话都是假的」—— 两个都不对，只能是不适用。
    """
    total = len(claims)
    if total == 0:
        return None, 0, 0
    supported = sum(1 for c in claims if c["verdict"] == SUPPORTED)
    contradicted = sum(1 for c in claims if c["verdict"] == CONTRADICTED)
    unsupported = sum(1 for c in claims if c["verdict"] == INSUFFICIENT)
    return supported / total, contradicted, unsupported


def _tool_compliance(trajectory: Any) -> tuple[float | None, int, int, int]:
    """``T`` 从轨迹**机械**算，不问 LLM。

    :returns: ``(T, 调用总数, 非法调用数, 非法任务比例分母)``。轨迹为空 ⇒ ``T=None``
        （不适用）。**没调用工具不等于协议违规** —— 有些题本来就不该调，判成 0 会把
        「选择不调」和「乱调」混成同一件事。
    """
    if not isinstance(trajectory, list) or not trajectory:
        return None, 0, 0, 0
    illegal = 0
    for call in trajectory:
        if isinstance(call, dict) and "error" in call:
            illegal += 1
    total = len(trajectory)
    return (total - illegal) / total, total, illegal, total


def _evidence_coverage(rubric: list[dict[str, Any]], trajectory: Any) -> float | None:
    """关键证据覆盖率：rubric 要的 ``evidence_ids`` 有多少真被 agent 检索到了。

    **独立诊断指标，不混进 T**（判据 §2.5 第 6 条）：检索没查到证据和「调用违反协议」
    是两回事，合成一个数就谁也说不清了。
    """
    wanted = {
        doc_id
        for point in rubric
        for doc_id in (point.get("evidence_ids") or [])
    }
    if not wanted:
        return None
    retrieved: set[str] = set()
    # 轨迹可能压根不是列表（result 落盘被截断、调用方传了别的类型）—— 迭代它会抛，
    # 判分器这一层的铁律是不抛，所以先认类型。
    if not isinstance(trajectory, list):
        return 0.0
    for call in trajectory:
        result = call.get("result") if isinstance(call, dict) else None
        hits = result.get("hits") if isinstance(result, dict) else None
        if not isinstance(hits, list):
            continue
        for hit in hits:
            if isinstance(hit, dict) and isinstance(hit.get("doc_id"), str):
                retrieved.add(hit["doc_id"])
    return len(wanted & retrieved) / len(wanted)


def _judge_by_rubric(task: Any, answer: Any, **kwargs: Any) -> dict[str, Any]:
    """B 场景判分：逐点 rubric 打分 → 程序聚合出 G / F / T。

    返回 ``{"correct", "G", "F", "T", "scores", "capped", "reason", "claims",
    "evidence_coverage", "model", "n_contradicted", "n_unsupported"}``。

    **任何情况下都不抛异常。** judge 挂掉 ⇒ 全部记 0 + 原因（判据 1 的 ④）：
    一条判不出分的题必须显式显示成「0 分且原因如此」，而不是让整批评测挂掉。
    """
    notes: list[str] = []
    rubric = getattr(task, "rubric", None) or []
    trajectory = kwargs.get("trajectory")
    model = kwargs.get("model") or JUDGE_MODEL

    # model 已经在上面归一过一次，展开 kwargs 时必须把它摘掉 —— 否则 model 传两次。
    # （这行以前不会炸：调用方从不下发 model。现在 judge 注入是正式入口了。）
    payload, failure = _ask_judge(task, answer, model=model,
                                  **{k: v for k, v in kwargs.items()
                                     if k not in ("trajectory", "model")})
    scores = {p["criterion_id"]: 0.0 for p in rubric}
    claims: list[dict[str, Any]] = []
    if payload is None:
        notes.append(failure)
    else:
        scores = _collect_scores(rubric, payload.get("scores"), notes)
        claims = _collect_claims(payload.get("claims"), notes)

    # 聚合交给 qa.aggregate_rubric（与生成期同一份加权 + critical 封顶逻辑）。
    # 本模块不 import qa：qa 构造器会 import 一堆生成期依赖，而 judge 是评测运行期
    # 的热路径，且 A 场景连包都不该碰。聚合失败（键不齐 / 分值非法）已被上面收敛过，
    # 这里再兜一层，绝不让它冒到调用方。
    g, capped = 0.0, False
    try:
        from .tasks.qa import aggregate_rubric

        g, capped = aggregate_rubric(rubric, scores)
    except Exception as exc:
        notes.append(f"rubric 聚合失败：{type(exc).__name__}: {exc}")

    f_score, contradicted, unsupported = _faithfulness(claims)
    t_score, n_calls, n_illegal, _ = _tool_compliance(trajectory)

    # 任务类型只用来决定 ``unsupported`` 的严格程度，取不到就按事实型（最严）走。
    initial = getattr(task, "initial_data", None)
    task_type = initial.get("任务类型", "") if isinstance(initial, dict) else ""
    if not isinstance(task_type, str):
        task_type = ""

    if _is_empty_answer(answer):
        f_score = None
        notes.append("答案为空：F 不适用（空答案不因「什么都没说」而算忠实）")
    coverage = _evidence_coverage(rubric, trajectory)

    # 主榜口径：关键点满足 + G ≥ 阈值 + 无矛盾/无依据断言 + 工具协议合规。
    # 「关键点满足」已由 aggregate_rubric 的 critical 封顶表达（G>capped 即关键点漏了）。
    #
    # ``f_score is not None`` 也在条件里：F 不适用（空答案、没抽出任何 claim）不是「满分」，
    # 是「判不出来」。判不出来不能算成功 —— 否则一份只有 rubric 套话、零可核断言的答案
    # 能靠 G 满分白拿主榜成绩，忠实度这一栏形同虚设。
    #
    # **矛盾（contradicted）任何类型都不放过**：与证据原文直接冲突的断言在哪种任务里都是硬伤。
    #
    # **证据不足（unsupported）按任务类型分开判**（TASK-011 实测修的）：
    # 声明型任务（「证据不足」/「不可回答」）的正确行为本身就是**说「库里没有 / 我答不了」**
    # —— 答案里必然存在「拿不到证据」的句子，那是得分点不是污点。用 ``unsupported == 0``
    # 卡它等于**用正确答案的固有特征去否决正确答案**，实测会把 G=1.0 的满分答案全部毙掉。
    # 事实型仍卡 ``unsupported == 0``：那里每个数字都该有出处。
    correct = bool(
        g >= PASS_THRESHOLD_G
        and not capped
        and f_score is not None
        and contradicted == 0
        and (t_score is None or t_score == 1.0)
        and (unsupported == 0 or task_type in DECLARATIVE_TASK_TYPES)
    )

    return {
        "correct": correct,
        "G": g,
        "F": f_score,
        "T": t_score,
        "scores": scores,
        "capped": capped,
        "reason": "；".join(notes) if notes else "逐条 rubric 判定完成",
        "claims": claims,
        "evidence_coverage": coverage,
        "model": model,
        "n_contradicted": contradicted,
        "n_unsupported": unsupported,
        "task_type": task_type,
        "diagnostics": {"n_calls": n_calls, "n_illegal_calls": n_illegal},
    }


# ======================================================================================
# C 场景：沙箱最终状态（delta 三段）+ 回复质量
#
# 「答案不在文本里，在世界状态里」：A 场景拿返回值比、B 场景拿证据原文比，
# C 场景只能拿 agent 走完之后的沙箱快照比。只判回复文本 = 退回 B 场景，
# 一个把沙箱改得一塌糊涂却嘴上说得漂亮的 agent 会满分。
# ======================================================================================

#: C 场景的 G 门槛。**与 B 的 ``PASS_THRESHOLD_G`` 分开冻结**：两个场景的 rubric
#: 构成不同（编排点 + 回复点 vs 纯检索点），共用一个值等于假装它们一样准；
#: 同样地，一旦按某个榜单的成绩回调这个阈值，C 场景的分数就不再可比。
PASS_THRESHOLD_C = 0.8

#: C 场景归一化输入里**不参与 delta 比对**的字段。
#:
#: 退款会改 ``status``、改地址会改 ``address``，其余字段（金额、客户、商品、时间）
#: 一律应当原样。``created_at`` 尤其要排除：它是环境出厂就带着的元数据，
#: 判「该保持原样的东西没变」时它本就该不变，所以留在排除表里只是省一次比较，
#: 但 ``amount_cents`` 绝对不能排除 —— 「顺手改了金额」正是要抓的那类越权改动。
_INVARIANT_EXCLUDED_FIELDS = frozenset({"created_at"})


def _sandbox_facts(task: Any, final_snapshot: Any, delta: dict[str, Any]) -> str:
    """给 C 场景 judge 的「沙箱关键事实」：只给 delta 牵涉到的那几个对象。

    **为什么不全量塞**：8 单 + 4 商品 + 2 券的全量快照会让 prompt 变长且信息稀释，
    而判分只需看得出「哪些单该是什么状态」。**为什么不给 example_answer**：
    拿参考答案当判分依据就是 B 场景 P0-1 点名要消灭的判法（复述即满分）。
    """
    initial = getattr(task, "initial_data", None)
    initial_snap = initial.get("sandbox") if isinstance(initial, dict) else None
    if not isinstance(final_snapshot, dict):
        return "（拿不到沙箱最终快照）"

    facts: list[str] = []
    for row in list(delta.get("orders") or []) + list(delta.get("must_not_change") or []):
        token = row.get("order_id")
        if not token:
            continue
        was = next((o for o in (initial_snap or {}).get("orders", [])
                    if o.get("order_id") == token), None)
        now = next((o for o in final_snapshot.get("orders", [])
                    if o.get("order_id") == token), None)
        facts.append(
            f"- 订单 {token}：初始 {json.dumps(was, ensure_ascii=False)}"
            f" → 最终 {json.dumps(now, ensure_ascii=False)}"
        )
    for row in list(delta.get("coupons") or []) + list(delta.get("must_not_change") or []):
        code = row.get("code")
        if not code:
            continue
        was = (initial_snap or {}).get("coupons", {}).get(code)
        now = final_snapshot.get("coupons", {}).get(code)
        facts.append(
            f"- 优惠券 {code}：初始 {json.dumps(was, ensure_ascii=False)}"
            f" → 最终 {json.dumps(now, ensure_ascii=False)}"
        )
    return "\n".join(facts) if facts else "（本任务 delta 未牵涉任何具体对象）"


def build_orchestration_judge_prompt(
    task: Any, answer: Any, sandbox_facts: str
) -> list[dict[str, Any]]:
    """渲染 C 场景 judge 的 messages。

    与 B 场景刻意分成两个函数而不是共用一个：B 的 prompt 讲的是「依据检索到的文档原文」，
    C 讲的是「依据沙箱最终状态 + 逐条评分点」—— 两者的证据来源根本不是一回事，
    硬塞进一个 builder 里靠 if 分支只会让两边一起漂。
    """
    rubric = getattr(task, "rubric", None) or []

    sections: list[str] = []
    for i, point in enumerate(rubric, start=1):
        cid = point.get("criterion_id", f"#{i}")
        block = [
            f"### 评分点 {i}（criterion_id = {cid}）",
            f"要求：{point.get('requirement', '')}",
            f"权重：{point.get('weight', '')}",
            f"半分规则：{point.get('partial_rule', '')}",
        ]
        if point.get("critical"):
            block.append("（关键点：完全没做到时整题判失败）")
        alts = point.get("accepted_alternatives") or []
        if alts:
            block.append("以下表述视为等价，同样给满分：" + "；".join(str(a) for a in alts))
        block.append(
            f"沙箱证据（判分依据，以最终状态为准）：{sandbox_facts}"
        )
        sections.append("\n".join(block))

    answer_text = "" if _is_empty_answer(answer) else (
        answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
    )

    system = (
        "你是中文多工具编排任务的评分员。给你一份评分细则、一份待评回复和沙箱关键事实，"
        "你只负责逐条判断回复有没有做到每个评分点，不要给总分。\n"
        "判分原则：\n"
        "1. 沙箱最终状态是**唯一的事实来源**。回复说自己已经处理了但沙箱里没改，"
        "按「没处理」判分，不采信回复里的陈述；\n"
        "2. 依据沙箱事实逐条打 0 / 0.5 / 1，0.5 的触发情形以该评分点的「半分规则」为准；\n"
        "3. 措辞不同但意思等价，算答到；\n"
        "4. 回复里没有明确写出来的内容一律不得算答到（不替回复脑补）；\n"
        "5. 「应当拒绝执行的陷阱任务」（如未支付订单不可退款、金额未达券门槛），"
        "把该说的原因说清楚就是满分，**不要**因为「什么都没改」而扣分 —— "
        "沙箱本来就该保持原样。\n"
        "只输出一个 JSON 对象，不要输出任何解释文字或 Markdown 代码块标记。"
    )
    user = (
        f"## 任务指令\n{getattr(task, 'instruction', '')}\n\n"
        f"## 待评回复\n{answer_text or '（空）'}\n\n"
        f"## 评分细则（共 {len(sections)} 个评分点）\n\n"
        + "\n\n".join(sections)
        + "\n\n## 你要输出的 JSON 结构\n"
        '{"scores": {"<criterion_id>": 0 | 0.5 | 1, ...}, "notes": "<一句话总评>"}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _ask_orchestration_judge(
    messages: list[dict[str, Any]], **kwargs: Any
) -> tuple[dict[str, Any] | None, str]:
    """发一次 C 场景 judge 请求。返回 ``(payload, reason)``；失败时 ``payload=None``。

    与 :func:`_ask_judge` 同样铁律：**不抛**。LLM 挂了 / 返回非 JSON / mock 抛异常，
    一律降级成「payload=None + 原因」，由上层记成回复分全 0，绝不让整批评测掀桌。
    """
    try:
        response = llm_client.chat(
            messages,
            model=kwargs.get("model") or JUDGE_MODEL,
            temperature=kwargs.get("temperature", JUDGE_TEMPERATURE),
            **kwargs.get("client_kwargs", {}),
        )
    except Exception as exc:
        return None, f"judge 调用失败：{type(exc).__name__}: {exc}"
    if not isinstance(response, dict) or "error" in response:
        err = response.get("error") if isinstance(response, dict) else None
        detail = err.get("message") if isinstance(err, dict) else str(response)
        return None, f"judge 调用失败：{detail}"
    message = llm_client.first_message(response)
    if message is None:
        return None, "judge 返回没有 choices[0].message"
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return None, "judge 返回 content 为空"
    return _loads_strict(content), "judge 返回的不是合法 JSON"


def _json_copy(value: Any) -> Any:
    """JSON 结构深拷贝。走 ``json`` 往返而非 ``copy.deepcopy``：判分/harness 那条
    stdlib 白名单里没有 ``copy``（见 ``tests/test_harness.py``），而沙箱快照本来就是
    ``snapshot()`` 产出的纯 JSON，round-trip 与 deepcopy 对它是等价的。

    round-trip 失败（结构里混了不可序列化值）时退回原值：判分器这一层铁律是不抛，
    比不出真分数也不能让整批评测掀桌。
    """
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return value


def _row_drift(token: str, row: dict[str, Any],
               initial: dict[str, Any], final: dict[str, Any]) -> list[str]:
    """列出某一对象「除 delta 声明字段外」的**额外漂移**，按字段给中文说明。"""
    was = initial.get(token)
    now = final.get(token)
    if not isinstance(was, dict) or not isinstance(now, dict):
        return [f"{token}：最终状态里不存在该对象"]
    declared = set(row) - {"order_id", "code"}
    drift: list[str] = []
    for field_name in sorted(set(was) | set(now)):
        if field_name in declared or field_name in _INVARIANT_EXCLUDED_FIELDS:
            continue
        if was.get(field_name) != now.get(field_name):
            drift.append(
                f"{token}.{field_name}：{was.get(field_name)!r} → {now.get(field_name)!r}"
            )
    return drift


def _judge_orchestration(
    task: Any, answer: Any, *, final_snapshot: Any = None, trajectory: Any = None,
    **judge_kwargs: Any
) -> dict[str, Any]:
    """C 场景判分：delta 三段（机器判）+ 回复质量（LLM 判），二者合流成 ``correct``。

    返回 ``{"correct", "delta_ok", "reply_ok", "orders_ok", "coupons_ok",
    "must_not_change_ok", "orders_detail", "coupons_detail", "must_not_change_detail",
    "G", "capped", "scores", "model", "reason"}``。

    **三段口径（TASK-019 ``_validate_delta`` / RESULT-019 §5.2）**：

    - ``orders`` / ``coupons`` —— **目标态**。逐项判「这单 / 这券有没有改成目标状态」；
    - ``must_not_change`` —— **不变式**。逐项判「该保持原样的对象有没有保持原样」。

    只判前两段 = 陷阱类（退款判断 / 券抵扣，正确答案就是「什么都没改」）恒满分 ——
    那等于这两类完全没测出来。这是本单最不能出错的地方。

    ``must_not_change`` 比的是**全量状态**（delta 声明的字段 + 其余字段有没有漂），
    而不是只比声明的那一两个字段：契约原话是「比『未受影响』的全量状态」。
    只比声明字段的话，agent 把订单地址、金额、客户一起改了照样满分。
    """
    notes: list[str] = []
    initial_data = getattr(task, "initial_data", None)
    initial_snap = (initial_data or {}).get("sandbox") if isinstance(initial_data, dict) else None
    ground_truth = getattr(task, "ground_truth", None)
    delta = ground_truth.get("expected_final_state") if isinstance(ground_truth, dict) else None
    if not isinstance(delta, dict):
        raise ValueError("C 场景任务缺少 ground_truth.expected_final_state，无从判分")

    orders_rows = list(delta.get("orders") or [])
    coupons_rows = list(delta.get("coupons") or [])
    must_rows = list(delta.get("must_not_change") or [])
    if not (orders_rows or coupons_rows or must_rows):
        # 构造器（tasks/orchestration._validate_delta）已挡；这里再挡一次是因为
        # 「三段全空」静默判满分是最危险的一种错 —— 出题人自己没想清楚要判什么，
        # 判分器却给出一个漂亮的 100%。宁可炸，也不给假分。
        raise ValueError(
            "delta 为空：orders / coupons / must_not_change 三段全空，无从判对错。"
            "陷阱任务请用 must_not_change 写明「哪些东西必须保持原样」"
        )

    # 深拷贝后再比：判分器自己排序 / 筛选 / 建索引都不许污染进来的快照，
    # 更不许污染 task.initial_data（同一份 initial_data 可能被复用去判第二条任务）。
    # 走 json 往返而不是 copy.deepcopy —— 后者不在 harness 那份 stdlib 白名单里，
    # 而沙箱快照本来就是 JSON 结构，round-trip 就是它的深拷贝。
    final = _json_copy(final_snapshot) if isinstance(final_snapshot, dict) else None
    initial = _json_copy(initial_snap) if isinstance(initial_snap, dict) else {}
    final_orders = {o.get("order_id"): o for o in (final or {}).get("orders", [])
                    if isinstance(o, dict)}
    final_coupons = dict((final or {}).get("coupons", {}) or {})
    initial_orders = {o.get("order_id"): o for o in initial.get("orders", [])
                      if isinstance(o, dict)}
    initial_coupons = dict(initial.get("coupons", {}) or {})
    if final is None:
        notes.append("拿不到沙箱最终快照：agent 未跑或环境未回传，delta 全部判不通过")

    def _check_targets(rows: list[dict[str, Any]], pool: dict[str, Any],
                       pool_name: str) -> tuple[bool, list[str]]:
        detail: list[str] = []
        ok = True
        for row in rows:
            token = row.get("order_id") or row.get("code")
            now = pool.get(token)
            if now is None:
                ok = False
                detail.append(f"{token}：最终状态里不存在该{pool_name}")
                continue
            bad = [
                f"{token}.{field_name}：期望 {value!r}，实际 {now.get(field_name)!r}"
                for field_name, value in row.items()
                if field_name not in ("order_id", "code") and now.get(field_name) != value
            ]
            if bad:
                ok = False
                detail.extend(bad)
        return ok, detail

    orders_ok, orders_detail = _check_targets(orders_rows, final_orders, "订单")
    coupons_ok, coupons_detail = _check_targets(coupons_rows, final_coupons, "优惠券")

    must_not_change_ok, must_detail = True, []
    for row in must_rows:
        token = row.get("order_id") or row.get("code")
        pool = final_orders if row.get("order_id") else final_coupons
        source = initial_orders if row.get("order_id") else initial_coupons
        now = pool.get(token)
        if now is None:
            must_not_change_ok = False
            must_detail.append(f"{token}：最终状态里不存在该对象（不该动的对象消失了）")
            continue
        bad = [
            f"{token}.{field_name}：应保持 {value!r}，实际 {now.get(field_name)!r}"
            for field_name, value in row.items()
            if field_name not in ("order_id", "code") and now.get(field_name) != value
        ]
        # 声明字段对上了还不够：「不该动」意味着**别的字段也不许动**
        # （把地址、金额、客户顺手改了，同样是动了不该动的对象）。
        bad.extend(_row_drift(str(token), row, source, pool))
        if bad:
            must_not_change_ok = False
            must_detail.extend(bad)

    delta_ok = bool(orders_ok and coupons_ok and must_not_change_ok)

    # ---- 回复质量：LLM 逐点打 rubric，最后一条恒为回复点 ----
    rubric = getattr(task, "rubric", None) or []
    reply_points = [p for p in rubric if _is_reply_point(p)]
    if not reply_points:
        # 没有回复点等于放弃了一半的判据（RESULT-019 §5.5 的反面）。
        # 不静默当满分：记成不通过并写明原因。
        notes.append("任务 rubric 里没有回复点，无法判回复质量，按回复不通过记")
        reply_g, reply_scores, capped = 0.0, {}, False
    else:
        messages = build_orchestration_judge_prompt(
            task, answer, _sandbox_facts(task, final, delta)
        )
        payload, failure = _ask_orchestration_judge(messages, **judge_kwargs)
        reply_scores = {p["criterion_id"]: 0.0 for p in reply_points}
        if payload is None:
            notes.append(failure)
        else:
            reply_scores = _collect_scores(reply_points, payload.get("scores"), notes)
        try:
            from .tasks.qa import aggregate_rubric

            reply_g, capped = aggregate_rubric(reply_points, reply_scores)
        except Exception as exc:
            capped = False
            notes.append(f"回复点聚合失败：{type(exc).__name__}: {exc}")
        if _is_empty_answer(answer):
            notes.append("回复为空：回复质量记 0")

    reply_ok = bool(reply_g >= PASS_THRESHOLD_C and not capped)
    correct = bool(delta_ok and reply_ok)

    reason_parts: list[str] = []
    if not orders_ok:
        reason_parts.append("orders 未达目标态：" + "；".join(orders_detail))
    if not coupons_ok:
        reason_parts.append("coupons 未达目标态：" + "；".join(coupons_detail))
    if not must_not_change_ok:
        reason_parts.append("must_not_change 被破坏：" + "；".join(must_detail))
    if not reply_ok:
        reason_parts.append(
            f"回复质量不达标：回复点 G={reply_g:.3f} < {PASS_THRESHOLD_C}"
            + ("（关键回复点漏了）" if capped else "")
        )
    if not reason_parts:
        reason_parts.append("delta 三段全中且回复达标")
    reason_parts.extend(notes)

    return {
        "correct": correct,
        "delta_ok": delta_ok,
        "reply_ok": reply_ok,
        "orders_ok": orders_ok,
        "coupons_ok": coupons_ok,
        "must_not_change_ok": must_not_change_ok,
        "orders_detail": orders_detail,
        "coupons_detail": coupons_detail,
        "must_not_change_detail": must_detail,
        "G": reply_g,
        "capped": capped,
        "scores": reply_scores,
        "model": judge_kwargs.get("model") or JUDGE_MODEL,
        "reason": "；".join(reason_parts),
        "diagnostics": {"n_reply_points": len(reply_points)},
    }


#: 回复点的识别口径。rubric 字段本身不标「这条是回复点」，而是在 requirement 里
#: 带「回复 / 告知 / 说明 / 复述」这些面向客户的措辞。**只有最后一条**参与回复质量判定
#: —— 它是 B 场景打过的同一批判据；编排点（查了什么、调了什么）已由 delta 三段
#: 机器判过，再让 LLM 判一遍只会重复计分。
_REPLY_POINT_KEYWORDS = ("回复", "告知", "说明", "复述")


def _is_reply_point(point: dict[str, Any]) -> bool:
    """该评分点是否面向客户（要判「说清楚了没」）。"""
    requirement = point.get("requirement", "")
    if not isinstance(requirement, str):
        return False
    return any(k in requirement for k in _REPLY_POINT_KEYWORDS)
