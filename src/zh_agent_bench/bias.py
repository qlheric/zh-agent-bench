"""bias —— judge 偏置配对实验（TASK-017，M2「中文 judge 校准」的另一半）。

**这张报告回答的问题**：judge 判分有没有**系统性歪**（:mod:`annotate` 回答的是
「judge 判得准不准」—— 单点校准度）。偏置与校准是两件事：一个 judge 每次都把
同一条题判对，但只要它偏爱长答案、或偏爱 rubric 点的第一个，那它排出来的榜单
就整体歪着，而单看准确率永远看不出来。

三类配对，都做成**同一份内容换一种呈现**，让差异无法用「题目变难了」解释：

======================  ===========================================  ==========================
偏置                    配对构造                                        看什么
======================  ===========================================  ==========================
:func:`position_pairs`   rubric 点顺序 原样 vs 逆序                   G 翻转率
:func:`length_pairs`    同一答案的 简短版 vs 冗长版                    G 差（长度 vs 质量）
:func:`irrelevant_pairs` 真答案 vs 高质量但无关的空话                  G 应大幅下降
======================  ===========================================  ==========================

三类都走**同一个判分入口** :func:`judge.judge_task`，只把「题目/答案/顺序」换掉 ——
不另写一套打分逻辑，否则测的不是线上那个 judge。

**位置偏置怎么做到只改顺序**：:func:`judge.build_judge_prompt` 是从
``task.rubric`` 逐条渲染的，所以只要造一个 ``rubric`` 逆序的 Task 副本，
prompt 里评分点的呈现顺序就跟着翻了；而 :mod:`judge` 里
:func:`judge._collect_scores` 按 ``criterion_id`` 配权重，
:func:`judge._evidence_coverage` 也按 id 走 —— **所以逆序不改变 G 的定义，
只改变模型读到的顺序**。这正是位置偏置实验要的：分母分子不动，只有顺序变。

用 ``dataclasses.replace`` 做副本而不是重造一个 Task：字段一多就漏，
而漏掉的字段会静默变成 ``None`` 而不是报错 —— 那比不改更坏。

**答非所问那一类最容易「测了个寂寞」**：拿空话当答案，任何 judge 都会判低分，
数字很好看，但什么都没证明。这里给空话配**真实原文**（同一条任务的证据），
也就是说 judge 手上资料齐全、答案却答非所问 —— 这样低分才是「判得对」，
而不是「没东西可判」。空话本身按答案长度加权生成（:func:`_filler`），
和冗长版走同一个函数，避免「答非所问恰好比较短」这个混淆变量。

**运行时仅 stdlib + 本包**，与全项目一致。

用法::

    python -m zh_agent_bench.bias --split trial --out benchmarks/qa/calibration-report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .calibrate import load_formal_split
from .harness import _json_safe
from .judge import PASS_THRESHOLD_G, judge_task
from .tasks.schema import Task

#: 判分失败标记。偏置报告必须把「judge 挂了」和「judge 判低」分开 ——
#: 前者是实验故障，后者才是待观察的行为。
_FAILED: dict[str, Any] = {"G": None, "correct": None, "failed": True}


# --------------------------------------------------------------------------------------
# 判分入口（唯一）
# --------------------------------------------------------------------------------------


def _score(task: Task, answer: Any, judge_fn: Any = judge_task) -> dict[str, Any]:
    """跑一次 judge，返回 ``{G, correct, failed}``。

    ``judge_task`` 在调用失败时**不抛异常**，而是返回兜底 verdict（``G=0``、
    reason 里带降级说明）。这里把那种情况标成 ``failed=True`` 而不是当成一个
    ``G=0`` 观测值 —— 否则一次网络抖动会以「judge 判低」的身份混进翻转率。
    """
    try:
        verdict = judge_fn(task, answer)
    except Exception as exc:  # noqa: BLE001 —— 实验不该被单条崩掉掀掉
        return {**_FAILED, "reason": f"judge 抛异常：{type(exc).__name__}: {exc}"}

    reason = verdict.get("reason") or ""
    degraded = "judge 调用失败" in reason or "降级" in reason
    if degraded:
        return {**_FAILED, "reason": reason}
    return {
        "G": verdict.get("G"),
        "correct": bool(verdict.get("correct")),
        "failed": False,
        "reason": reason,
    }


# --------------------------------------------------------------------------------------
# 配对一：位置偏置（rubric 顺序）
# --------------------------------------------------------------------------------------


def reversed_task(task: Task) -> Task:
    """造一份 ``rubric`` 逆序的 Task 副本（其余字段不动）。"""
    rubric = list(task.rubric or ())
    return replace(task, rubric=tuple(reversed(rubric)))


def position_pairs(
    tasks: Sequence[Task],
    judge_fn: Any = judge_task,
    *,
    answers: dict[str, Any] | None = None,
    judge_name: str = "judge",
) -> dict[str, Any]:
    """位置偏置：同一题、同答案，只把 rubric 点的呈现顺序翻转。

    报告 **G 翻转率**（两侧都有有效 G 时，G 是否跨过主榜门槛 ``PASS_THRESHOLD_G``
    翻转）。用门槛而不是「G 值有变化」：LLM 本来就有抖动，G 动 0.03 不构成偏置；
    只有**从「过线」翻到「不过线」**才真的改变榜单结论。

    ``G`` 的**平均差**（逆序 − 原序）一并给出，方向性比翻转率更敏感。

    ``answers`` 为空时退回任务自带的 ``example_answer`` 当答案（与
    :func:`irrelevant_pairs` 同一份来源）—— 没有它就等于拿空答案测位置偏置，
    测出来的是「judge 对空答案判几分」，与顺序无关。
    """
    if not answers:
        answers = _true_answers_from_tasks(tasks)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        answer = answers.get(task.id)
        base = _score(task, answer, judge_fn)
        rev = _score(reversed_task(task), answer, judge_fn)
        rows.append({
            "task_id": task.id,
            "n_rubric": len(task.rubric or ()),
            "g_base": base["G"],
            "g_reversed": rev["G"],
            "delta": _delta(rev["G"], base["G"]),
            "flipped": _flipped(base, rev),
            "failed": bool(base["failed"] or rev["failed"]),
        })

    usable = [r for r in rows if not r["failed"] and r["flipped"] is not None]
    return {
        "judge": judge_name,
        "n": len(rows),
        "n_failed": sum(1 for r in rows if r["failed"]),
        "n_usable": len(usable),
        "flip_rate": (sum(1 for r in usable if r["flipped"]) / len(usable)) if usable else None,
        "mean_delta": _mean([r["delta"] for r in usable]),
        "rows": rows,
    }


# --------------------------------------------------------------------------------------
# 配对二：长度偏置（同一答案的短/长版）
# --------------------------------------------------------------------------------------

#: 冗长版追加的填充语。**不引入任何事实性内容** —— 只重复已知句式与客套，
#: 目的是只拉长度、不改变信息量。若填充语里混进新事实，长度实验就退化成
#: 「加信息会加分还是减分」，测不到长度偏置了。
_FILLER_SENTENCES = (
    "综上所述，相关事项需要结合实际情况进一步确认。",
    "如有其他疑问，欢迎随时与我们联系沟通。",
    "以上内容仅供参考，具体请以最新文件为准。",
    "为避免遗漏，建议再次核对相关流程的每个环节。",
    "整个处理过程应当遵循既定的管理规范执行。",
)


def _filler(target_chars: int) -> str:
    """拼**至少** ``target_chars`` 字的填充段。

    隔句取（不连着取同一批）让句式有变化；一轮不够就再循环一轮 ——
    宁可重复客套句，也不能悄悄少加字数：``added_chars`` 达不到时长度实验
    实际没加长度，量出来的 Δ 会被当成「judge 对长度不敏感」而误报。
    """
    ordered = _FILLER_SENTENCES[1::2] + _FILLER_SENTENCES[0::2]
    out: list[str] = []
    total = 0
    while total < target_chars:
        for sentence in ordered:
            out.append(sentence)
            total += len(sentence)
            if total >= target_chars:
                break
    return "".join(out)


#: 冗长版的最低增量（字）。:func:`_filler` 能满足任意 ``--added-chars``，但
#: CLI 默认值必须真的够长才有实验意义 —— 太短的话 Δ 是噪声而不是长度效应。
MIN_ADDED_CHARS = 100


def verbose_variant(answer: Any, *, added_chars: int = 120) -> Any:
    """把答案加长但**不改语义**：在字符串答案末尾追加填充段。

    非字符串答案（结构化 list/dict）原样返回 —— 本实验只对开放文本答案有意义，
    给结构化答案硬拼字符串会直接改变它的类型与判分路径。
    """
    if not isinstance(answer, str) or not answer.strip():
        return answer
    return answer + "\n" + _filler(added_chars)


def length_pairs(
    tasks: Sequence[Task],
    answers: dict[str, Any],
    judge_fn: Any = judge_task,
    *,
    added_chars: int = 120,
    judge_name: str = "judge",
) -> dict[str, Any]:
    """长度偏置：同一条答案的**原版 vs 冗长版**（语义保持），比较 G。

    ``answers`` 是 ``task_id -> agent 答案``，来自 ``eval_run --save-answers``
    产出的 ``records.jsonl``（:mod:`annotate` 读的是同一份）。

    **报 mean_delta（冗长 − 原版）而不是只看翻转率**：长度偏置的典型表现是
    「长答案稳定高一点分但没到改变结论的程度」，只报翻转率会把它算作「无偏置」。
    """
    rows: list[dict[str, Any]] = []
    for task in tasks:
        answer = answers.get(task.id)
        if answer is None:
            continue  # 没有答案的题不参与（不补造答案，那会污染成另一个实验）
        base = _score(task, answer, judge_fn)
        long = _score(task, verbose_variant(answer, added_chars=added_chars), judge_fn)
        rows.append({
            "task_id": task.id,
            "len_base": len(answer) if isinstance(answer, str) else None,
            "len_verbose": len(verbose_variant(answer, added_chars=added_chars))
            if isinstance(answer, str) else None,
            "g_base": base["G"],
            "g_verbose": long["G"],
            "delta": _delta(long["G"], base["G"]),
            "flipped": _flipped(base, long),
            "failed": bool(base["failed"] or long["failed"]),
        })

    usable = [r for r in rows if not r["failed"] and r["flipped"] is not None]
    return {
        "judge": judge_name,
        "n": len(rows),
        "n_missing_answer": len(tasks) - len(rows),
        "n_failed": sum(1 for r in rows if r["failed"]),
        "n_usable": len(usable),
        "flip_rate": (sum(1 for r in usable if r["flipped"]) / len(usable)) if usable else None,
        "mean_delta": _mean([r["delta"] for r in usable]),
        "rows": rows,
    }


# --------------------------------------------------------------------------------------
# 配对三：答非所问（高质量但无关）
# --------------------------------------------------------------------------------------


def irrelevant_answer(task: Task, *, added_chars: int = 0) -> str:
    """构造「语言质量高、但与本任务无关」的空话答案。

    取任务的**证据原文**做骨架（而不是凭空造话）是有意的：这样答案里出现的
    名词、术语全都来自本知识库，judge 若只看「像不像行话」就会被骗过去 ——
    正好是要测的那层抗性。若答案里全是不相干的词，任何 judge 都会判 0，
    测不出抗性。
    """
    doc_ids = [p.get("evidence_ids") or [] for p in (task.rubric or ())]
    ids = [i for ids_ in doc_ids for i in ids_]
    text = "，".join(str(ids[i % len(ids)]) for i in range(3)) if ids else "相关文档"
    answer = (
        f"整体来看，{text}所涉及的流程安排是清晰的，各环节衔接也比较顺畅。"
        "从日常执行的角度来说，相关各方保持及时沟通、认真核对每一步材料，"
        "就能把事情办得稳妥，不至于出现疏漏。"
    )
    return answer + ("\n" + _filler(added_chars) if added_chars else "")


def irrelevant_pairs(
    tasks: Sequence[Task],
    judge_fn: Any = judge_task,
    *,
    added_chars: int = 120,
    judge_name: str = "judge",
) -> dict[str, Any]:
    """答非所问：真答案 vs 高质量但无关的空话。

    **期望方向是 G 大幅下降**（判低才是判对）。这里把结论判成
    ``g_true`` / ``g_irrelevant`` 的均值差：``true - irrelevant`` 为正才说明
    judge 能识破空话。若它 ≤ 0，说明 judge 被行话唬住了 —— 那是要报出去的问题。
    """
    answers = _true_answers_from_tasks(tasks)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        answer = answers[task.id]
        true = _score(task, answer, judge_fn)
        irr = _score(task, irrelevant_answer(task, added_chars=added_chars), judge_fn)
        rows.append({
            "task_id": task.id,
            "g_true": true["G"],
            "g_irrelevant": irr["G"],
            "gap": _delta(true["G"], irr["G"]),
            # **严格大于**：打平不算识破。一个恒定分的假 judge 在真答案与空话上
            # 都给 0.3，点差=0 —— 那是「测不出差别」，不是「抗空话」。用 >= 会把它
            # 记成抗性成立，正是常量替身最容易糊弄过去的地方。
            "judge_got_it": (irr["G"] is not None and true["G"] is not None
                             and true["G"] > irr["G"]),
            "failed": bool(true["failed"] or irr["failed"]),
        })

    usable = [r for r in rows if not r["failed"] and r["gap"] is not None]
    return {
        "judge": judge_name,
        "n": len(rows),
        "n_failed": sum(1 for r in rows if r["failed"]),
        "n_usable": len(usable),
        "n_judge_got_it": sum(1 for r in usable if r["judge_got_it"]),
        "resistance_rate": (sum(1 for r in usable if r["judge_got_it"]) / len(usable))
        if usable else None,
        "mean_g_true": _mean([r["g_true"] for r in usable]),
        "mean_g_irrelevant": _mean([r["g_irrelevant"] for r in usable]),
        "mean_gap": _mean([r["gap"] for r in usable]),
        # 真答案平均分：恒定分的假 judge 会给 0，这正是「只报抗空话率」要掩盖的另一半
        # —— 抗空话率高也可能意味着 judge 谁都不给分，一并报出来才看得出。
        "mean_g_true_pass_rate": _mean(
            [1.0 if r["g_true"] is not None and r["g_true"] >= PASS_THRESHOLD_G else 0.0
             for r in usable]
        ),
        "rows": rows,
    }


def _true_answers_from_tasks(tasks: Sequence[Task]) -> dict[str, Any]:
    """拿任务自带的 ``example_answer`` 当「真答案」。

    这是本模块唯一一处用参考答案的地方，且**只用于生成对照**、不进任何 prompt ——
    :func:`judge.build_judge_prompt` 本来就不含它（复述即满分那条禁令）。
    注意它因此**不是真实 agent 答案**（那要真跑一遍），所以本实验测的是
    「judge 分不出满分答案与行话空话」，不是端到端榜单上的抗性。
    """
    return {t.id: (t.ground_truth or {}).get("example_answer") for t in tasks}


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------


def _delta(a: Any, b: Any) -> float | None:
    """``a - b``，任一侧不是数字就返回 ``None``（不拿 0 冒充）。"""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    return a - b


def _flipped(base: dict[str, Any], other: dict[str, Any]) -> bool | None:
    """两侧 ``correct`` 是否跨过主榜门槛翻转。任一侧无效返回 ``None``。"""
    if base["correct"] is None or other["correct"] is None:
        return None
    return base["correct"] != other["correct"]


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sum(present) / len(present) if present else None


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


# --------------------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------------------


def render_report(
    position: dict[str, Any],
    length: dict[str, Any],
    irrelevant: dict[str, Any],
    *,
    judge_name: str,
) -> str:
    """渲染 ``bias.md``。**无时间戳**（LLM 有随机性，结构稳定即可）。"""
    lines: list[str] = []
    lines.append("# B 场景 judge 偏置配对实验报告")
    lines.append("")
    lines.append(f"- judge：{judge_name}")
    lines.append(f"- 主榜门槛：G ≥ {PASS_THRESHOLD_G}")
    lines.append("- 对照答案：位置/答非所问用任务自带 `example_answer`，长度用 `eval_run --save-answers` 的 agent 答案")
    lines.append("")
    lines.append("> 本报告**不含时间戳**：同配置两次跑结构一致、分数允许不同（LLM 有随机性）。")
    lines.append("")
    lines.append("**读法**：翻转率 = 两侧都判分成功时，`correct` 跨过主榜门槛的比例；"
                 "`mean_delta` = 后者 G − 前者 G，均值方向性比翻转率更敏感。")
    lines.append("")

    # --- 位置 ---
    lines.append("## 1. 位置偏置（rubric 呈现顺序 原样 vs 逆序）")
    lines.append("")
    lines.append(f"- 任务数：{position['n']}（判分失败 {position['n_failed']}，可用 {position['n_usable']}）")
    lines.append(f"- G 翻转率：{_pct(position['flip_rate'])}")
    lines.append(f"- mean_delta（逆序 − 原序）：{_fmt(position['mean_delta'])}")
    verdict = _position_verdict(position)
    lines.append(f"- **结论**：{verdict}")
    lines.append("")
    lines.append("| task_id | 评分点数 | G 原序 | G 逆序 | 差 | 翻转 |")
    lines.append("|---|---|---|---|---|---|")
    for r in position["rows"]:
        lines.append(
            f"| {r['task_id']} | {r['n_rubric']} | {_fmt(r['g_base'])} | {_fmt(r['g_reversed'])} "
            f"| {_fmt(r['delta'])} | {_flipped_cell(r['flipped'], r['failed'])} |"
        )
    lines.append("")

    # --- 长度 ---
    lines.append("## 2. 长度偏置（同一答案 原版 vs 冗长版）")
    lines.append("")
    lines.append(f"- 任务数：{length['n']}（无 agent 答案 {length['n_missing_answer']}，"
                 f"判分失败 {length['n_failed']}，可用 {length['n_usable']}）")
    lines.append(f"- G 翻转率：{_pct(length['flip_rate'])}")
    lines.append(f"- mean_delta（冗长 − 原版）：{_fmt(length['mean_delta'])}")
    lines.append(f"- **结论**：{_length_verdict(length)}")
    lines.append("")
    lines.append("| task_id | 原字数 | 冗长字数 | G 原版 | G 冗长 | 差 | 翻转 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in length["rows"]:
        lines.append(
            f"| {r['task_id']} | {r['len_base']} | {r['len_verbose']} | {_fmt(r['g_base'])} "
            f"| {_fmt(r['g_verbose'])} | {_fmt(r['delta'])} | {_flipped_cell(r['flipped'], r['failed'])} |"
        )
    lines.append("")

    # --- 答非所问 ---
    lines.append("## 3. 答非所问（真答案 vs 高质量但无关的空话）")
    lines.append("")
    lines.append(f"- 任务数：{irrelevant['n']}（判分失败 {irrelevant['n_failed']}，可用 {irrelevant['n_usable']}）")
    lines.append(f"- G 均值 真答案：{_fmt(irrelevant['mean_g_true'])}")
    lines.append(f"- G 均值 答非所问：{_fmt(irrelevant['mean_g_irrelevant'])}")
    lines.append(f"- mean_gap（真 − 空话）：{_fmt(irrelevant['mean_gap'])}")
    lines.append(f"- 抗空话率（真答案分更高）：{_pct(irrelevant['resistance_rate'])}")
    lines.append(f"- 真答案过线率（G ≥ {PASS_THRESHOLD_G}）："
                 f"{_pct(irrelevant.get('mean_g_true_pass_rate'))}")
    lines.append(f"- **结论**：{_irrelevant_verdict(irrelevant)}")
    lines.append("")
    lines.append("| task_id | G 真答案 | G 答非所问 | 差 | 识破 |")
    lines.append("|---|---|---|---|---|")
    for r in irrelevant["rows"]:
        mark = "判分失败" if r["failed"] else ("是" if r["judge_got_it"] else "**否**")
        lines.append(
            f"| {r['task_id']} | {_fmt(r['g_true'])} | {_fmt(r['g_irrelevant'])} "
            f"| {_fmt(r['gap'])} | {mark} |"
        )
    lines.append("")
    return "\n".join(lines)


def _flipped_cell(flipped: bool | None, failed: bool) -> str:
    if failed:
        return "判分失败"
    if flipped is None:
        return "n/a"
    return "是" if flipped else "否"


#: 偏置「算不算大」的门槛。用翻转率而不是 G 差做主判据：
#: 榜单上「过线/不过线」是唯一真正改变排名结论的事，
#: G 差 0.02 只是同一档内抖动，不构成系统性歪。
_BIAS_HOT = 0.20


def _position_verdict(data: dict[str, Any]) -> str:
    if not data["n_usable"]:
        return "无可用数据（judge 全部失败），本项未测出来。"
    rate = data["flip_rate"] or 0.0
    if rate > _BIAS_HOT:
        return (f"**有位置偏置** —— {rate * 100:.1f}% 的题仅因评分点呈现顺序反转就翻转过线，"
                "榜单对 rubric 排序敏感，属于要修的缺陷。")
    if abs(data["mean_delta"] or 0.0) >= 0.05:
        return (f"翻转率不高（{rate * 100:.1f}%），但 G 平均移动了 "
                f"{data['mean_delta']:+.3f} —— 顺序对分数有系统性影响，只是还没到改结论的程度。")
    return f"未检出明显位置偏置（翻转率 {rate * 100:.1f}%，G 平均移动 {(data['mean_delta'] or 0.0):+.3f}）。"


def _length_verdict(data: dict[str, Any]) -> str:
    if not data["n_usable"]:
        return "无可用数据（没有 agent 答案，或 judge 全部失败），本项未测出来。"
    delta = data["mean_delta"] or 0.0
    if delta > 0.05:
        return (f"**有长度偏置** —— 语义不变的加长平均多拿 {delta:+.3f} G，"
                "说明 judge 在为篇幅而非内容付分。")
    if delta < -0.05:
        return f"反向：加长反而少拿 {delta:+.3f} G（不是典型的长度偏置，可能是填充句干扰了判读）。"
    return f"未检出长度偏置（加长平均 {(data['mean_delta'] or 0.0):+.3f} G，翻转率 {_pct(data['flip_rate'])}）。"


def _irrelevant_verdict(data: dict[str, Any]) -> str:
    if not data["n_usable"]:
        return "无可用数据（judge 全部失败），本项未测出来。"
    gap = data["mean_gap"] or 0.0
    rate = data["resistance_rate"] or 0.0
    if gap <= 0:
        return (f"**抗空话失败** —— 答非所问的空话平均分不低于真答案（gap {gap:+.3f}），"
                f"抗空话率仅 {rate * 100:.1f}%。judge 被行话唬住了，这是要修的缺陷。")
    if rate < 0.9:
        return (f"总体能识破（gap {gap:+.3f}），但有 {100 - rate * 100:.0f}% 的题被空话骗过，属个案。")
    pass_rate = data.get("mean_g_true_pass_rate") or 0.0
    tail = ("" if pass_rate >= 0.8 else
            f"　⚠ 但真答案的过线率只有 {pass_rate * 100:.1f}% —— "
            "抗空话率高也可能意味着 judge 谁都不给分，两者要一起读。")
    return f"抗空话正常（gap {gap:+.3f}，抗空话率 {rate * 100:.1f}%）。{tail}"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def load_answers(records_path: Path) -> dict[str, Any]:
    """从 ``eval_run --save-answers`` 的 ``records.jsonl`` 读 ``task_id -> answer``。

    文件不存在或没有 ``answer`` 字段时返回空 dict —— 长度偏置那一项会因此
    变成「无可用数据」并在报告里明说，而不是静默变成 0 条。
    """
    if not records_path.is_file():
        return {}
    answers: dict[str, Any] = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "answer" in row and row["answer"] is not None:
            answers[row.get("task_id")] = row["answer"]
    return answers


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m zh_agent_bench.bias",
        description="judge 偏置配对实验：位置 / 长度 / 答非所问",
    )
    parser.add_argument("--split", default="trial", choices=("trial", "dev", "blind"),
                        help="正式集 split（默认 trial）")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="正式集根目录（默认仓库 benchmarks/qa）")
    parser.add_argument("--answers", type=Path, default=None,
                        help="eval_run --save-answers 产的 records.jsonl（长度偏置用；缺则该项无数据）")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/qa/calibration-report"),
                        help="输出目录（写 bias.md + bias-results.json）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 = 全跑）")
    parser.add_argument("--added-chars", type=int, default=120,
                        help="冗长版/答非所问追加的填充字数（默认 120）")
    args = parser.parse_args(argv)

    try:
        tasks = load_formal_split(args.split, args.data_root)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover —— parser.error 不返回
    if args.limit > 0:
        tasks = tasks[: args.limit]

    answers = load_answers(args.answers) if args.answers else {}
    judge_name = "judge（llm_client.DEFAULT_MODEL）"

    position = position_pairs(tasks, answers=answers, judge_name=judge_name)
    length = length_pairs(tasks, answers, added_chars=args.added_chars, judge_name=judge_name)
    irrelevant = irrelevant_pairs(tasks, added_chars=args.added_chars, judge_name=judge_name)

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "bias.md"
    report_path.write_text(
        render_report(position, length, irrelevant, judge_name=judge_name),
        encoding="utf-8", newline="\n",
    )
    (args.out / "bias-results.json").write_text(
        json.dumps(
            _json_safe({"position": position, "length": length, "irrelevant": irrelevant}),
            ensure_ascii=False, sort_keys=True, indent=2,
        ) + "\n",
        encoding="utf-8", newline="\n",
    )

    print(
        f"split={args.split} tasks={len(tasks)} "
        f"位置: 翻转={_pct(position['flip_rate'])} Δ={_fmt(position['mean_delta'])} | "
        f"长度: 翻转={_pct(length['flip_rate'])} Δ={_fmt(length['mean_delta'])} "
        f"(可用 {length['n_usable']}/{length['n']}) | "
        f"答非所问: gap={_fmt(irrelevant['mean_gap'])} "
        f"抗空话={_pct(irrelevant['resistance_rate'])}"
    )
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
