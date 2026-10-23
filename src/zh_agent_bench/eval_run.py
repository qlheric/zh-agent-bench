"""eval_run —— B 场景真实 LLM agent 跑分器（TASK-014）。

**这张报告回答的问题**：「拿一个真 agent 去答，rubric 判不判得动、判出来的分数长什么样」。
校准闭环（:mod:`calibrate`）用**人造答案**测 judge —— 那是纯判分能力；
本单用**真实 agent 答案**跑同一套 :func:`judge.judge_task`，测的是端到端。

三件必须分开记的东西，混在一起就分不清是谁坏了：

1. **agent 失败**（LLM 挂了 / 步数用尽 / 返回 None）；
2. **judge 降级**（复用 :mod:`calibrate` 的 :func:`classify_degradation` 口径 —— 同一套
   降级定义抄两份必然漂移）；
3. **agent 真的答错**（前两者都没有，judge 给了低分）。

**不要求逐字节可复现** —— LLM 有随机性。要求的只是**报告结构稳定、记录里无时间戳**：
同 agent 配置两次跑，表格形状一样、字段集合一样，分数允许不同。

**运行时仅 stdlib + 本包**，与全项目一致。

用法::

    python -m zh_agent_bench.eval_run --split dev          # 产物落到 benchmarks/qa/dev/<agent>/
    python -m zh_agent_bench.eval_run --split dev --out /tmp/x   # --out 仍可覆盖
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from .calibrate import classify_degradation, load_formal_split
from .harness import (
    _json_safe,
    _ToolBox,
    run_orchestration_trajectory,
)
from .judge import JUDGE_MODEL, PASS_THRESHOLD_C, PASS_THRESHOLD_G, judge_task
from .llm_client import API_KEY_ENV, DEFAULT_MODEL
from .tasks.orchestration import TYPE_LABELS, build_orchestration_dataset
from .tasks.schema import Task


# --------------------------------------------------------------------------------------
# 产物路径布局（TASK-028）
#
# ``benchmarks/<场景>/<档>/<agent>/records.jsonl`` —— agent 名做目录。**两 agent 跑同一档
# 不再互相覆盖**，这是「一个基准只能评一个 agent」的解法。
#
# 这张表是**唯一真相源**：跑分侧（:func:`default_out_dir`）写哪儿，榜单侧
# （:mod:`zh_agent_bench.leaderboard`）就扫哪儿。两边各写一份路径必然漂 ——
# 改一处忘了另一处，榜单就静默读了个空目录还照常出一张全「未跑」的表。
#
# A 场景叫 ``office`` 而不是沿用老的 ``benchmarks/results/records.jsonl``：
# 裸 ``results`` 读不出是哪个场景（``benchmarks/calibration/`` 下面还有个同名的 ``results``
# 概念），而契约 §2.5 点名 ``office``。A 的**任务集**仍在 ``benchmarks/tasks/``，不受影响。
# --------------------------------------------------------------------------------------

SCENARIO_ROOTS: dict[str, Path] = {
    "A": Path("benchmarks/office"),
    "B": Path("benchmarks/qa"),
    "C": Path("benchmarks/orchestration"),
}

#: 目录名安全化的白名单。``:``（agent 自报名里的 ``llm-agent:deepseek-chat``）、
#: ``.``（模型名 ``gpt-6.1``）、``/`` 与 Windows 的 ``<>:"|?*`` 全不在白名单里，一律换 ``-``。
_SAFE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")


def safe_agent_dirname(agent_name: str) -> str:
    """agent 名 → 文件系统安全的目录名。

    **只做安全化，不做改名**：目录名就是 agent 的自报名小写化，与 ``report.md`` 的
    ``- agent：`` 和 records 里的 ``agent`` 字段一字对应 —— 目录要能自证是谁跑的，
    在这里把名字「整理」成另一套说法反而对不上了。

    空串或全是非白名单字符时回落 ``unknown``：目录名不能是空串或 ``.``，
    那种产物会落在 ``<档>/`` 自己身上，和旧布局撞在一起。
    """
    out: list[str] = []
    for char in agent_name.strip().lower():
        # 白名单只挡「文件系统不安全」的字符；**中文 agent 名必须原样保留** ——
        # 目录名要能与 report.md 的署名对上，把中文折成 unknown 就不是自证了。
        out.append(char if (char.isalnum() or char == "-") else "-")
    folded = "".join(out)
    while "--" in folded:  # 折叠连续分隔符：一个 agent 名不该产生两个连字符
        folded = folded.replace("--", "-")
    trimmed = folded.strip("-")
    return trimmed or "unknown"


def default_out_dir(scenario: str, grade: str, agent_name: str) -> Path:
    """``<场景根>/<档>/<agent 安全名>``。榜单按同一公式扫，两边不许各推一遍。"""
    return SCENARIO_ROOTS[scenario] / grade / safe_agent_dirname(agent_name)


def build_agent(args: argparse.Namespace) -> Any:
    """按 CLI 参数造出被测 agent（B/C 两条路径共用这一处）。

    **三个开关一个都不传时，造出来的 agent 与从前逐字节等价**：
    ``base_url`` / ``api_key`` 传 ``None``，:func:`agents.llm.make_llm_agent` 的缺省值
    就会生效，``llm_client.chat`` 仍走 ``DEEPSEEK_API_KEY`` + 默认端点。

    传了 ``--agent-key-env`` 但那个环境变量**没设**时，key 取到空串 ——
    与 judge 侧同纪律：空 key 由 :func:`llm_client.chat` 转成 ``llm_no_api_key`` 哨兵，
    agent 侧转成「答不出来」交 judge 判错，**不崩**。
    """
    from .agents.llm import make_llm_agent

    kwargs: dict[str, Any] = {"model": args.model or DEFAULT_MODEL}
    if args.agent_base_url:
        kwargs["base_url"] = args.agent_base_url
    if args.agent_key_env != API_KEY_ENV:
        kwargs["api_key"] = os.environ.get(args.agent_key_env, "")
    if args.agent_api_key:
        kwargs["api_key"] = args.agent_api_key
    return make_llm_agent(**kwargs)


# --------------------------------------------------------------------------------------
# 跑一批
# --------------------------------------------------------------------------------------


def _task_type(task: Task) -> str:
    """任务类型（B 场景四类）。取不到就标 ``unknown``，不抛 ——
    报告里分组靠它，但分组失败不该让整批跑分作废。"""
    value = (task.initial_data or {}).get("任务类型")
    return value if isinstance(value, str) and value else "unknown"


def evaluate_one(
    task: Task,
    agent: Callable[[Task, dict], Any],
    judge_fn: Callable[..., dict] = judge_task,
    *,
    save_answer: bool = False,
) -> dict[str, Any]:
    """跑一条：agent 答 → judge 判（G/F/T/capped），落成一条记录。

    **不吞 agent 的异常成「答错」**：工具用错了该扣分、agent 崩了该修 bug，混成同一件事
    就没法定位（:func:`harness.run_trajectory` 同样不吞）。但**单个 task 崩掉不该掀掉整批**
    —— 60 条里挂 1 条是数据问题，掀掉 60 条报告才是。这里接住、按 ``agent_exception``
    记一条，再让批处理继续。

    轨迹直接用 :class:`harness._ToolBox` 自己造一份，而不是走
    :func:`harness.evaluate`：那个函数用 ``run_trajectory`` 的窄 return 契约
    （``correct``/``reason``），**把 G/F/T/capped 全丢了** —— 而本单报告的主体正是这四列。
    复用 ``_ToolBox`` 而不是重抄一份，工具契约校验（:func:`tools.registry.check_contract`）
    与轨迹全量记录这两条既有保证才继续成立。

    ``save_answer=True`` 时**额外**在记录里带一个 ``answer`` 键（agent 原始答案），
    供 :mod:`annotate` 的双标用。默认关 —— 它让 records.jsonl 体积涨上好几倍，
    而日常跑分报告只需要分数。

    实现上把答案先塞进 ``_answer`` 占位、再按开关决定是留成 ``answer`` 还是丢掉：
    两条返回路径（正常 / agent 抛异常）都走同一个收尾，少一个分支就少一处漏改。
    """
    record = _evaluate_one(task, agent, judge_fn)
    answer = record.pop("_answer")
    if save_answer:
        record["answer"] = answer
    return record


def _evaluate_one(
    task: Task,
    agent: Callable[[Task, dict], Any],
    judge_fn: Callable[..., dict],
) -> dict[str, Any]:
    try:
        box = _ToolBox(task.tool_names)
        answer = agent(task, box)
    except Exception as exc:  # noqa: BLE001 —— 见上：单条崩不掀整批
        return {
            "task_id": task.id,
            "task_type": _task_type(task),
            "correct": False,
            "G": 0.0,
            "F": None,
            "T": None,
            "capped": False,
            "n_tool_calls": 0,
            "agent_failed": True,
            "judge_degradation": None,
            "reason": f"agent 抛异常：{type(exc).__name__}: {exc}",
            "_answer": None,
        }

    verdict = judge_fn(task, answer, trajectory=box.calls)
    expected_ids = [p["criterion_id"] for p in (task.rubric or ())]
    reason = verdict.get("reason") or ""
    return {
        "task_id": task.id,
        "task_type": _task_type(task),
        "correct": bool(verdict.get("correct")),
        "G": verdict.get("G"),
        "F": verdict.get("F"),
        "T": verdict.get("T"),
        "capped": bool(verdict.get("capped")),
        "n_tool_calls": len(box.calls),
        "agent_failed": answer is None,  # LLM 出错或步数用尽 ⇒ LlmAgent 返回 None
        "judge_degradation": classify_degradation(reason, verdict.get("scores"), expected_ids),
        "reason": reason,
        "_answer": answer,
    }


def run_split(
    tasks: Iterable[Task],
    agent: Callable[[Task, dict], Any],
    judge_fn: Callable[..., dict] = judge_task,
    *,
    save_answer: bool = False,
) -> list[dict[str, Any]]:
    """跑完整 split，返回记录列表（顺序 = 任务序，天然确定）。"""
    return [evaluate_one(task, agent, judge_fn, save_answer=save_answer) for task in tasks]


# --------------------------------------------------------------------------------------
# C 场景：跑一批
#
# 与 B 场景分成独立的一套函数而不是往 evaluate_one 里塞 if：B 的记录结构是
# G/F/T/capped，C 的记录结构是 delta 三段 + reply —— 两套字段几乎不重叠，
# 合成一张表只会在两边都要 nullable 的时候把「不适用」和「零分」混成一件事。
# --------------------------------------------------------------------------------------


def evaluate_one_orchestration(
    task: Task,
    agent: Callable[[Task, dict], Any],
    judge_fn: Callable[..., dict] = judge_task,
) -> dict[str, Any]:
    """跑一条 C 场景任务，落成一条记录。

    契约 §2.5 要求的七个字段全部在这里定死：``correct / delta_ok / reply_ok /
    orders_ok / coupons_ok / must_not_change_ok / reason``。

    **单条崩不掀整批**，与 B 场景同一条口径：8 条里挂 1 条是数据问题，
    掀掉 8 条报告才是。挂掉的那条七字段全记 False，reason 写明崩在哪。
    """
    try:
        row = run_orchestration_trajectory(task, agent, judge_fn)
    except Exception as exc:  # noqa: BLE001 —— 见上：单条崩不掀整批
        return {
            "task_id": task.id,
            "task_type": _task_type(task),
            "correct": False,
            "delta_ok": False,
            "reply_ok": False,
            "orders_ok": False,
            "coupons_ok": False,
            "must_not_change_ok": False,
            "G": None,
            "capped": False,
            "n_tool_calls": 0,
            "agent_failed": True,
            "reason": f"agent 抛异常：{type(exc).__name__}: {exc}",
        }
    verdict = row["verdict"]
    return {
        "task_id": row["task_id"],
        "task_type": row["task_type"],
        "correct": row["correct"],
        "delta_ok": bool(verdict.get("delta_ok")),
        "reply_ok": bool(verdict.get("reply_ok")),
        "orders_ok": bool(verdict.get("orders_ok")),
        "coupons_ok": bool(verdict.get("coupons_ok")),
        "must_not_change_ok": bool(verdict.get("must_not_change_ok")),
        "G": verdict.get("G"),
        "capped": bool(verdict.get("capped")),
        "n_tool_calls": len(row["tool_calls"]),
        "agent_failed": row["final_answer"] is None,
        "reason": row["reason"],
    }


def run_split_orchestration(
    tasks: Iterable[Task],
    agent: Callable[[Task, dict], Any],
    judge_fn: Callable[..., dict] = judge_task,
) -> list[dict[str, Any]]:
    """跑完整 C 场景任务集，返回记录列表（顺序 = 任务序）。"""
    return [evaluate_one_orchestration(task, agent, judge_fn) for task in tasks]


# --------------------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------------------

#: B 场景四种任务类型，固定顺序。**显式登记**而不是 ``sorted()``：
#: 报告表格的行序是契约的一部分（两次跑逐行可比），字典序会把「多跳」排在「单跳」前面，
#: 而那跟任务类型的既定顺序无关。取不到全集时按未知类型并到末尾。
_TASK_TYPE_ORDER = ("单跳", "多跳", "证据不足", "不可回答")

#: 四个指标列的显示名。``F`` / ``T`` 不适用时显示 ``n/a`` 而不是 ``None`` ——
#: 空答案的 F 本来就「不适用」（judge 自己会这么标），表格里写 ``None`` 会被
#: 读成「这一列没算」。
_METRICS = ("G", "F", "T")


def _mean(values: Sequence[float | None]) -> float | None:
    """只对**非 None** 取均值。``None`` = 该指标不适用（空答案的 F、无轨迹的 T），
    把它当 0 算进均值会把「不适用」伪装成「很差」。

    一个非 None 值都没有时返回 ``None``（而不是 0）—— 「没数据」和「数据全零」
    在榜单上是两件事。
    """
    present = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not present:
        return None
    return sum(present) / len(present)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """汇成结构化统计：总分 + 按 4 类型切片 + 判定分布 + 异常计数。

    四个指标列都算均值：``G`` 是 rubric 加权得分，``F`` 是忠实度，``T`` 是工具协议合规，
    ``capped`` 是关键点封顶的条数（不取均值 —— 它是计数不是分数）。
    """
    def _slice(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        n_correct = sum(1 for r in rows if r["correct"])
        return {
            "n": len(rows),
            "n_correct": n_correct,
            "score": (n_correct / len(rows)) if rows else 0.0,
            "G": _mean([r["G"] for r in rows]),
            "F": _mean([r["F"] for r in rows]),
            "T": _mean([r["T"] for r in rows]),
            "n_capped": sum(1 for r in rows if r["capped"]),
            "n_agent_failed": sum(1 for r in rows if r["agent_failed"]),
        }

    known = {t: [r for r in records if r["task_type"] == t] for t in _TASK_TYPE_ORDER}
    other = [r for r in records if r["task_type"] not in known]

    degradation: dict[str, int] = {}
    for row in records:
        kind = row["judge_degradation"]
        if kind:
            degradation[kind] = degradation.get(kind, 0) + 1

    return {
        "overall": _slice(records),
        "per_type": {t: _slice(rows) for t, rows in known.items() if rows},
        "per_type_other": _slice(other) if other else None,
        "degradation": dict(sorted(degradation.items())),
        "n_agent_failed": sum(1 for r in records if r["agent_failed"]),
        "n_degraded": sum(degradation.values()),
    }


# --------------------------------------------------------------------------------------
# 产物
# --------------------------------------------------------------------------------------


def _reason_summary(reason: str, width: int = 60) -> str:
    """失败样本表里的一格：截断 + 压掉换行（表格不能有多行单元格）。"""
    flat = " ".join(reason.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def render_report(records: Sequence[dict[str, Any]], *, split: str, agent_name: str,
                  judge_name: str = JUDGE_MODEL) -> str:
    """渲染 ``report.md``。**无时间戳**：同配置两次跑结构一致（LLM 分数允许不同）。

    表头用中文而 ASCII 分隔符，与 :mod:`calibrate` 的报告一致 —— 那份报告要同时含
    中文内容和 ASCII 表格框架，Windows 控制台代码页才不会把表格结构打花。
    """
    stats = summarize(records)
    total = stats["overall"]
    lines: list[str] = []
    lines.append(f"# B 场景真实 agent 跑分报告（split={split}）")
    lines.append("")
    lines.append(f"- agent：{agent_name}")
    lines.append(f"- judge 模型：{judge_name}")
    lines.append(f"- 任务数：{total['n']}")
    lines.append(f"- 主榜门槛：G ≥ {PASS_THRESHOLD_G}")
    lines.append("")
    lines.append("> 本报告**不含时间戳**：同 agent 配置两次跑，表格结构一致、分数允许不同（LLM 有随机性）。")
    lines.append("")
    lines.append("## 总分")
    lines.append("")
    lines.append(f"- 主榜正确率：{total['n_correct']}/{total['n']} = {total['score'] * 100:.2f}%")
    lines.append(f"- G 均值：{_fmt(total['G'])}")
    lines.append(f"- F 均值（忠实度）：{_fmt(total['F'])}")
    lines.append(f"- T 均值（工具协议合规）：{_fmt(total['T'])}")
    lines.append(f"- 关键点封顶（capped）：{total['n_capped']} 条")
    lines.append("")

    lines.append("## 按任务类型切片")
    lines.append("")
    lines.append("| 类型 | n | 正确率 | G | F | T | capped |")
    lines.append("|---|---|---|---|---|---|---|")
    slices = list(stats["per_type"].items())
    if stats["per_type_other"]:
        slices.append(("（其他）", stats["per_type_other"]))
    for name, s in slices:
        lines.append(
            f"| {name} | {s['n']} | {s['n_correct']}/{s['n']} = {s['score'] * 100:.1f}% "
            f"| {_fmt(s['G'])} | {_fmt(s['F'])} | {_fmt(s['T'])} | {s['n_capped']} |"
        )
    lines.append("")

    lines.append("## 异常计数（与「答错」分开记）")
    lines.append("")
    lines.append(f"- agent 失败（LLM 报错或步数用尽）：{stats['n_agent_failed']}")
    if stats["degradation"]:
        for kind, count in stats["degradation"].items():
            lines.append(f"- judge 降级 / `{kind}`：{count}")
        lines.append("")
        lines.append(f"**降级合计 {stats['n_degraded']} 条 —— 这些条的 G 是兜底的 0，"
                     "不是 agent 真答成了 0，上面的均值不可直接引用。**")
    else:
        lines.append("- judge 降级：无")
    lines.append("")

    lines.append("## 失败样本")
    lines.append("")
    wrong = [r for r in records if not r["correct"]]
    if not wrong:
        lines.append("无。")
    else:
        lines.append(f"共 {len(wrong)} 条。")
        lines.append("")
        lines.append("| task_id | 类型 | G | F | T | capped | agent 失败 | 原因摘要 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in wrong:
            lines.append(
                f"| {r['task_id']} | {r['task_type']} | {_fmt(r['G'])} | {_fmt(r['F'])} "
                f"| {_fmt(r['T'])} | {'是' if r['capped'] else '否'} "
                f"| {'是' if r['agent_failed'] else '否'} | {_reason_summary(r['reason'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    records: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    split: str,
    agent_name: str,
    judge_name: str = JUDGE_MODEL,
) -> Path:
    """写 ``records.jsonl`` + ``report.md``，返回 report 路径。

    **每行盖 ``agent`` / ``judge`` 署名**（在序列化这一刻盖，不动传进来的 records ——
    ``summarize`` / 榜单聚合的口径零影响）。以前两份不同 judge 的产物只能靠
    ``report.md`` 区分，``records.jsonl`` 单独拿走就答不出「谁跑的、谁判的」。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            signed = {**record, "agent": agent_name, "judge": judge_name}
            handle.write(json.dumps(_json_safe(signed), ensure_ascii=False, sort_keys=True) + "\n")
    report_path = out_dir / "report.md"
    report_path.write_text(
        render_report(records, split=split, agent_name=agent_name, judge_name=judge_name),
        encoding="utf-8", newline="\n",
    )
    return report_path


# --------------------------------------------------------------------------------------
# C 场景：统计 / 产物
# --------------------------------------------------------------------------------------

#: C 场景的 delta 三段 —— 报告的**主体**。三段分开记而不是只记一个 ``delta_ok``：
#: 「把该改的改错了」和「不该动的动了」是两种完全不同的失败，
#: 合成一个布尔就把它们压成同一件事了（RESULT-019 §5.2）。
_DELTA_SEGMENTS = ("orders", "coupons", "must_not_change")


def summarize_orchestration(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """C 场景统计：总分 + 按类型切片 + **三段各自**的通过数。"""
    def _slice(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        return {
            "n": n,
            "n_correct": sum(1 for r in rows if r["correct"]),
            "score": (sum(1 for r in rows if r["correct"]) / n) if n else 0.0,
            "n_delta_ok": sum(1 for r in rows if r["delta_ok"]),
            "n_reply_ok": sum(1 for r in rows if r["reply_ok"]),
            "segments": {seg: sum(1 for r in rows if r[f"{seg}_ok"]) for seg in _DELTA_SEGMENTS},
            "G": _mean([r["G"] for r in rows]),
            "n_agent_failed": sum(1 for r in rows if r["agent_failed"]),
        }

    known = {t: [r for r in records if r["task_type"] == t]
             for t in TYPE_LABELS}
    other = [r for r in records if r["task_type"] not in known]
    return {
        "overall": _slice(records),
        "per_type": {t: _slice(rows) for t, rows in known.items() if rows},
        "per_type_other": _slice(other) if other else None,
    }


def render_orchestration_report(
    records: Sequence[dict[str, Any]], *, agent_name: str, seed: int, per_type: int,
    judge_name: str = JUDGE_MODEL,
) -> str:
    """渲染 C 场景的 ``report.md``。**无时间戳**，与 B 场景同一承诺。"""
    stats = summarize_orchestration(records)
    total = stats["overall"]
    lines: list[str] = []
    lines.append("# C 场景真实 agent 跑分报告（多工具编排）")
    lines.append("")
    lines.append(f"- agent：{agent_name}")
    lines.append(f"- judge 模型：{judge_name}")
    lines.append(f"- 任务集：seed={seed} per_type={per_type}（共 {total['n']} 条）")
    lines.append(f"- 回复点门槛：G ≥ {PASS_THRESHOLD_C}")
    lines.append("")
    lines.append("> 本报告**不含时间戳**：同 agent 配置两次跑，表格结构一致、分数允许不同。")
    lines.append("> **规模极小（8 条），只验通路，任何准确率数字都不具统计意义**（RESULT-019 §5.3）。")
    lines.append("")

    lines.append("## 总分")
    lines.append("")
    lines.append(f"- 主榜正确率：{total['n_correct']}/{total['n']} = {total['score'] * 100:.2f}%")
    lines.append(f"- delta 三段全中：{total['n_delta_ok']}/{total['n']}")
    lines.append(f"- 回复质量达标：{total['n_reply_ok']}/{total['n']}")
    lines.append(f"- 回复点 G 均值：{_fmt(total['G'])}")
    lines.append("")

    lines.append("## delta 三段明细（全量）")
    lines.append("")
    lines.append("| 段 | 通过 | 含义 |")
    lines.append("|---|---|---|")
    meaning = {
        "orders": "该改成目标态的订单都改对了",
        "coupons": "该变成目标态的优惠券都变了",
        "must_not_change": "该保持原样的对象一个都没被动过",
    }
    for seg in _DELTA_SEGMENTS:
        lines.append(f"| {seg} | {total['segments'][seg]}/{total['n']} | {meaning[seg]} |")
    lines.append("")

    lines.append("## 按任务类型切片")
    lines.append("")
    lines.append("| 类型 | n | 正确率 | delta 全中 | 回复达标 | orders | coupons | must_not_change | G |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    slices = list(stats["per_type"].items())
    if stats["per_type_other"]:
        slices.append(("（其他）", stats["per_type_other"]))
    for name, s in slices:
        seg = s["segments"]
        lines.append(
            f"| {name} | {s['n']} | {s['n_correct']}/{s['n']} = {s['score'] * 100:.1f}% "
            f"| {s['n_delta_ok']}/{s['n']} | {s['n_reply_ok']}/{s['n']} "
            f"| {seg['orders']}/{s['n']} | {seg['coupons']}/{s['n']} "
            f"| {seg['must_not_change']}/{s['n']} | {_fmt(s['G'])} |"
        )
    lines.append("")

    lines.append("## 失败样本")
    lines.append("")
    wrong = [r for r in records if not r["correct"]]
    if not wrong:
        lines.append("无。")
    else:
        lines.append(f"共 {len(wrong)} 条。")
        lines.append("")
        lines.append("| task_id | 类型 | delta | 回复 | orders | coupons | must_not_change | 原因摘要 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in wrong:
            mark = lambda v: "✓" if v else "✗"  # noqa: E731 —— 表格里就地一行写完更清楚
            lines.append(
                f"| {r['task_id']} | {r['task_type']} | {mark(r['delta_ok'])} "
                f"| {mark(r['reply_ok'])} | {mark(r['orders_ok'])} | {mark(r['coupons_ok'])} "
                f"| {mark(r['must_not_change_ok'])} | {_reason_summary(r['reason'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_outputs_orchestration(
    records: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    agent_name: str,
    seed: int,
    per_type: int,
    judge_name: str = JUDGE_MODEL,
) -> Path:
    """写 C 场景的 ``records.jsonl`` + ``report.md``，返回 report 路径。署名口径与 B 场景一致。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            signed = {**record, "agent": agent_name, "judge": judge_name}
            handle.write(json.dumps(_json_safe(signed), ensure_ascii=False, sort_keys=True) + "\n")
    report_path = out_dir / "report.md"
    report_path.write_text(
        render_orchestration_report(records, agent_name=agent_name, seed=seed,
                                    per_type=per_type, judge_name=judge_name),
        encoding="utf-8", newline="\n",
    )
    return report_path


#: ``--split orchestration`` 时的任务集参数。C 场景**默认档**仍按 ``(seed, per_type)``
#: **现算** —— 与 ``benchmarks/orchestration/tasks/trial/orchestration.jsonl`` 那份入库产物
#: 同种子同参数时逐字节一致。调档走 ``--preset``，那条路读文件（见 :func:`load_orchestration_preset`）。
_C_SEED = 42
_C_PER_TYPE = 2

#: C 场景正式集的三档。**缺省 ``trial``** —— 老命令行一个字不加仍是原行为（向后兼容）。
_C_PRESETS = ("trial", "dev", "blind")

#: 封存档名。挡住这个档需要显式 ``--allow-blind``（TASK-029）。
BLIND = "blind"

#: 拒绝读 blind 时给的那句人话。**必须点名 blind 与 ``--allow-blind``** ——
#: 报错说不出怎么解，调用方就只能去翻源码。
_BLIND_REFUSED = (
    "blind 是封存集，开发流程请用 dev；确要跑请加 --allow-blind"
)

#: 封存清单：``blind`` 档产物/任务集的 sha256。校验入口
#: ``python -m zh_agent_bench.eval_run --verify-blind-manifest``。
_BLIND_MANIFEST = Path("benchmarks/blind-manifest.json")

#: 清单覆盖的目录。``qa`` 侧任务集与产物同目录，``orchestration`` 侧任务集在 ``tasks/`` 下。
_BLIND_ROOTS = ("benchmarks/qa/blind", "benchmarks/orchestration/blind",
               "benchmarks/orchestration/tasks/blind")

#: C 场景正式集任务集根目录。默认仓库 ``benchmarks/orchestration/tasks``。
_C_TASKS_ROOT = Path("benchmarks/orchestration/tasks")


def check_blind_allowed(grade: str, *, allow_blind: bool) -> None:
    """blind 是封存集：没显式 ``--allow-blind`` 就抛 :class:`PermissionError`。

    **抛异常而不是返回 False** —— 调用方遍布三个入口，每个都自己记得检查一遍，
    早晚漏一个；漏的那个就是「开发流程能随手读到 blind」，那正是本单要消灭的。
    """
    if grade == BLIND and not allow_blind:
        raise PermissionError(_BLIND_REFUSED)


def _blind_files(root: Path | None = None) -> list[Path]:
    """清单该覆盖的 ``*.jsonl``（相对 :data:`_BLIND_ROOTS`），排序后返回。

    排序是确定性的 —— 清单与校验输出都逐字节可复现。
    """
    base = Path(root) if root is not None else Path(".")
    found: list[Path] = []
    for rel in _BLIND_ROOTS:
        found.extend(sorted((base / rel).rglob("*.jsonl")))
    return found


def blind_manifest_payload(root: Path | None = None) -> dict[str, str]:
    """当前磁盘上的 blind 清单：``{"<仓库相对路径>": "<sha256>"}``。"""
    base = Path(root) if root is not None else Path(".")
    return {
        path.relative_to(base).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _blind_files(base)
    }


def build_blind_manifest(root: Path | None = None) -> str:
    """生成清单文本。**缩进 + 排好序**，同磁盘两次生成逐字节一致。"""
    payload = blind_manifest_payload(root)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def verify_blind_manifest(manifest: Path | None = None, root: Path | None = None) -> list[str]:
    """核验清单与磁盘一致，返回**差异列表**（空 = 一致）。

    三种差异都算不一致：内容变了、清单里有磁盘上没有的、磁盘上有清单里没有的 ——
    后两种正是「blind 产物被换掉了」和「有人绕过清单加了一份」。
    """
    path = Path(manifest) if manifest is not None else _BLIND_MANIFEST
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{path.as_posix()} 读不了：{exc}"]
    if not isinstance(recorded, dict):
        return [f"{path.as_posix()} 顶层不是 {{路径: sha256}}"]

    actual = blind_manifest_payload(root)
    diffs: list[str] = []
    for name in sorted(set(recorded) | set(actual)):
        want, got = recorded.get(name), actual.get(name)
        if want == got:
            continue
        if name not in actual:
            diffs.append(f"清单里有、磁盘上没有：{name}")
        elif name not in recorded:
            diffs.append(f"磁盘上有、清单里没有：{name}（改了 blind 产物必须同步更新清单）")
        else:
            diffs.append(f"内容变了：{name}（清单 {want[:12]}… / 磁盘 {got[:12]}…）")
    return diffs

def load_orchestration_preset(preset: str, data_root: Path | None = None) -> list[Task]:
    """读 C 场景正式集某一档的 ``orchestration.jsonl`` → ``list[Task]``。

    :param preset: :data:`_C_PRESETS` 之一（大小写敏感，与目录名一字对应）。
    :param data_root: 任务集根目录，默认 :data:`_C_TASKS_ROOT`；给定时读
        ``<data_root>/<preset>/orchestration.jsonl``。

    与 :func:`calibrate.load_formal_split` 同一套骨架：只负责读，**不碰条数** ——
    60/120/150（C 侧 8/16/16）由测试钉死，加载器硬编码条数只会变成绊脚石。

    档位名落在 ``initial_data["评测档位"]`` 而非 ``split`` 字段
    （``schema.Split`` 只有 TRAIN/HELDOUT 两值），读档位要读那里。
    """
    if preset not in _C_PRESETS:
        raise ValueError(f"--preset 必须是 {_C_PRESETS} 之一，收到 {preset!r}")
    root = Path(data_root) if data_root is not None else _C_TASKS_ROOT
    path = root / preset / "orchestration.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 C 场景正式集 {path}。若产物未入库，先跑："
            f"python -m zh_agent_bench.tasks.orchestration --seed 42 --preset formal "
            f"--output-dir {root}"
        )
    return [
        Task.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_orchestration_cli(args: argparse.Namespace) -> int:
    """``--split orchestration`` 的执行体。

    ``--preset trial``（缺省）任务集**现算**（:func:`build_orchestration_dataset`）而不是读 jsonl：
    入库那份是给发布的，跑分要能临时换 seed / per_type 试而不必重生成产物。
    ``--preset dev|blind`` 则**按档读入库的正式集**（:func:`load_orchestration_preset`）——
    交分要能指明「跑的是哪一档」，而档位名写在产物路径里，不是算出来的。
    """
    # 闸**单独一个 except**：封存拒绝不是「任务集构造失败」，两者混在一句人话里，
    # 交分的人会以为是自己路径写错了，去翻数据而不是加 --allow-blind。
    try:
        check_blind_allowed(args.preset, allow_blind=args.allow_blind)
    except PermissionError as exc:
        print(f"封存闸拒绝：{exc}")
        return 2
    try:
        if args.preset == "trial":
            tasks = build_orchestration_dataset(per_type=args.per_type, seed=args.seed)
        else:
            tasks = load_orchestration_preset(args.preset, args.data_root)
    except (ValueError, OSError) as exc:
        print(f"C 场景任务集构造失败：{exc}")
        return 2
    if args.limit > 0:
        tasks = tasks[: args.limit]

    # 与 B 场景同一条纪律：真 agent 到这里才 import。
    agent = build_agent(args)
    records = run_split_orchestration(tasks, agent, args.judge_fn)

    out_dir = args.out or default_out_dir("C", args.preset, agent.name)
    report = write_outputs_orchestration(
        records, out_dir, agent_name=agent.name, seed=args.seed, per_type=args.per_type,
        judge_name=args.judge_name,
    )
    total = summarize_orchestration(records)["overall"]
    print(
        f"split=orchestration agent={agent.name} tasks={total['n']} "
        f"correct={total['n_correct']}/{total['n']} "
        f"delta_ok={total['n_delta_ok']}/{total['n']} "
        f"reply_ok={total['n_reply_ok']}/{total['n']} "
        f"orders={total['segments']['orders']} coupons={total['segments']['coupons']} "
        f"must_not_change={total['segments']['must_not_change']} "
        f"G={_fmt(total['G'])} agent_failed={total['n_agent_failed']}"
    )
    print(f"report -> {report}")
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 的 parser。

    抽成独立函数是为了让**测试能拿到与命令行完全同构的默认值** ——
    开关的默认值写在别处（测试里另建一个 parser）迟早会漂。
    """
    parser = argparse.ArgumentParser(
        prog="python -m zh_agent_bench.eval_run",
        description="跑 B 场景真实 LLM agent：G/F/T + 按类型切片 + 失败样本；"
                    "--split orchestration 跑 C 场景：delta 三段 + 回复质量",
    )
    parser.add_argument("--split", default="trial",
                        choices=("trial", "dev", "blind", "orchestration"),
                        help="B 场景正式集 split（默认 trial）；orchestration = C 场景多工具编排")
    parser.add_argument("--out", type=Path, default=None,
                        help="输出目录（写 records.jsonl + report.md）。"
                             "缺省按 benchmarks/<场景>/<档>/<agent>/ 自动派生 —— 两个 agent "
                             "跑同一档因此各落各的目录，不会互相覆盖。显式给了就原样用")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="正式集根目录：B 场景默认仓库 benchmarks/qa，"
                             "C 场景（--split orchestration --preset dev|blind）默认 benchmarks/orchestration/tasks")
    parser.add_argument("--model", default=None, help="被评 agent 的模型 id（默认走 llm_client.DEFAULT_MODEL）")
    parser.add_argument("--agent-base-url", default=None,
                        help="被评 agent 的 API 端点（缺省 = llm_client 的默认端点）。"
                             "换通道跑第二个 agent 时用它，与 --judge-base-url 同构")
    parser.add_argument("--agent-key-env", default=API_KEY_ENV,
                        help=f"被评 agent 的密钥取自哪个环境变量（缺省 = {API_KEY_ENV}）。"
                             "换通道时传另一个名字，如 LLMAPI_GPT_API_KEY")
    parser.add_argument("--agent-api-key", default=None,
                        help="直接给被评 agent 的密钥（不鼓励 —— 会进 shell 历史与进程列表）。"
                             "优先用 --agent-key-env")
    parser.add_argument("--limit", type=int, default=0,
                        help="只跑前 N 条（0 = 全跑）。调试用，正式报告不要用")
    parser.add_argument("--save-answers", action="store_true",
                        help="records.jsonl 每条多带一个 answer 字段（agent 原始答案），"
                             "供 annotate 的双标用。不带则产物与原来逐字节一致")
    parser.add_argument("--per-type", type=int, default=_C_PER_TYPE,
                        help="C 场景每种任务类型的条数（默认 2，即共 8 条；--preset dev|blind 时忽略）")
    parser.add_argument("--seed", type=int, default=_C_SEED,
                        help="C 场景任务集的沙箱 seed（默认 42，与入库产物同种子；--preset dev|blind 时忽略）")
    parser.add_argument("--preset", default="trial", choices=_C_PRESETS,
                        help="C 场景跑哪一档（默认 trial = 按 (seed, per_type) 现算，向后兼容；"
                             "dev / blind 读 benchmarks/orchestration/tasks/<档>/orchestration.jsonl 入库正式集）")
    parser.add_argument("--judge-model", default=None,
                        help="judge 的模型 id（缺省 = judge.JUDGE_MODEL，即与被评 agent 同模型）。"
                             "换 judge 就是为了量「同模型自评有没有偏置」")
    parser.add_argument("--judge-base-url", default=None,
                        help="judge 的 API 端点（缺省 = llm_client 的默认端点）")
    parser.add_argument("--judge-key-env", default=API_KEY_ENV,
                        help=f"judge 的密钥取自哪个环境变量（缺省 = {API_KEY_ENV}）。"
                             "换 judge 通道时传另一个名字，如 LLMAPI_API_KEY")
    parser.add_argument("--allow-blind", action="store_true",
                        help="放行读 blind 档（--split blind / --preset blind）。"
                             "blind 是封存集，缺省拒绝：开发流程能随手读到它，"
                             "就等于允许对着盲测调参。交分或核验封存时才加")
    parser.add_argument("--verify-blind-manifest", action="store_true",
                        help=f"核验 {_BLIND_MANIFEST.as_posix()} 与磁盘一致，不跑 agent。"
                             "一致 exit 0，不一致 exit 2 并逐条列出差异")
    parser.add_argument("--write-blind-manifest", action="store_true",
                        help=f"按当前磁盘重生成 {_BLIND_MANIFEST.as_posix()}。"
                             "**改 blind 产物后跑一次** —— 不更新清单，校验就会红")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # 两个清单入口在最前面：不带任何 split/preset 也能单独跑，也不碰 agent。
    if args.write_blind_manifest:
        text = build_blind_manifest()
        _BLIND_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        _BLIND_MANIFEST.write_text(text, encoding="utf-8", newline="\n")
        print(f"已写 {_BLIND_MANIFEST.as_posix()}（{len(blind_manifest_payload())} 份文件）")
        return 0

    if args.verify_blind_manifest:
        diffs = verify_blind_manifest()
        if not diffs:
            print(f"blind 封存清单与磁盘一致（{len(blind_manifest_payload())} 份文件）")
            return 0
        print(f"blind 封存清单与磁盘不一致，共 {len(diffs)} 处：")
        for diff in diffs:
            print(f"  {diff}")
        return 2

    # judge 注入：**三个都不传就一个 dict 都不建**，产物与从前逐字节一致。
    judge_kwargs: dict[str, Any] = {}
    if args.judge_model:
        judge_kwargs["model"] = args.judge_model
    client_kwargs: dict[str, Any] = {}
    if args.judge_base_url:
        client_kwargs["base_url"] = args.judge_base_url
    if args.judge_key_env != API_KEY_ENV:
        client_kwargs["api_key"] = os.environ.get(args.judge_key_env, "")
    if client_kwargs:
        judge_kwargs["client_kwargs"] = client_kwargs

    def _judge(task: Any, answer: Any, **call_kwargs: Any) -> dict[str, Any]:
        return judge_task(task, answer, **call_kwargs, **judge_kwargs)

    args.judge_fn = _judge
    args.judge_name = args.judge_model or JUDGE_MODEL

    if args.split == "orchestration":
        return _run_orchestration_cli(args)

    try:
        check_blind_allowed(args.split, allow_blind=args.allow_blind)
        tasks = load_formal_split(args.split, args.data_root)
    except PermissionError as exc:
        print(f"封存闸拒绝：{exc}")
        return 2
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover —— parser.error 不返回
    if args.limit > 0:
        tasks = tasks[: args.limit]

    # 真 agent 在这里才 import（而不是模块顶部）：--help 与失败路径不该拖着
    # agents/llm.py 一起加载。
    agent = build_agent(args)
    records = run_split(tasks, agent, args.judge_fn, save_answer=args.save_answers)

    out_dir = args.out or default_out_dir("B", args.split, agent.name)
    report = write_outputs(records, out_dir, split=args.split, agent_name=agent.name,
                           judge_name=args.judge_name)
    stats = summarize(records)
    total = stats["overall"]
    print(
        f"split={args.split} agent={agent.name} tasks={total['n']} "
        f"correct={total['n_correct']}/{total['n']} "
        f"G={_fmt(total['G'])} F={_fmt(total['F'])} T={_fmt(total['T'])} "
        f"capped={total['n_capped']} agent_failed={stats['n_agent_failed']} "
        f"judge_degraded={stats['n_degraded']}"
    )
    print(f"report -> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
