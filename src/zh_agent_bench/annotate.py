"""annotate —— **LLM 辅助双标**（TASK-017）。

双标回答一个问题：**judge 判得准不准，脱离它自己的口径**。

两个**独立**标注者（``gpt-6-astra`` 走 LLMAPI 通道、``deepseek-chat`` 走默认通道）
各自对同一批（任务 × agent 答案）逐条 rubric 打分，只回 ``{criterion_id: 0|0.5|1}``：

- **不给总分** —— 与 :mod:`judge` 同一条纪律：总分不可回溯到丢分点；
- **不看对方的分、不看 judge 的分、不看 agent 身份** —— prompt 里压根没有这些字段，
  不是「叮嘱模型别看」，是它拿不到；
- **两次调用互不参考** —— 两次独立 :func:`llm_client.chat`，不共享上下文。

产出三组数（:func:`summarize`）：

1. **标注者间一致率**（逐点）**+ Cohen's κ** —— 人类一致性代理；
2. **judge vs 共识** —— judge 的 G 与「A/B 共识 G」的差、通过率翻转、分歧样本；
3. **降级计数** —— 某标注者失败或给出非法分值的条数。

**不要求逐字节可复现**：LLM 有随机性，同配置两次跑报告**结构**一致、分数允许不同。

**运行时仅 stdlib + 本包**，与全项目一致。

用法::

    python -m zh_agent_bench.eval_run --split blind --save-answers --out <dir>
    python -m zh_agent_bench.annotate --split blind --records <dir>/records.jsonl \\
        --out benchmarks/qa/calibration-report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from . import llm_client
from .calibrate import load_formal_split
from .judge import PASS_THRESHOLD_G, build_judge_prompt

#: 标注者的合法三档。与 :data:`judge.LEGAL_POINT_SCORES` 同一个取值域 ———
#: 两边口径若不同，「judge vs 标注」比的就是两套标准而不是两个判分者。
LABELS: tuple[float, ...] = (0.0, 0.5, 1.0)

#: 标注者登记表。**通道与 key 环境变量分开登记**是为了让「两个标注者真的走了两个
#: 服务」这件事在代码里看得见 —— 同厂商两个模型不是独立标注，是同一个脑子。
ANNOTATORS: dict[str, dict[str, str]] = {
    "a": {
        "name": "gpt-6-astra",
        "api_key_env": "LLMAPI_API_KEY",
        "base_url_env": "LLMAPI_BASE_URL",
    },
    "b": {
        "name": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "",          # 留空 = 用 llm_client 的默认端点
    },
}


# --------------------------------------------------------------------------------------
# prompt：复用 judge 的 rubric 渲染，只换输出契约
# --------------------------------------------------------------------------------------


#: judge prompt 的 user 消息以这一行结尾，其后是要输出的 JSON 结构说明。
_OUTPUT_MARKER = "## 你要输出的 JSON 结构"


def build_annotator_prompt(task: Any, answer: Any) -> list[dict[str, Any]]:
    """渲染标注者 prompt。**rubric 渲染整段复用 :func:`judge.build_judge_prompt`**。

    只把末尾那段「要输出的 JSON 结构」换掉：judge 要 scores + claims + notes，
    标注者只要 scores —— claims 与忠实度是 judge 的活，混进标注只会让两个标注者
    先在「什么算一条 claim」上吵起来，而吵的内容跟一致率毫无关系。

    复用而不是重写：rubric 渲染（权重、关键点、半分规则、证据原文、等价表述）
    一旦抄第二份，两边就会漂，而漂了之后 A/B 分歧到底是「标注者有主见」还是
    「两边看到的不是同一份细则」根本分不开。
    """
    messages = [dict(m) for m in build_judge_prompt(task, answer)]
    user = messages[-1]["content"]
    cut = user.find(_OUTPUT_MARKER)
    if cut >= 0:
        user = user[:cut].rstrip()
    messages[-1]["content"] = (
        user
        + "\n\n## 你要输出的 JSON 结构\n"
        + '{"scores": {"<criterion_id>": 0 | 0.5 | 1, ...}, '
        + '"notes": "<一句话总评>"}\n'
        + "只输出 JSON 对象，不要输出任何解释文字或 Markdown 代码块标记。"
        + "每个评分点都必须给出分值，一个都不能漏。"
    )
    return messages


# --------------------------------------------------------------------------------------
# 单条标注
# --------------------------------------------------------------------------------------


def parse_scores(rubric: Sequence[dict[str, Any]], raw: Any) -> tuple[dict[str, float] | None, str]:
    """把标注者回的 ``scores`` 收敛成「每个 criterion_id 都有 0/0.5/1」。

    与 :func:`judge._collect_scores` **故意不同**：judge 漏判补 0（答案没提到就是 0 分），
    标注者漏判是**降级**（标注者没干活，拿 0 当成「他觉得这点没答到」就是伪造意见）。
    所以这里任何一个点缺分值 / 非法分值，整条作废并说明原因，不猜、不补。
    """
    if not isinstance(raw, dict):
        return None, f"scores 不是对象（收到 {type(raw).__name__}）"
    scores: dict[str, float] = {}
    for point in rubric:
        cid = point.get("criterion_id")
        value = raw.get(cid)
        # bool 必须先挡掉：``float(True) == 1.0``，标注者回了个 ``true`` 会被当成满分。
        if isinstance(value, bool) or value is None:
            return None, f"{cid}: 未给出分值（收到 {value!r}）"
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None, f"{cid}: 分值 {value!r} 不是数字"
        if number not in LABELS:
            return None, f"{cid}: 分值 {number!r} 不在 0/0.5/1 内"
        scores[cid] = number
    return scores, ""


def annotate_one(
    task: Any,
    answer: Any,
    *,
    model: str,
    call: Callable[..., dict],
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """标一条。**永不抛异常** —— 失败落成 ``ok=False`` + 原因，与 judge 的降级同形。

    :param call: ``(messages, model=…, base_url=…, api_key=…, temperature=…) -> dict``。
        测试注入假 call，不联网。
    """
    record: dict[str, Any] = {
        "task_id": task.id, "model": model, "scores": None, "ok": False, "reason": "",
    }
    try:
        response = call(
            build_annotator_prompt(task, answer),
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 —— mock 抛异常 / 未来的 SDK 变更，一律降级
        record["reason"] = f"标注调用失败：{type(exc).__name__}: {exc}"
        return record
    if not isinstance(response, dict) or "error" in response:
        err = response.get("error") if isinstance(response, dict) else None
        detail = err.get("message") if isinstance(err, dict) else str(response)
        record["reason"] = f"标注调用失败：{detail}"
        return record
    message = llm_client.first_message(response)
    if message is None:
        record["reason"] = "标注返回没有 choices[0].message"
        return record
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        record["reason"] = "标注返回 content 为空"
        return record

    # 复用 judge 的「剥 ```json 围栏再抠对象」——模型爱包围栏，两边各抄一份
    # 迟早有一边忘了剥，把一份完全正确的打分扔了。
    from .judge import _loads_strict

    payload = _loads_strict(content)
    if payload is None:
        record["reason"] = "标注返回的不是合法 JSON"
        return record
    scores, why = parse_scores(task.rubric or (), payload.get("scores"))
    if scores is None:
        record["reason"] = why
        return record
    record["scores"] = scores
    record["ok"] = True
    return record


def _default_call(messages: list[dict[str, Any]], **kwargs: Any) -> dict:
    """默认 call：转 :func:`llm_client.chat`。测试用假 call 替掉它，不联网。"""
    base_url = kwargs.pop("base_url", None) or llm_client.DEFAULT_BASE_URL
    return llm_client.chat(messages, base_url=base_url, **kwargs)


# --------------------------------------------------------------------------------------
# 一致率
# --------------------------------------------------------------------------------------


def _nearest_label(value: float) -> float:
    """把共识分值吸到最近的合法档。

    两个标注者给 0 与 0.5 时中点是 0.25 —— 合法档里没有它。不吸的话共识就没法过
    :func:`tasks.qa.aggregate_rubric`（那边对分值非法直接抛），整批统计作废。
    平手（正好 0.25 / 0.75）取低档：``min`` 遇并列返回先出现的那一个，行为确定。
    """
    return min(LABELS, key=lambda label: (abs(label - value), label))


def consensus_scores(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    """A/B 共识：一致就取该值，不一致取两者中点并吸到合法档。

    「不一致就不给共识」听上去更严谨，实际是把难题全踢出统计 —— 而**恰恰是 A/B
    分歧的那些点才是 judge 该被检验的地方**。取中点是有偏的（有向低档），
    所以 :func:`summarize` 同时报 ``n_consensus_by_midpoint``，让人知道
    共识里有几个点是靠「折中」来的。
    """
    return {cid: _nearest_label((a[cid] + b[cid]) / 2) for cid in a if cid in b}


def cohens_kappa(pairs: Sequence[tuple[float, float]]) -> float | None:
    """Cohen's κ。无定义时返回 ``None``（而不是 0）。

    两种无定义情形：样本为空；**两人都只给同一档**（期望一致率 = 1，公式 0/0）。
    这两种都返回 ``None`` 是刻意的 —— 报 0 会被读成「两人毫无共识」，
    而真相是「这批样本没有分歧可言」或「压根没数据」。两者在报告里是完全不同的结论。
    """
    n = len(pairs)
    if not n:
        return None
    labels = sorted({x for pair in pairs for x in pair})
    if len(labels) < 2:
        return None
    observed = sum(1 for x, y in pairs if x == y) / n
    count_a = Counter(x for x, _ in pairs)
    count_b = Counter(y for _, y in pairs)
    expected = sum((count_a[l] / n) * (count_b[l] / n) for l in labels)
    if expected >= 1.0:  # 浮点上期望一致率也可能贴到 1.0，防御
        return None
    return (observed - expected) / (1 - expected)


def point_pairs(rows: Iterable[tuple[dict[str, float], dict[str, float]]]) -> list[tuple[float, float]]:
    """把 ``(A 的分, B 的分)`` 摊平成逐点对。按 criterion_id 排序 —— κ 与一致率
    都与顺序无关，但报告里逐点表按 id 走，两次跑才能逐行比。"""
    return [(a[cid], b[cid]) for a, b in rows for cid in sorted(a) if cid in b]


def aggregate_g(rubric: Sequence[dict[str, Any]], scores: dict[str, float]) -> float | None:
    """把逐点分聚合成 G。**聚合复用生成期那一份**（:func:`tasks.qa.aggregate_rubric`），
    判定者与被判定者必须用同一个加权 + 关键点封顶口径，否则「judge vs 共识」比的是
    两套聚合规则。失败返回 ``None`` 而不是抛 —— 一条坏样本不该掀掉整份报告。"""
    try:
        from .tasks.qa import aggregate_rubric  # 局部 import：qa 会拉进生成期依赖

        g, _capped = aggregate_rubric(list(rubric), scores)
    except (ValueError, KeyError, ZeroDivisionError, TypeError):
        return None
    return g


# --------------------------------------------------------------------------------------
# 跑一批
# --------------------------------------------------------------------------------------


def _task_type(task: Any) -> str:
    value = (task.initial_data or {}).get("任务类型")
    return value if isinstance(value, str) and value else "unknown"


def run_double_annotation(
    items: Sequence[tuple[Any, Any]],
    *,
    call_a: Callable[..., dict],
    call_b: Callable[..., dict],
    model_a: str,
    model_b: str,
    key_a: str | None = None,
    url_a: str | None = None,
    key_b: str | None = None,
    url_b: str | None = None,
) -> list[dict[str, Any]]:
    """对一批 ``(task, answer)`` 跑双标，返回每条的并列记录。

    **A 与 B 是两次独立调用**：不共享 messages、不共享上下文、不把对方的返回喂回去。
    顺序固定「先 A 后 B」，所以同一批的记录顺序天然确定。

    行里带一份 ``rubric``（纯 dict，可序列化）—— :func:`summarize` 要拿它算共识 G，
    而把 :class:`Task` 整个塞进行里会让这批记录没法直接写 jsonl。
    """
    rows: list[dict[str, Any]] = []
    for task, answer in items:
        row_a = annotate_one(task, answer, model=model_a, call=call_a, api_key=key_a, base_url=url_a)
        row_b = annotate_one(task, answer, model=model_b, call=call_b, api_key=key_b, base_url=url_b)
        rows.append({
            "task_id": task.id,
            "task_type": _task_type(task),
            "rubric": [dict(p) for p in (task.rubric or ())],
            "a": row_a,
            "b": row_b,
        })
    return rows


def load_answer_records(
    split: str,
    records_path: Path,
    *,
    data_root: Path | None = None,
    limit: int = 0,
) -> list[tuple[Any, Any]]:
    """把 ``eval_run --save-answers`` 的 ``records.jsonl`` 对回任务，返回 ``(task, answer)``。

    对不上的 ``task_id`` 直接报错而不是跳过：产物与正式集不同源时（最常见的原因是
    跑的是冻结前的旧 sha）静默跳过会让一致率看着正常而实际样本对不上，
    而「样本对不上」是双标里最致命、也最不容易被看出来的一类错。
    """
    if not records_path.exists():
        raise FileNotFoundError(
            f"找不到 {records_path}；双标需要带 answer 的 records，先跑 "
            f"`python -m zh_agent_bench.eval_run --split {split} --save-answers --out <dir>`"
        )
    answers: dict[str, Any] = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "answer" not in row:
            raise ValueError(f"{records_path} 里的记录没有 answer 字段 —— 是漏了 --save-answers？")
        answers[row["task_id"]] = row["answer"]

    tasks = load_formal_split(split, data_root)
    missing = [t.id for t in tasks if t.id not in answers]
    if missing:
        raise ValueError(
            f"records 缺 {len(missing)} 条正式集任务（如 {missing[0]}）—— "
            "产物与正式集不同源？确认跑的是冻结后那一版"
        )
    pairs = [(task, answers[task.id]) for task in tasks]
    return pairs[:limit] if limit > 0 else pairs


# --------------------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def judge_vs_consensus(
    rows: Sequence[dict[str, Any]], judge_records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """judge 的 G vs「A/B 共识 G」：一致率、平均绝对误差、通过率翻转、分歧样本。

    比 **G** 而不是逐点分：榜单口径就是 G，而逐点分要先假设 judge 的 scores 拿得到
    （``eval_run`` 的 records 不落 scores）。逐点一致已在另一块报了，两块合起来才
    既能定位「哪类题判不准」又能给出「榜单上差多少」。
    """
    judge_g = {
        r["task_id"]: r.get("G") for r in judge_records
        if isinstance(r.get("G"), (int, float)) and not isinstance(r.get("G"), bool)
    }
    diffs: list[float] = []
    flips = 0
    n = 0
    divergences: list[dict[str, Any]] = []
    for row in rows:
        task_id = row["task_id"]
        merged = consensus_scores(row["a"]["scores"], row["b"]["scores"])
        g_consensus = aggregate_g(row["rubric"], merged)
        g_judge = judge_g.get(task_id)
        if g_consensus is None or g_judge is None:
            continue
        n += 1
        diff = g_judge - g_consensus
        diffs.append(abs(diff))
        if (g_judge >= PASS_THRESHOLD_G) != (g_consensus >= PASS_THRESHOLD_G):
            flips += 1
        if abs(diff) > 0.2:
            divergences.append({
                "task_id": task_id, "task_type": row["task_type"],
                "judge_G": g_judge, "consensus_G": g_consensus,
            })
    return {
        "n": n,
        "agreement": (sum(1 for d in diffs if d < 0.05) / n) if n else None,
        "mean_abs_error": (sum(diffs) / n) if n else None,
        "n_pass_flip": flips,
        "pass_flip_rate": (flips / n) if n else None,
        "n_divergence": len(divergences),
        "divergences": divergences,
    }


def summarize(
    rows: Sequence[dict[str, Any]],
    *,
    judge_records: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """汇成双标报告的结构化统计（报告的三块内容）。

    **一致率只统计「两边都成功」的条目** —— 把降级条目的分值当 0 算进一致率，
    等于把「标注者没干活」记成「他觉得这点不行」，降级会伪装成一致或不一致。
    降级条数单列在 ``n_degraded_a/b``，一眼看得到有多少条被排除在外。
    """
    both_ok = [r for r in rows if r["a"]["ok"] and r["b"]["ok"]]
    pairs = point_pairs([(r["a"]["scores"], r["b"]["scores"]) for r in both_ok])
    n_points = len(pairs)
    agree = sum(1 for x, y in pairs if x == y)

    by_midpoint = 0
    for row in both_ok:
        merged = consensus_scores(row["a"]["scores"], row["b"]["scores"])
        if any(merged[cid] != row["a"]["scores"][cid] for cid in merged):
            by_midpoint += 1

    stats: dict[str, Any] = {
        "n_items": len(rows),
        "n_both_ok": len(both_ok),
        "n_degraded_a": sum(1 for r in rows if not r["a"]["ok"]),
        "n_degraded_b": sum(1 for r in rows if not r["b"]["ok"]),
        "n_points": n_points,
        "n_points_agree": agree,
        "point_agreement": (agree / n_points) if n_points else None,
        "cohens_kappa": cohens_kappa(pairs),
        "n_consensus_by_midpoint": by_midpoint,
        "degradation_reasons": {
            "a": dict(sorted(Counter(r["a"]["reason"] for r in rows if not r["a"]["ok"]).items())),
            "b": dict(sorted(Counter(r["b"]["reason"] for r in rows if not r["b"]["ok"]).items())),
        },
    }
    if judge_records is not None:
        stats["judge_vs_consensus"] = judge_vs_consensus(both_ok, judge_records)
    return stats


# --------------------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------------------


def render_report(stats: dict[str, Any], *, split: str, model_a: str, model_b: str,
                  judge_model: str) -> str:
    """渲染 ``annotate.md``。**无时间戳**：同配置两次跑结构一致（分数允许不同）。

    表头用中文而 ASCII 分隔符，与 :mod:`eval_run` / :mod:`calibrate` 的报告一致。
    """
    lines: list[str] = []
    lines.append("# 双标一致性报告（LLM 辅助双标）")
    lines.append("")
    lines.append(f"- split：{split}")
    lines.append(f"- 标注者 A：{model_a}")
    lines.append(f"- 标注者 B：{model_b}")
    lines.append(f"- 被比对的 judge：{judge_model}（G 门槛 {PASS_THRESHOLD_G}）")
    lines.append(f"- 样本：{stats['n_items']} 条（两边都成功的 {stats['n_both_ok']} 条）")
    lines.append("")
    lines.append("> 标注者互相独立：各自一次调用，不看对方、不看 judge 分、不看 agent 身份。")
    lines.append("> 本报告**不含时间戳**：同配置两次跑结构一致，分数允许不同（LLM 有随机性）。")
    lines.append("")

    lines.append("## 一、标注者间一致率（人类一致性代理）")
    lines.append("")
    lines.append(f"- 逐点一致率：**{_pct(stats['point_agreement'])}**"
                 f"（{stats['n_points_agree']}/{stats['n_points']} 点）")
    kappa = stats["cohens_kappa"]
    lines.append(f"- Cohen's κ：{_fmt(kappa)}"
                 + ("　（样本无分歧 / 全同档 ⇒ κ 无定义，不是 0）" if kappa is None else ""))
    lines.append(f"- 共识靠「A/B 分歧取中点」而来的条数：{stats['n_consensus_by_midpoint']}")
    lines.append("")
    lines.append("> κ 会受标签分布影响（两人都爱给 1 时 κ 虚高），所以与原始一致率一起报。")
    lines.append("")

    jc = stats.get("judge_vs_consensus")
    lines.append("## 二、judge vs 共识")
    lines.append("")
    if not jc:
        lines.append("未提供 judge 记录（本轮只跑了双标）。")
    else:
        lines.append(f"- 可比条数：{jc['n']}")
        lines.append(f"- G 一致率（|Δ| < 0.05）：{_pct(jc['agreement'])}")
        lines.append(f"- G 平均绝对误差：{_fmt(jc['mean_abs_error'])}")
        lines.append(f"- **通过率翻转**（G ≥ {PASS_THRESHOLD_G} 与否）：{jc['n_pass_flip']} 条"
                     f" = {_pct(jc['pass_flip_rate'])}")
        lines.append(f"- 大幅分歧（|Δ| > 0.2）：{jc['n_divergence']} 条")
        lines.append("")
        if jc["divergences"]:
            lines.append("| task_id | 类型 | judge_G | 共识_G | Δ |")
            lines.append("|---|---|---|---|---|")
            for d in jc["divergences"]:
                lines.append(
                    f"| {d['task_id']} | {d['task_type']} | {_fmt(d['judge_G'])} "
                    f"| {_fmt(d['consensus_G'])} | {_fmt(d['judge_G'] - d['consensus_G'])} |"
                )
            lines.append("")

    lines.append("## 三、降级计数")
    lines.append("")
    lines.append(f"- 标注者 A 降级：**{stats['n_degraded_a']}** 条")
    lines.append(f"- 标注者 B 降级：**{stats['n_degraded_b']}** 条")
    lines.append("")
    for who, reasons in stats["degradation_reasons"].items():
        if reasons:
            lines.append(f"标注者 {who.upper()} 的降级原因：")
            lines.append("")
            for reason, count in reasons.items():
                lines.append(f"- `{reason}`：{count}")
            lines.append("")
    if not stats["n_degraded_a"] and not stats["n_degraded_b"]:
        lines.append("无降级。")
    else:
        lines.append("**降级条已从一致率里排除**（不按 0 计），否则「没干活」会被记成「判为 0 分」。")
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    rows: Sequence[dict[str, Any]],
    stats: dict[str, Any],
    out_dir: Path,
    *,
    split: str,
    model_a: str,
    model_b: str,
    judge_model: str,
) -> Path:
    """写 ``annotations.jsonl`` + ``annotate.md``，返回报告路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "annotations.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    path = out_dir / "annotate.md"
    path.write_text(
        render_report(stats, split=split, model_a=model_a, model_b=model_b, judge_model=judge_model),
        encoding="utf-8", newline="\n",
    )
    return path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _env(name: str) -> str | None:
    import os

    value = os.environ.get(name)
    return value or None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m zh_agent_bench.annotate",
        description="LLM 辅助双标：两个独立标注者盲标，报一致率 / κ / judge vs 共识",
    )
    parser.add_argument("--split", default="blind", choices=("trial", "dev", "blind"),
                        help="正式集 split（默认 blind）")
    parser.add_argument("--records", type=Path, required=True,
                        help="eval_run --save-answers 产出的 records.jsonl")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="正式集根目录（默认仓库 benchmarks/qa）")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/qa/calibration-report"),
                        help="输出目录（写 annotations.jsonl + annotate.md）")
    parser.add_argument("--limit", type=int, default=0, help="只标前 N 条（0 = 全标）")
    parser.add_argument("--model-a", default=None, help="标注者 A 的模型 id")
    parser.add_argument("--model-b", default=None, help="标注者 B 的模型 id")
    parser.add_argument("--judge-model", default=None,
                        help="被比对的 judge 模型 id（只写进报告，不影响调用）")
    args = parser.parse_args(argv)

    spec_a, spec_b = ANNOTATORS["a"], ANNOTATORS["b"]
    model_a = args.model_a or spec_a["name"]
    model_b = args.model_b or spec_b["name"]
    key_a, key_b = _env(spec_a["api_key_env"]), _env(spec_b["api_key_env"])
    url_a = _env(spec_a["base_url_env"]) if spec_a["base_url_env"] else None
    url_b = _env(spec_b["base_url_env"]) if spec_b["base_url_env"] else None
    missing = [
        name for name, key in ((spec_a["api_key_env"], key_a), (spec_b["api_key_env"], key_b))
        if not key
    ]
    if missing:
        print(f"缺少 API key 环境变量：{', '.join(missing)}（无 key 不算失败，见判据 4）")
        return 3

    try:
        items = load_answer_records(args.split, args.records,
                                    data_root=args.data_root, limit=args.limit)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover —— parser.error 不返回

    rows = run_double_annotation(
        items, call_a=_default_call, call_b=_default_call,
        model_a=model_a, model_b=model_b,
        key_a=key_a, url_a=url_a, key_b=key_b, url_b=url_b,
    )
    # judge 记录从同一个 records.jsonl 取 G —— 那正是正式跑分写下的那个 G。
    judge_records = [
        json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stats = summarize(rows, judge_records=judge_records)

    path = write_outputs(
        rows, stats, args.out, split=args.split,
        model_a=model_a, model_b=model_b,
        judge_model=args.judge_model or model_b,
    )
    print(f"split={args.split} n={stats['n_items']} both_ok={stats['n_both_ok']} "
          f"一致率={_pct(stats['point_agreement'])} κ={_fmt(stats['cohens_kappa'])} "
          f"降级A={stats['n_degraded_a']} 降级B={stats['n_degraded_b']}")
    jc = stats.get("judge_vs_consensus") or {}
    if jc:
        print(f"judge_vs_consensus n={jc.get('n')} 一致率={_pct(jc.get('agreement'))} "
              f"MAE={_fmt(jc.get('mean_abs_error'))} 通过翻转={jc.get('n_pass_flip')}")
    print(f"report -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
