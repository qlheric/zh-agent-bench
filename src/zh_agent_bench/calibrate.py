"""calibrate —— judge 校准最小闭环。

**这张表回答一个问题：中文 judge 到底判得稳不稳。**
不是「判得准不准」（那要 330 条 + 双人盲标 + 配对偏置实验，是后续单），
而是「同一条 rubric，example 判不判对、明显挖掉一个点的变体判不判错」——
**判别力**（discrimination）：如果 example 和错误变体拿到一样的分，judge 就没在判分，
只是在复读。

跑一遍产出三样东西：

1. **混淆表**：judge 判「对/错」× 预期「对/错」的 2×2 + 一致率；
2. **按类型切片**：四个 B 场景类型的分开一致率（整体一致率高但全靠单跳撑着，
   在报告里看不出，必须分开报）；
3. **降级计数**：judge 调用失败 / 分值非法 / 漏评分点，各多少次。**这一栏不能省** ——
   judge 挂了会静默把 G 拉到 0，那看起来就像「agent 变差了」，而不是「judge 坏了」。

**变体怎么造**：不是拿 LLM 再写一份答案（那就变成测 LLM 了），而是按任务类型
用**确定性的程序规则**破坏 example 答案（见 :data:`CORRUPT_RULES`）。
人工定的只有「哪一类破坏能真正打掉那个类型的 rubric 关键点」，执行是纯程序 —— 于是
同一份数据重跑得到逐字节相同的记录表。

**运行时只依赖 stdlib + 本包**，与其他模块一致。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from .judge import PASS_THRESHOLD_G, judge_task
from .tasks.qa import SCENARIOS, TASK_TYPES, build_qa_dataset
from .tasks.schema import Task

#: 答案的两类：数据集自带的合格示例，和人工破坏出来的错误变体。
ANSWER_EXAMPLE = "example"
ANSWER_CORRUPTED = "corrupted"

#: 预期分带。``high`` = G 应达主榜门槛（关键点全满），``low`` = G 应明显低于门槛。
#: 带的名字用 ASCII：报告要同时含中文内容和 ASCII 表头，混排会让 Windows 控制台
#: 代码页把非 ASCII 部分打成 ``?``（findings.md §二.2 的同一条理由）。
BAND_HIGH = "high"
BAND_LOW = "low"

#: 「对错」这个二值标签的预期来源。混淆表的主对角线全靠它。
EXPECTED_CORRECT = {ANSWER_EXAMPLE: True, ANSWER_CORRUPTED: False}

EXPECTED_BAND = {ANSWER_EXAMPLE: BAND_HIGH, ANSWER_CORRUPTED: BAND_LOW}


# --------------------------------------------------------------------------------------
# 错误变体构造（规则是人工定的，执行是纯程序）
# --------------------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d+")


def _corrupt_number(text: str) -> str:
    """把第一个数字加一。

    事实型任务（单跳/多跳）的 rubric 判的就是「数字/日期对不对」，
    改掉一个就必然打掉对应的那一点，并额外制造一条**矛盾 claim**（F 会跟着掉）。
    """
    return _NUMBER_RE.sub(lambda m: str(int(m.group()) + 1), text, count=1)


def _hard_answer(text: str, task_id: str) -> str:
    """把「答不了」的答案整体换成一句**硬答**，声明型任务的破坏变体。

    **为什么不用「删句」**：TASK-010 实测，删句删不准 ——
    ``_keep_first_sentence`` 删掉的是末句建议，三个 critical 点原句全留着，
    judge 给 G=0.85 是**正确行为**（FP 表成了 judge 的冤案）；
    改成按关键词删声明句同样不稳（「没有任何一篇给出…」这种句子本身带否定词，
    会被当成「答不了」删掉，剩下的恰好还是 c1 的原句）。

    **改成硬答就不依赖删句准不准**：声明型任务的 critical 点**全部**是
    「明确指出证据不足 / 信息不足、要求澄清」这一族（见 ``tasks/qa.py`` 的生成期注释
    「正确句抵消」），一句硬答把「答不了」的语义整条抹掉 ⇒ critical 点必然全丢 ⇒
    judge 必须判错。**判错是构造出来的，不依赖任何一句具体措辞。**
    """
    return HARD_ANSWER_POOL[zlib.crc32(task_id.encode("utf-8")) % len(HARD_ANSWER_POOL)]


#: 硬答变体池。**每句都不含任何具体数字/事实** —— 变体要证明的是
#: 「不声明证据不足」这个行为本身被扣分，不是「答错了内容」；
#: 塞进具体断言会把「漏关键点」和「内容说错」两件事混进同一格，诊断不出根因。
#:
#: **池子为什么不止一句**（TASK-012）：TASK-011 用单句拿到 100% 一致率，但那是
#: **4 条任务说同一句话** —— judge 只要认得这一句就够，测不出「它认的是硬答这个行为
#: 还是这句话的字面」。多句同义不同措辞是这条 100% 的**负向验证**：句式换过还判得对，
#: 才说明认的是行为。措辞差异也要刻意拉开（递进推诿 vs 轻描淡写），免得两句其实同源。
HARD_ANSWER_POOL: tuple[str, ...] = (
    "按公司规定执行即可，具体额度以系统审批结果为准。",
    "这种情况很常见，一般直接找对应的负责人协商就能解决。",
)


#: 按 B 场景任务类型分派的破坏规则。**不分派不行**：两类任务的 critical 点设计不同，
#: 事实型考「数字对不对」、声明型考「有没有明说答不了」，一条规则打不掉另一类。
CORRUPT_RULES: dict[str, str] = {
    "单跳": "corrupt_number",
    "多跳": "corrupt_number",
    "证据不足": "hard_answer",
    "不可回答": "hard_answer",
}

#: 破坏函数的统一签名：``(原答案, task_id) -> 变体答案``。
#: ``task_id`` 只被硬答用（轮换用），``corrupt_number`` 不需要它 —— 但让两者签名一致，
#: 比在 :func:`build_answers` 里为硬答单开一个 ``if`` 分支便宜。
_CORRUPTORS: dict[str, Callable[[str, str], str]] = {
    "corrupt_number": lambda text, _task_id: _corrupt_number(text),
    "hard_answer": _hard_answer,
}


def max_per_type() -> int:
    """场景池每类最多几条 —— ``--per-type`` 的上限。超出即**循环复用**，
    而复用会让同一场景被 judge 判两遍、一致率被同一条样本计两次，数字直接失真。"""
    return min(len(pool) for pool in SCENARIOS.values())


#: 破坏规则必须真的打掉 critical 点的**自检**依据 —— 声明型 critical requirement 里
#: 「明说答不了」这一族的锚词。命中即认为该 critical 点**仍可能被判到**。
_DECLARATIVE_ANCHORS = ("无法", "不能", "答不了", "不足", "未提供", "没有", "需要", "请")


def _assert_corruption_hits_critical(task: Task, example: str, corrupted: str) -> None:
    """破坏变体必须打掉至少一个 critical 点，否则这张混淆表测的是规则不是 judge。

    TASK-010 的教训：``_keep_first_sentence`` 删的是末句建议、三个 critical 点原句全留，
    judge 给 G=0.85 判「对」**完全正确**，但混淆表会把这 8 条 FP 记成 judge 的冤案。
    **破坏规则和 judge 一样会出错** —— 规则悄悄没打中 critical 点时，报告和
    ``findings.md`` 里所有的结论都会跟着错，且没有任何信号提示。

    **本自检是保守的**：它只看「声明型 critical 的锚词还在不在」，锚词在就报错。
    锚词不在**不保证** critical 真丢（judge 可能宽松），所以它防的是
    「规则明确没打中」这一类，够用且不会误伤 —— 宁可误报也不漏报。
    """
    crit = [p for p in (task.rubric or ()) if p.get("critical")]
    if not crit:
        return
    anchors_gone = not any(a in corrupted for a in _DECLARATIVE_ANCHORS)
    if anchors_gone:
        return
    # 事实型用数字锚点：改数字必然打掉「数字对不对」那一点。
    if _NUMBER_RE.search(example) and not _NUMBER_RE.search(corrupted):
        return
    raise ValueError(
        f"{task.id}（{task.initial_data['任务类型']}）的破坏规则没打掉任何 critical 点："
        f"corrupted 仍带声明锚词/数字锚点。这样的样本会被 judge 判对，"
        f"混进混淆表就成了 FP 冤案，而不是在测 judge"
    )


def build_answers(task: Task) -> list[dict[str, Any]]:
    """给一条任务造两类答案及其预期标签。

    预期是**构造出来的、不是事后填的**：``example`` 一定预期判对、``corrupted``
    一定预期判错。judge 与预期不一致，正是这张表要抓的东西。
    """
    example = task.ground_truth.get("example_answer")
    if not isinstance(example, str) or not example.strip():
        raise ValueError(f"{task.id} 没有可用的 ground_truth['example_answer']")
    rule = CORRUPT_RULES[task.initial_data["任务类型"]]
    corrupted = _CORRUPTORS[rule](example, task.id)
    _assert_corruption_hits_critical(task, example, corrupted)
    return [
        {
            "answer_kind": ANSWER_EXAMPLE,
            "answer": example,
            "expected_correct": EXPECTED_CORRECT[ANSWER_EXAMPLE],
            "expected_band": EXPECTED_BAND[ANSWER_EXAMPLE],
            "corruption": None,
        },
        {
            "answer_kind": ANSWER_CORRUPTED,
            "answer": corrupted,
            "expected_correct": EXPECTED_CORRECT[ANSWER_CORRUPTED],
            "expected_band": EXPECTED_BAND[ANSWER_CORRUPTED],
            "corruption": rule,
        },
    ]


# --------------------------------------------------------------------------------------
# 跑一批
# --------------------------------------------------------------------------------------


#: 降级原因 → judge 在 ``reason`` 里留下的稳定文案。**从 reason 认，不另开一条探测通道**：
#: 这些文案就写在 judge 的降级分支里（``_collect_scores`` / ``_ask_judge``），
#: 另写一套探测等于把「judge 什么时候算坏」的定义抄第二份，两份迟早不一致。
DEGRADATION_MARKERS: dict[str, tuple[str, ...]] = {
    "llm_failure": ("judge 调用失败", "judge 返回"),
    "invalid_score": ("不在 0/0.5/1 内", "不是数字"),
    "missing_criterion": ("judge 未给出分值", "全部评分点按 0 记"),
}


def classify_degradation(reason: str, scores: Any, expected_ids: Sequence[str]) -> str | None:
    """判断这一条算不算**降级**，返回降级类型；正常返回 ``None``。

    三种降级各有用处，少一种就有一批坏样本静默混进一致率里。
    """
    # **先按 reason 文案认**：judge 的 notes 是按发生顺序追加的，一次调用同时挂了
    # LLM（调用失败）和漏评分点（scores 整块缺失）时，调用失败才是**根因** ——
    # 报成 missing_criterion 会让人去查 rubric，而真正坏的是网络/key。
    for kind, markers in DEGRADATION_MARKERS.items():
        if any(marker in reason for marker in markers):
            return kind
    # scores 键与 rubric 对不上也算降级：judge 漏判不得被当成「这点没提到」静默放过。
    if not isinstance(scores, dict) or set(scores) != set(expected_ids):
        return "missing_criterion"
    return None


def judge_one(task: Task, item: dict[str, Any], judge_fn: Callable[..., dict]) -> dict[str, Any]:
    """判一条答案，落成一条记录。

    轨迹不传：校准的对象是 judge 的**纯判分能力**（同一份答案给不给轨迹都该判同分），
    而 T 指标在 B 场景没有轨迹时是「不适用」，不参与 ``correct``，不影响混淆表。
    """
    expected_ids = [p["criterion_id"] for p in (task.rubric or ())]
    verdict = judge_fn(task, item["answer"])
    g = verdict.get("G")
    scores = verdict.get("scores")
    if not isinstance(g, (int, float)) or isinstance(g, bool):
        scores = None  # 没有合法 G ⇒ 按非法分值算，别让 None 混进统计
    reason = verdict.get("reason") or ""
    record = {
        "task_id": task.id,
        "task_type": task.initial_data["任务类型"],
        "answer_kind": item["answer_kind"],
        "corruption": item["corruption"],
        "judge_G": g,
        "judge_F": verdict.get("F"),
        "judge_correct": bool(verdict.get("correct")),
        "n_contradicted": verdict.get("n_contradicted"),
        "n_unsupported": verdict.get("n_unsupported"),
        "expected_correct": item["expected_correct"],
        "expected_band": item["expected_band"],
        "judge_model": verdict.get("model"),
        "degradation": classify_degradation(reason, scores, expected_ids),
        "reason": reason,
    }
    return record


def run_calibration(
    tasks: Iterable[Task],
    *,
    judge_fn: Callable[..., dict] | None = None,
) -> list[dict[str, Any]]:
    """跑完整批校准，返回记录列表（顺序 = 任务序 × 答案类序，天然确定）。"""
    call = judge_fn or judge_task
    return [
        judge_one(task, item, call)
        for task in tasks
        for item in build_answers(task)
    ]


# --------------------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------------------


def confusion(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """judge 判「对/错」× 预期「对/错」的 2×2 表 + 一致率。

    正类 = 「判对」。所以 ``TP`` 是好答案被放过，``FN`` 是好答案被误杀，
    ``FP`` 是坏答案被放过（**最危险的一格**），``TN`` 是坏答案被抓住。
    一致率 = ``(TP+TN)/N``，与 :mod:`pytest` 无关，纯算术。
    """
    tp = tn = fp = fn = 0
    for record in records:
        expected = record["expected_correct"]
        got = record["judge_correct"]
        if expected and got:
            tp += 1
        elif expected and not got:
            fn += 1
        elif not expected and got:
            fp += 1
        else:
            tn += 1
    total = tp + tn + fp + fn
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total,
        "agreement": (tp + tn) / total if total else 0.0,
    }


def per_type_agreement(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 B 场景四类分开统计。

    整体一致率有个陷阱：某一类全对就能把整体拉得很高，而另一类**全错**在
    总数上看着也不算刺眼。分开报才看得出是哪一类不稳。
    """
    rows = list(records)
    out: dict[str, dict[str, Any]] = {}
    for task_type in TASK_TYPES:
        subset = [r for r in rows if r["task_type"] == task_type]
        if subset:
            out[task_type] = confusion(subset)
    return out


def degradation_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    """降级次数按类型分。0 是唯一健康值，非 0 必须在报告里显眼。"""
    counts = {kind: 0 for kind in DEGRADATION_MARKERS}
    for record in records:
        kind = record.get("degradation")
        if kind:
            counts[kind] += 1
    return counts


def discrimination(records: Iterable[dict[str, Any]]) -> dict[str, float | None]:
    """判别力：example 平均分 vs 错误变体平均分，差值越大越会判。

    混淆表是「判对/判错」这个二值口径，G 分带则能看出区分是**大**还是**刚好压线** ——
    全部变体都只差 0.01 压线过关，说明 judge 判得勉强。
    """
    rows = list(records)
    out: dict[str, float | None] = {}
    for kind in (ANSWER_EXAMPLE, ANSWER_CORRUPTED):
        gs = [r["judge_G"] for r in rows
              if r["answer_kind"] == kind and isinstance(r["judge_G"], (int, float))]
        out[kind] = (sum(gs) / len(gs)) if gs else None
    gap = out[ANSWER_EXAMPLE] - out[ANSWER_CORRUPTED] if None not in out.values() else None
    out["gap"] = gap
    return out


# --------------------------------------------------------------------------------------
# 产物
# --------------------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_report(records: Sequence[dict[str, Any]], *, per_type: int, seed: int) -> str:
    """渲染 ``report.md``。**无时间戳、无随机序**：同输入两次逐字节一致（findings §二）。"""
    table = confusion(records)
    lines: list[str] = []
    lines.append("# judge 校准报告（最小闭环）")
    lines.append("")
    lines.append(f"- 任务数：{len(records) // 2}（--per-type {per_type} --seed {seed}）")
    lines.append(f"- 答案数：{table['total']}（每条任务 2 个：example + 错误变体）")
    lines.append(f"- 主榜门槛 G：{PASS_THRESHOLD_G}")
    lines.append("")
    lines.append("## 混淆表（正类 = 判对）")
    lines.append("")
    lines.append("|  | judge 判对 | judge 判错 |")
    lines.append("|---|---|---|")
    lines.append(f"| **预期对** | TP {table['tp']} | FN {table['fn']} |")
    lines.append(f"| **预期错** | FP {table['fp']} | TN {table['tn']} |")
    lines.append("")
    lines.append(f"一致率：{_pct(table['agreement'])}（({table['tp']}+{table['tn']})/{table['total']}）")
    lines.append("")
    lines.append("> FP = 坏答案被判对（最危险）；FN = 好答案被判错（会误伤 agent）。")
    lines.append("")
    lines.append("## 按类型切片")
    lines.append("")
    lines.append("| type | n | TP | TN | FP | FN | agreement |")
    lines.append("|---|---|---|---|---|---|---|")
    for task_type, stats in per_type_agreement(records).items():
        lines.append(
            f"| {task_type} | {stats['total']} | {stats['tp']} | {stats['tn']} "
            f"| {stats['fp']} | {stats['fn']} | {_pct(stats['agreement'])} |"
        )
    lines.append("")
    lines.append("## 判别力（G 分带）")
    lines.append("")
    disc = discrimination(records)
    for kind in (ANSWER_EXAMPLE, ANSWER_CORRUPTED):
        mean = disc[kind]
        lines.append(f"- {kind}：{'n/a' if mean is None else f'{mean:.4f}'}")
    gap = disc["gap"]
    lines.append(f"- gap（example - corrupted）：{'n/a' if gap is None else f'{gap:+.4f}'}")
    lines.append("")
    lines.append("## 降级计数")
    lines.append("")
    counts = degradation_counts(records)
    lines.append("| kind | count |")
    lines.append("|---|---|")
    for kind, count in counts.items():
        lines.append(f"| {kind} | {count} |")
    total_bad = sum(counts.values())
    lines.append("")
    lines.append(f"降级合计：{total_bad} / {table['total']}。"
                 + ("（无降级）" if total_bad == 0 else " **有降级，上面的分数不可直接引用**"))
    lines.append("")
    lines.append("## 不一致样本")
    lines.append("")
    wrong = [r for r in records if r["judge_correct"] != r["expected_correct"]]
    if not wrong:
        lines.append("无。")
    else:
        # 判错的 FN 分两类，根因完全不同，混在一张表里看不出该修哪：
        # G 不够（judge 没认出挖掉/改错的那个点）和 G 够但 F 把它毙了
        # （答案里那句「声明库里没有」被 judge 自己抽成了待证 claim，再判它证据不足）。
        f_killer = [
            r for r in wrong
            if r["expected_correct"] and not r["judge_correct"]
            and isinstance(r["judge_G"], (int, float)) and r["judge_G"] >= PASS_THRESHOLD_G
        ]
        lines.append(
            f"其中 **{len(f_killer)} 条是 G 已达门槛却被判错** —— 问题在 F 的 claim 口径，不在 rubric 判分。"
            if f_killer else "无「G 达标却判错」的样本。"
        )
        lines.append("")
        lines.append("| task_id | type | kind | judge_G | F | n_unsup | judge | expected | reason |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in wrong:
            f_score = r["judge_F"]
            lines.append(
                f"| {r['task_id']} | {r['task_type']} | {r['answer_kind']} | {r['judge_G']} "
                f"| {'n/a' if f_score is None else f'{f_score:.3f}'} | {r['n_unsupported']} "
                f"| {'对' if r['judge_correct'] else '错'} | {'对' if r['expected_correct'] else '错'} "
                f"| {r['reason'][:80]} |"
            )
    return "\n".join(lines) + "\n"


def write_outputs(records: Sequence[dict[str, Any]], out_dir: Path, *, per_type: int, seed: int) -> Path:
    """写 ``records.jsonl`` + ``report.md``，返回 report 路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    report_path = out_dir / "report.md"
    report_path.write_text(render_report(records, per_type=per_type, seed=seed), encoding="utf-8", newline="\n")
    return report_path


# --------------------------------------------------------------------------------------
# 数据装载
# --------------------------------------------------------------------------------------


def _select_per_type(tasks: Sequence[Task], per_type: int) -> list[Task]:
    """按类型取前 ``per_type`` 条。**类型顺序固定**（:data:`TASK_TYPES`），不靠字典序。"""
    selected: list[Task] = []
    for task_type in TASK_TYPES:
        rows = [t for t in tasks if t.initial_data.get("任务类型") == task_type]
        if len(rows) < per_type:
            raise ValueError(
                f"类型 {task_type} 只有 {len(rows)} 条，取不满 --per-type {per_type}；"
                f"场景池每类最多 {max_per_type()} 条，不许循环复用"
            )
        selected.extend(rows[:per_type])
    return selected


#: 正式集三 split 的档名（与 ``benchmarks/qa/<split>/qa.jsonl`` 的目录同名）。
FORMAL_SPLITS = ("trial", "dev", "blind")

#: 正式集默认根目录。**从本文件位置反推仓库根**，不从 CWD 推 ——
#: 跑分器可能被从任意目录调起（IDE / pytest / 另一个 shell），CWD 依赖会让
#: 「找不到数据集」变成一件与代码正确性无关的玄学故障。
_DEFAULT_QA_ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "qa"


def load_formal_split(split: str, data_root: Path | None = None) -> list[Task]:
    """读正式集某一 split 的 ``qa.jsonl`` → ``list[Task]``。

    :param split: :data:`FORMAL_SPLITS` 之一（大小写敏感，与目录名一字对应）。
    :param data_root: 数据根目录，默认 :data:`_DEFAULT_QA_ROOT`；给定时读
        ``<data_root>/<split>/qa.jsonl``。

    **只负责读，不碰条数**。60/120/150 的条数由测试钉（``tests/test_eval_run.py``），
    加载器硬编码「trial 必须是 60 条」只会在有人扩数据集时把加载器变成绊脚石 ——
    而那正是本函数存在的理由（把三份产物读进来）。
    """
    if split not in FORMAL_SPLITS:
        raise ValueError(f"split 必须是 {FORMAL_SPLITS} 之一，收到 {split!r}")
    root = Path(data_root) if data_root is not None else _DEFAULT_QA_ROOT
    path = root / split / "qa.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到正式集 {path}。若产物未入库，先跑："
            f"python -m zh_agent_bench.tasks.qa --seed 42 --preset formal --output-dir {root}"
        )
    return [
        Task.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_tasks(per_type: int, seed: int, dataset: Path | None = None) -> list[Task]:
    """取校准任务。

    :param dataset: 给定就读现成 ``qa.jsonl``；不给就按 ``(per_type, seed)`` 从场景池
        现生成。两者产物同构 —— 生成路径是读路径的超集（``build_qa_dataset`` 按
        下标独立开随机源，``per_type=2`` 生成的 8 条与入库文件逐条相同）。
    """
    if per_type < 1:
        raise ValueError(f"--per-type 必须 >= 1，收到 {per_type}")
    if per_type > max_per_type():
        raise ValueError(
            f"--per-type {per_type} 超过场景池每类容量 {max_per_type()}，"
            f"再大就得循环复用同一批场景，一致率会被同一条样本计两次"
        )
    if dataset is not None:
        tasks = [
            Task.from_dict(json.loads(line))
            for line in dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return _select_per_type(tasks, per_type)
    return _select_per_type(build_qa_dataset(per_type=per_type, seed=seed), per_type)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m zh_agent_bench.calibrate",
        description="跑 judge 校准最小闭环：混淆表 + 一致率 + 按类型切片 + 降级计数",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（决定挑哪些场景）")
    parser.add_argument("--per-type", type=int, default=4, help="每个任务类型取几条（<= 场景池容量）")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/calibration"), help="输出目录")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="读现成 qa.jsonl（不给则按 seed/per_type 从场景池生成）")
    args = parser.parse_args(argv)

    try:
        tasks = load_tasks(args.per_type, args.seed, args.dataset)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover —— parser.error 不返回

    records = run_calibration(tasks)
    report = write_outputs(records, args.out, per_type=args.per_type, seed=args.seed)

    table = confusion(records)
    counts = degradation_counts(records)
    lines = [
        f"tasks={len(tasks)} answers={table['total']} per_type={args.per_type} seed={args.seed}",
        f"{'type':<14}{'n':>4}{'TP':>5}{'TN':>5}{'FP':>5}{'FN':>5}{'agreement':>12}",
    ]
    for task_type, stats in per_type_agreement(records).items():
        lines.append(
            f"{task_type:<14}{stats['total']:>4}{stats['tp']:>5}{stats['tn']:>5}"
            f"{stats['fp']:>5}{stats['fn']:>5}{_pct(stats['agreement']):>12}"
        )
    lines.append(f"agreement = {_pct(table['agreement'])}")
    disc = discrimination(records)
    gap = disc["gap"]
    lines.append(f"G(example) = {disc[ANSWER_EXAMPLE]}, G(corrupted) = {disc[ANSWER_CORRUPTED]}, "
                 f"gap = {'n/a' if gap is None else f'{gap:+.4f}'}")
    lines.append("degradation: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print("\n".join(lines))
    print(f"report -> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
