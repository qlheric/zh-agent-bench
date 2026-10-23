"""leaderboard —— A/B/C 三场景统一榜单（TASK-024）。

**这个模块只做聚合，不跑 agent。** 三场景各有自己的跑分器（``python -m zh_agent_bench`` /
``eval_run --split trial|dev|blind`` / ``eval_run --split orchestration``），跑分是那些命令的
事；本模块只读它们**已经落盘的** ``records.jsonl``，把三份结果收敛成同一张表 ——
「一个 agent 在三个场景各几分」是 S3 对外交付时要能一句话说清的事，
而三套报告各渲染各的，谁也拼不出一张全的。

三件事是这个模块存在的原因：

1. **口径不许漂**：A 的 ``exact`` 逐字相等、B 的 ``G ≥ 0.8``、C 的 ``delta_ok ∧ reply_ok``
   写死在表头里，不藏在代码深处。三条门槛不同的场景挤进一张表，门槛不写出来那张表就是错的。
2. **缺场景写「未跑」**：没跑的场景在表里是 ``未跑`` 而不是 ``0`` ——
   ``0`` 会被读成「跑了得零分」，那是两条完全不同的结论。
   旧布局（路径里没有 agent 目录）的产物同理写「未发现产物」，不静默当成 0。
3. **降级/失败单列**：judge 降级、agent 崩了都不是「agent 答错」，
   混进正确率会让榜单替 judge 的故障背锅（口径与 ``eval_run.summarize`` 一致）。

**一 agent 一行（TASK-028）**：产物布局是 ``benchmarks/<场景>/<档>/<agent>/records.jsonl``，
本模块扫目录出多行。多 agent 横评是基准的核心价值，一个基准只能评一个 agent 时这张表没有意义。

**运行时仅 stdlib**，与全项目一致。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from .eval_run import SCENARIO_ROOTS, check_blind_allowed, safe_agent_dirname
from .judge import PASS_THRESHOLD_C, PASS_THRESHOLD_G

#: 三个场景的固定顺序。榜单列序是契约的一部分（两次跑逐行可比），不按字典序排。
SCENARIOS = ("A", "B", "C")

#: A 场景（客观题）读哪一档。榜单只认 heldout —— 与 ``cli.py`` 的 ``--heldout-only`` 同一口径。
A_SPLIT = "heldout"


# --------------------------------------------------------------------------------------
# 扫产物
# --------------------------------------------------------------------------------------


class ScanResult(NamedTuple):
    """扫出来的东西。**分开存新旧两类产物，不混** —— 旧布局没有 agent 目录，
    猜不出是谁跑的，混进表里就是编数据。"""

    #: agent 目录名 → {场景: records.jsonl 路径或 None}。按 agent 目录名排好序。
    agents: dict[str, dict[str, Path | None]]
    #: 场景 → 旧布局产物路径列表（已排序）。**按场景分桶**：表格里「未发现产物」是
    #: 某一场景的状态（那个场景跑过、但产物是老格式），不分桶就没法渲染成单元格。
    legacy: dict[str, list[Path]]


def _read_records(path: Path) -> list[dict[str, Any]]:
    """读一份 ``records.jsonl``。空行跳过，坏行报错不静默丢 ——
    榜单是对外材料，静默丢记录等于报了一个偏乐观的分数。"""
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no} 解析失败：{exc}") from exc
    return records


def scan_products(
    root: Path,
    *,
    wanted: Sequence[str] = SCENARIOS,
    b_split: str = "dev",
    c_preset: str = "trial",
) -> ScanResult:
    """扫 ``benchmarks/<场景>/<档>/*/records.jsonl`` → 每个 agent 目录一条来源映射。

    路径公式复用 :data:`eval_run.SCENARIO_ROOTS`（跑分写哪儿，榜单就扫哪儿，
    两边各推一遍必然漂）。**扫的是子目录而不是写死一个路径** —— 这就是「一 agent 一行」的来源。

    每个场景**各读各那一档**（A 取 :data:`A_SPLIT`，B 取 ``b_split``，C 取 ``c_preset``）：
    把 trial 与 blind 的产物混进同一张表，那张表答不出「这是哪一档的分」。

    档目录下直接躺 ``records.jsonl``（没有 agent 目录）= **旧布局**：进 ``legacy``，
    **不读、不算分**。它没 agent 署名，读它等于编一个 agent 名。
    """
    grades = {"A": A_SPLIT, "B": b_split, "C": c_preset}
    agents: dict[str, dict[str, Path | None]] = {}
    legacy: dict[str, list[Path]] = {scenario: [] for scenario in wanted}

    for scenario in wanted:
        grade_dir = root / SCENARIO_ROOTS[scenario] / grades[scenario]
        if not grade_dir.is_dir():
            continue
        for child in sorted(grade_dir.iterdir()):
            if child.is_dir() and (child / "records.jsonl").is_file():
                agents.setdefault(child.name, {})[scenario] = child / "records.jsonl"
            elif child.is_file() and child.name == "records.jsonl":
                legacy[scenario].append(child)

    # 每个 agent 的场景键补齐成全集：缺的那格必须是 None（渲染成「未跑」），
    # 而不是「字典里没这个键」—— 后者让 build_entry 与渲染层都要记得兜底。
    for entry in agents.values():
        for scenario in wanted:
            entry.setdefault(scenario, None)

    return ScanResult(
        agents=dict(sorted(agents.items())),
        legacy={k: sorted(v) for k, v in legacy.items()},
    )


# --------------------------------------------------------------------------------------
# 聚合
# --------------------------------------------------------------------------------------


def _rate(correct: int, total: int) -> float | None:
    """正确率。**空样本返回 ``None`` 而不是 0** —— 0 分与「没有样本」不是一回事。"""
    return (correct / total) if total else None


def _mean(values: Sequence[float | None]) -> float | None:
    """只对非 None 取均值（``None`` = 该指标不适用，如空答案的 ``F``）。"""
    picked = [v for v in values if v is not None]
    return (sum(picked) / len(picked)) if picked else None


def aggregate_a(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """A 场景聚合。门槛 = ``exact`` 逐字相等，产物里没有 LLM 维度（G/F/T 列不适用）。

    支持两种产物格式：``harness`` 轨迹式用 ``correct``，``calibrate`` 校准式用 ``judge_correct``。
    """
    if records and "judge_correct" in records[0]:
        correct_key = "judge_correct"
        degraded = [bool(r.get("degradation")) for r in records]
    else:
        correct_key = "correct"
        degraded = [False] * len(records)
    n = len(records)
    n_correct = sum(1 for r in records if r.get(correct_key))
    return {
        "n": n,
        "n_correct": n_correct,
        "rate": _rate(n_correct, n),
        "G": None,
        "F": None,
        "T": None,
        "n_delta_ok": None,
        "n_reply_ok": None,
        "n_degraded": sum(1 for flag in degraded if flag),
        "degradation_kinds": {},
        "n_agent_failed": 0,
    }


def aggregate_b(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """B 场景聚合。门槛 = ``G ≥ {PASS_THRESHOLD_G}``（产物里的 ``correct`` 就是按它判的）。"""
    n = len(records)
    kinds: dict[str, int] = {}
    for row in records:
        kind = row.get("judge_degradation")
        if kind:
            kinds[kind] = kinds.get(kind, 0) + 1
    n_correct = sum(1 for r in records if r.get("correct"))
    return {
        "n": n,
        "n_correct": n_correct,
        "rate": _rate(n_correct, n),
        "G": _mean([r.get("G") for r in records]),
        "F": _mean([r.get("F") for r in records]),
        "T": _mean([r.get("T") for r in records]),
        "n_delta_ok": None,
        "n_reply_ok": None,
        "n_degraded": sum(kinds.values()),
        "degradation_kinds": dict(sorted(kinds.items())),
        "n_agent_failed": sum(1 for r in records if r.get("agent_failed")),
    }


def aggregate_c(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """C 场景聚合。门槛 = ``delta_ok ∧ reply_ok``（其中 ``reply_ok`` 要求回复点 ``G ≥ {PASS_THRESHOLD_C}``）。

    主榜正确率按 ``delta_ok ∧ reply_ok`` **现算**，不直接读 ``correct`` ——
    口径写在榜单这一层，产物里那个 ``correct`` 是上一任写下的，读它等于把口径藏起来。
    """
    n = len(records)
    n_delta = sum(1 for r in records if r.get("delta_ok"))
    n_reply = sum(1 for r in records if r.get("reply_ok"))
    n_correct = sum(
        1 for r in records if r.get("delta_ok") and r.get("reply_ok")
    )
    return {
        "n": n,
        "n_correct": n_correct,
        "rate": _rate(n_correct, n),
        "G": _mean([r.get("G") for r in records]),
        "F": None,
        "T": None,
        "n_delta_ok": n_delta,
        "n_reply_ok": n_reply,
        "n_degraded": 0,
        "degradation_kinds": {},
        "n_agent_failed": sum(1 for r in records if r.get("agent_failed")),
    }


_AGGREGATORS = {"A": aggregate_a, "B": aggregate_b, "C": aggregate_c}


def build_entry(
    agent_name: str,
    sources: dict[str, Path | None],
    wanted: Sequence[str] = SCENARIOS,
) -> dict[str, Any]:
    """把三场景产物收敛成一行榜单数据。缺场景记 ``None``（渲染成「未跑」）。"""
    entry: dict[str, Any] = {"agent": agent_name}
    for scenario in wanted:
        path = sources.get(scenario)
        if path is None or not path.is_file():
            entry[scenario] = None
            continue
        entry[scenario] = _AGGREGATORS[scenario](_read_records(path))
        entry[scenario]["source"] = path.as_posix()
    return entry


# --------------------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------------------

#: 表头。三条门槛直接写进列名 —— 三场景挤一张表，门槛不写出来这张表就是错的。
COLUMNS = (
    "agent",
    "A 场景正确率 (exact 逐字相等)",
    f"B 场景正确率 (G≥{PASS_THRESHOLD_G})",
    "B G",
    "B F",
    "B T",
    f"C 场景正确率 (delta_ok∧reply_ok, 回复点 G≥{PASS_THRESHOLD_C})",
    "C delta_ok",
    "C reply_ok",
    "样本数",
    "降级/失败",
)

_MISSING = "未跑"

#: 旧布局（路径里没有 agent 目录）的产物。**不是「未跑」** —— 那句是说「没跑」，
#: 这句是说「跑过，但产物是老格式，猜不出是谁跑的，所以不进榜」。两回事，不能混。
_LEGACY = "未发现产物"

#: 旧布局产物在表里的占位行名。**不是 agent 名** —— 它不该占第一列冒充一个参赛者。
_LEGACY_ROW = "（旧布局·未纳入榜单）"

#: 一个 agent 目录都没扫到时的占位行名。
_NO_AGENT_ROW = "（未发现任何 agent 产物）"


def _cell(value: Any, *, digits: int = 4) -> str:
    """一个数字格。``None`` 一律渲染成 ``n/a`` 而不是 ``0``。"""
    return "n/a" if value is None else f"{value:.{digits}f}"


def _accuracy(stat: dict[str, Any] | None) -> str:
    if stat is None:
        return _MISSING
    rate = stat["rate"]
    if rate is None:
        return f"{stat['n_correct']}/{stat['n']} = {_MISSING}"
    return f"{stat['n_correct']}/{stat['n']} = {rate * 100:.2f}%"


def _count(stat: dict[str, Any] | None, key: str) -> str:
    if stat is None or stat[key] is None:
        return "n/a"
    return f"{stat[key]}/{stat['n']}"


def _degraded(entry: dict[str, Any], wanted: Sequence[str]) -> str:
    """降级/失败单列。只统计**跑了**的场景 —— 没跑的场景贡献 0 而不是把这一列撑大。"""
    parts: list[str] = []
    total = 0
    for scenario in wanted:
        stat = entry.get(scenario)
        if stat is None:
            continue
        n = stat["n_degraded"] + stat["n_agent_failed"]
        total += n
        if n:
            detail = stat["degradation_kinds"]
            detail = ("（" + "，".join(f"{k}×{v}" for k, v in detail.items()) + "）") if detail else ""
            parts.append(f"{scenario} {n}{detail}")
    return f"{total}｜" + "，".join(parts) if parts else f"{total}"


def _missing_cell(stat: dict[str, Any] | None, legacy_here: Sequence[Path]) -> str:
    """空场景格的两态。**只有这两态** —— 「没跑」与「跑了但老格式」不能混成同一个词。"""
    return _LEGACY if legacy_here else _MISSING


def _row_cells(
    entry: dict[str, Any],
    wanted: Sequence[str],
    legacy: dict[str, list[Path]] | None = None,
) -> list[str]:
    """一个 agent 的一行单元格。

    空场景格是三态：有分出分、无分但在 ``legacy`` 里出「未发现产物」、无分且无 legacy
    出「未跑」。**绝不出 0**。
    """
    legacy = legacy or {}
    a, b, c = (entry.get(s) for s in ("A", "B", "C"))
    samples = sum(
        (entry.get(s) or {}).get("n", 0) for s in wanted if entry.get(s)
    )
    return [
        entry["agent"],
        _accuracy(a) if a else _missing_cell(a, legacy.get("A", ())),
        _accuracy(b) if b else _missing_cell(b, legacy.get("B", ())),
        _cell((b or {}).get("G")),
        _cell((b or {}).get("F")),
        _cell((b or {}).get("T")),
        _accuracy(c) if c else _missing_cell(c, legacy.get("C", ())),
        _count(c, "n_delta_ok"),
        _count(c, "n_reply_ok"),
        str(samples) if samples else "0",
        _degraded(entry, wanted),
    ]


def render_markdown(
    entries: Sequence[dict[str, Any]],
    wanted: Sequence[str] = SCENARIOS,
    legacy: dict[str, list[Path]] | None = None,
) -> str:
    """渲染榜单正文。**一 agent 一行，无时间戳**：同输入两次输出逐字节一致。

    行序 = ``entries`` 的传入序（:func:`scan_products` 已按 agent 目录名排好），
    渲染层不再排一次 —— 两层各排一次，早晚有一层把顺序排反。

    **旧布局另起一行**（agent 名写「（旧布局）」）：产物确实存在，只是不该被
    静默丢掉，也不该混进排名。
    """
    legacy = legacy or {}
    rows: list[tuple[dict[str, Any], bool]] = [(e, False) for e in entries]
    if any(legacy.get(s) for s in wanted):
        rows.append(({"agent": _LEGACY_ROW, **{s: None for s in SCENARIOS}}, True))
    if not rows:
        # 一个 agent 目录都没扫到：仍然出一张全「未跑」的表，而不是一张空表。
        # 空表读起来像「加载失败」，全「未跑」才读得出「还没跑过」。
        rows.append(({"agent": _NO_AGENT_ROW, **{s: None for s in SCENARIOS}}, False))
    lines = [
        "# zh-agent-bench 统一榜单（A/B/C 三场景）",
        "",
        f"> 口径：A 场景按 `exact` 逐字相等；B 场景按主榜门槛 `G ≥ {PASS_THRESHOLD_G}`；"
        f"C 场景按 `delta_ok ∧ reply_ok`（回复点 `G ≥ {PASS_THRESHOLD_C}`）。",
        f"> 「{_MISSING}」= 该场景没有跑分产物；**不等于 0 分**。"
        f"「{_LEGACY}」= 有产物但是旧布局（路径里没有 agent 目录），"
        "猜不出是谁跑的，不进榜。judge 降级与 agent 失败单列，不计入正确率。",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    for entry, _ in rows:
        lines.append("| " + " | ".join(_row_cells(entry, wanted, legacy)) + " |")
    lines.extend(["", "## 样本来源", "", "| agent | 场景 | 产物 | 样本数 |", "|---|---|---|---|"])

    for entry, is_legacy in rows:
        for scenario in wanted:
            stat = entry.get(scenario)
            if stat is not None:
                lines.append(
                    f"| {entry['agent']} | {scenario} | `{stat['source']}` | {stat['n']} |"
                )
            else:
                lines.append(
                    f"| {entry['agent']} | {scenario} | "
                    f"{_LEGACY if is_legacy else _MISSING} | — |"
                )

    lines.extend(["", "## 旧布局产物（未纳入榜单）", ""])
    listed = [(s, p) for s in wanted for p in legacy.get(s, ())]
    if listed:
        lines.append(
            "以下产物**没有 agent 目录**，无法自证是谁跑的，故不参与排名。"
            "迁移到 `benchmarks/<场景>/<档>/<agent>/records.jsonl` 后自动上榜："
        )
        lines.append("")
        lines.extend(f"- `{scenario}`：`{path.as_posix()}`" for scenario, path in listed)
    else:
        lines.append("无。")
    lines.append("")
    return "\n".join(lines)


def to_json(
    entries: Sequence[dict[str, Any]],
    legacy: dict[str, list[Path]] | None = None,
) -> str:
    """榜单的机器可读形态。``sort_keys`` 保证同输入逐字节一致。

    顶层是 ``{"agents": [...], "legacy": {...}}`` 而不是单个 entry ——
    一 agent 一行之后「一份榜单 = 一个 dict」已经不成立了。
    """
    legacy = legacy or {}
    payload = {
        "agents": list(entries),
        "legacy": {k: [p.as_posix() for p in v] for k, v in legacy.items() if v},
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def write_outputs(
    entries: Sequence[dict[str, Any]],
    out_dir: Path,
    wanted: Sequence[str] = SCENARIOS,
    legacy: dict[str, list[Path]] | None = None,
) -> tuple[Path, Path]:
    """写 ``leaderboard.md`` + ``leaderboard.json``，返回两个路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "leaderboard.md"
    md_path.write_text(
        render_markdown(entries, wanted, legacy), encoding="utf-8", newline="\n"
    )
    json_path = out_dir / "leaderboard.json"
    json_path.write_text(
        to_json(entries, legacy), encoding="utf-8", newline="\n"
    )
    return md_path, json_path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

DEFAULT_ROOT = Path(".")
DEFAULT_OUT = Path("benchmarks/leaderboard")
DEFAULT_B_SPLIT = "dev"
DEFAULT_C_PRESET = "trial"
C_PRESETS = ("trial", "dev", "blind")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m zh_agent_bench.leaderboard",
        description="A/B/C 三场景统一榜单：扫各场景已有 records.jsonl，一 agent 一行（不跑 agent）",
    )
    parser.add_argument("--scenario", default="all",
                        choices=("all", "A", "B", "C"),
                        help="要上榜的场景（默认 all：三场景一张表）")
    parser.add_argument("--agent", default=None,
                        help="只保留这一个 agent 目录（默认全出）。**过滤的是目录名**，"
                             "agent 名来自产物路径而不是命令行 —— 榜单自己扫得出来是谁跑的")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="仓库根目录")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="榜单产物目录")
    parser.add_argument("--b-split", default=DEFAULT_B_SPLIT,
                        help="B 场景读哪一档的产物（benchmarks/qa/<档>/<agent>/）")
    parser.add_argument("--c-preset", default=DEFAULT_C_PRESET, choices=C_PRESETS,
                        help="C 场景读哪一档的产物（benchmarks/orchestration/<preset>/<agent>/，与 eval_run --preset 同名同目录）")
    parser.add_argument("--allow-blind", action="store_true",
                        help="放行读 blind 档产物（--b-split blind / --c-preset blind）。"
                             "blind 是封存集，缺省拒绝 —— 开发流程能随手读到它，"
                             "就等于允许对着盲测调参")
    args = parser.parse_args(argv)

    try:
        check_blind_allowed(args.b_split, allow_blind=args.allow_blind)
        check_blind_allowed(args.c_preset, allow_blind=args.allow_blind)
    except PermissionError as exc:
        print(f"封存闸拒绝：{exc}")
        return 2

    wanted = SCENARIOS if args.scenario == "all" else (args.scenario,)
    try:
        scanned = scan_products(
            args.root, wanted=wanted, b_split=args.b_split, c_preset=args.c_preset
        )
        agents = dict(scanned.agents)
        if args.agent is not None:
            agents = {
                name: sources for name, sources in agents.items()
                if name == safe_agent_dirname(args.agent)
            }
            if not agents:
                print(f"没找到 agent 目录 {args.agent!r}（扫的是 {args.root} 下各场景/<档>/<agent>/）")
                return 2
        entries = [build_entry(name, sources, wanted) for name, sources in agents.items()]
        md_path, json_path = write_outputs(entries, args.out_dir, wanted, scanned.legacy)
    except (OSError, ValueError) as exc:
        print(f"榜单生成失败：{exc}")
        return 2

    for entry in entries:
        ran = [s for s in wanted if entry.get(s)]
        print(f"agent={entry['agent']} ran={len(ran)}/{len(wanted)} scenarios")
        for scenario in wanted:
            stat = entry.get(scenario)
            if stat is None:
                print(f"  scenario={scenario} {_MISSING}")
                continue
            print(
                f"  scenario={scenario} n={stat['n']} correct={stat['n_correct']}/{stat['n']} "
                f"degraded={stat['n_degraded']} agent_failed={stat['n_agent_failed']} "
                f"source={stat['source']}"
            )
    for scenario, paths in scanned.legacy.items():
        for path in paths:
            print(f"legacy scenario={scenario} {_LEGACY} {path.as_posix()}")
    print(f"agents={len(entries)} -> {md_path} + {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
