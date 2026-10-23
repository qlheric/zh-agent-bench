"""clone 即跑入口 —— ``python -m zh_agent_bench``。

默认动作就是**跑 train（调试集）+ 打榜单**：

.. code-block:: bash

    python -m zh_agent_bench                      # 跑 train（dev 调试集）
    python -m zh_agent_bench --split train --limit 8   # 同上，显式写出来
    python -m zh_agent_bench --heldout-only --allow-heldout   # 只读 heldout（官方口径，封存集）
    python -m zh_agent_bench --agent llm --limit 4     # M2：接真 LLM 跑（需 DEEPSEEK_API_KEY）

**heldout 是封存口径（TASK-031）**：默认跑 train，heldout 必须显式加 ``--allow-heldout``。
理由与 B/C 的 ``--allow-blind`` 同构 —— 默认就能读到官方集，等于允许对着它调参，
那「heldout」就只是名字。**不许静默降级**：要 heldout 没给旗子就 exit 2，不偷跑 train。

**防污染仍是硬约束**：``--heldout-only`` 只打开 ``benchmarks/tasks/heldout/*.jsonl``，
连 train 目录都不 list —— 不是「读了不用」，是根本不碰。M2 接真 LLM 时这条不能松。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .agents.llm import make_llm_agent
from .eval_run import safe_agent_dirname
from .harness import evaluate, leaderboard, write_trace
from .llm_client import API_KEY_ENV, DEFAULT_BASE_URL, DEFAULT_MODEL
from .reference import REFERENCE_AGENT_NAME, solve
from .tasks.schema import SCHEMA_VERSION, Split, Task
from .tools.registry import TOOL_CONTRACT_VERSION

#: 数据集根目录。默认按 cwd 找 ``benchmarks/tasks``；找不到就回退到仓库内相对本文件的位置，
#: 这样 clone 下来在任意子目录跑 ``python -m zh_agent_bench`` 也能跑通。
_DEFAULT_ROOT = Path("benchmarks/tasks")
_REPO_ROOT = Path(__file__).resolve().parents[2]


#: 拒绝读 heldout 时给的那句人话。**必须点名 heldout 与 ``--allow-heldout``** ——
#: 报错说不出怎么解，调用方就只能去翻源码。
_HELDOUT_REFUSED = (
    "heldout 是封存口径，日常调试请用 train；确要跑官方集请加 --allow-heldout"
)

#: heldout 封存清单：``{"<仓库相对路径>": "<sha256>"}``。校验入口
#: ``python -m zh_agent_bench --verify-heldout-manifest``。
_HELDOUT_MANIFEST = Path("benchmarks/heldout-manifest.json")

#: 清单覆盖的目录。A 的任务集在 ``tasks/heldout``、产物在 ``office/heldout``。
_HELDOUT_ROOTS = ("benchmarks/tasks/heldout", "benchmarks/office/heldout")


def check_heldout_allowed(split: Split, *, allow_heldout: bool) -> None:
    """heldout 是封存口径：没显式 ``--allow-heldout`` 就抛 :class:`PermissionError`。

    与 :func:`eval_run.check_blind_allowed` 同构。**抛异常而不是返回 False** ——
    调用方自己记得检查一遍，早晚漏一个；漏的那个就是「随手能读官方集」，正是本单要消灭的。
    """
    if split is Split.HELDOUT and not allow_heldout:
        raise PermissionError(_HELDOUT_REFUSED)


def _heldout_files(root: Path | None = None) -> list[Path]:
    """清单该覆盖的 ``*.jsonl``（相对 :data:`_HELDOUT_ROOTS`），排序后返回。

    排序是确定性的 —— 清单与校验输出都逐字节可复现。
    """
    base = Path(root) if root is not None else Path(".")
    found: list[Path] = []
    for rel in _HELDOUT_ROOTS:
        found.extend(sorted((base / rel).rglob("*.jsonl")))
    return found


def heldout_manifest_payload(root: Path | None = None) -> dict[str, str]:
    """当前磁盘上的 heldout 清单：``{"<仓库相对路径>": "<sha256>"}``。"""
    base = Path(root) if root is not None else Path(".")
    return {
        path.relative_to(base).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _heldout_files(base)
    }


def build_heldout_manifest(root: Path | None = None) -> str:
    """生成清单文本。**缩进 + 排好序**，同磁盘两次生成逐字节一致。"""
    payload = heldout_manifest_payload(root)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def verify_heldout_manifest(manifest: Path | None = None, root: Path | None = None) -> list[str]:
    """核验清单与磁盘一致，返回**差异列表**（空 = 一致）。

    三种差异都算不一致：内容变了、清单里有磁盘上没有的、磁盘上有清单里没有的 ——
    后两种正是「官方集被换掉了」和「有人绕过清单加了一份」。
    """
    path = Path(manifest) if manifest is not None else _HELDOUT_MANIFEST
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{path.as_posix()} 读不了：{exc}"]
    if not isinstance(recorded, dict):
        return [f"{path.as_posix()} 顶层不是 {{路径: sha256}}"]

    actual = heldout_manifest_payload(root)
    diffs: list[str] = []
    for name in sorted(set(recorded) | set(actual)):
        want, got = recorded.get(name), actual.get(name)
        if want == got:
            continue
        if name not in actual:
            diffs.append(f"清单里有、磁盘上没有：{name}")
        elif name not in recorded:
            diffs.append(f"磁盘上有、清单里没有：{name}（改了 heldout 必须同步更新清单）")
        else:
            diffs.append(f"内容变了：{name}（清单 {want[:12]}… / 磁盘 {got[:12]}…）")
    return diffs


def _resolve_root(explicit: str | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    if _DEFAULT_ROOT.is_dir():
        return _DEFAULT_ROOT
    return _REPO_ROOT / _DEFAULT_ROOT


def load_tasks(root: Path, split: Split) -> list[Task]:
    """从 ``root/<split>/*.jsonl`` 读任务。**只 list 目标 split 目录。**"""
    directory = root / split.value
    if not directory.is_dir():
        raise FileNotFoundError(f"找不到数据集目录: {directory}（先跑 `python -m zh_agent_bench.tasks.build` 生成）")
    tasks: list[Task] = []
    # sorted 保证读入顺序确定 → 榜单逐字节可复现（判据 4）。
    for path in sorted(directory.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                tasks.append(Task.from_dict(json.loads(line)))
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{path.name}:{line_no} 解析失败: {exc}") from exc
    # 再按 id 排一次：文件名排序与 id 排序未必一致（query.jsonl 先于 stats.jsonl，
    # 但 id 是 stats-0001 < query-0001）。榜单顺序按 id 才与文件布局解耦。
    tasks.sort(key=lambda t: t.id)
    return tasks


def _record_rows(results: Sequence[dict], agent_name: str) -> list[dict]:
    """把 :func:`harness.evaluate` 的结果转成榜单可读的 ``records.jsonl`` 每行。

    字段与 B/C 两场景的产物**同构**（``task_id/category/correct/G/score/reason``），
    这样 :mod:`zh_agent_bench.leaderboard` 的 ``aggregate_a`` 不用为 A 场景单开一套解析。

    - ``G`` 恒为 ``null``：A 场景是客观题 ``exact`` 比对，没有 LLM 打分维度
      （榜单那一列本来就渲染成 ``n/a``，给个数反而会被读成「打过分」）。
    - ``score`` 是**单条**的 1.0/0.0（该题对不对），整批的正确率在榜单里现算 ——
      把总分写进每行会让「按行平均」和「整批相除」给出两个不同的数。
    - ``agent`` 是自证署名，与 B/C 两场景的 records 同构。``judge`` 恒为 ``null``：
      A 场景是逐字比对，**根本不经过 judge** —— 写个 judge 名等于谎称这条分是模型判的。
    - 按 ``task_id`` 排序：同输入逐字节可复现，且与文件布局解耦（与 ``load_tasks`` 同理）。
    """
    rows = [
        {
            "task_id": row["task_id"],
            "category": row["category"],
            "correct": bool(row["correct"]),
            "G": None,
            "score": 1.0 if row["correct"] else 0.0,
            "reason": row["reason"],
            "agent": agent_name,
            "judge": None,
        }
        for row in results
    ]
    rows.sort(key=lambda r: r["task_id"])
    return rows


def write_records(results: Sequence[dict], path: str | Path, *, agent_name: str) -> Path:
    """写逐条跑分产物 ``records.jsonl``（UTF-8 / 不转义中文 / LF）。

    ``--out-records`` 的两种形态在这里分流（见 :func:`resolve_records_path`）：
    **完整文件路径**原样写、且**父目录不存在直接报错不自动建** —— 路径打错就查不到
    东西，自动 mkdir 会把「路径打错了」变成「静默写到别处去了」；榜单是对外材料，
    写不出来必须看得见 —— 静默跳过等于报了一份上一次的分数。

    **目录**则自动派生成 ``<目录>/<agent 安全名>/records.jsonl`` 并建目录 ——
    那条路径没人手打过，两个 agent 跑同一档各落各的目录，不会互相覆盖。
    """
    out = resolve_records_path(path, agent_name=agent_name)
    if Path(path).suffix != ".jsonl":
        out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in _record_rows(results, agent_name):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return out


def resolve_records_path(path: str | Path, *, agent_name: str) -> Path:
    """``--out-records`` 的值 → 真正的 ``records.jsonl`` 路径。

    **以 ``.jsonl`` 结尾 = 用户给的是文件路径**，原样用（父目录不存在由调用方报错）；
    **否则 = 目录**，派生成 ``<目录>/<agent 安全名>/records.jsonl``。

    分流依据是**有没有 ``.jsonl`` 后缀**，而不是新加一个开关 —— 加开关等于让调用方
    自己记住「这回要不要拼 agent 目录」，而拼错的后果是静默互相覆盖，正是本单要消灭的。
    """
    given = Path(path)
    if given.suffix == ".jsonl":
        return given
    return given / safe_agent_dirname(agent_name) / "records.jsonl"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m zh_agent_bench",
        description="zh-agent-bench M1 评测入口：跑参考 agent，出榜单。",
    )
    parser.add_argument("--heldout-only", action="store_true",
                        help="只跑 heldout（等价于 --split heldout）——封存口径，需显式加 --allow-heldout")
    parser.add_argument("--split", choices=[s.value for s in Split],
                        help="跑指定 split（默认 train=dev 调试集；heldout 是封存口径，需 --allow-heldout）")
    parser.add_argument("--allow-heldout", action="store_true",
                        help="明确同意跑 heldout（官方口径）。不加就 exit 2，**不会静默降级到 train**")
    parser.add_argument("--verify-heldout-manifest", action="store_true",
                        help="核验 benchmarks/heldout-manifest.json 与磁盘一致（不一致 exit 2 并列出差异）")
    parser.add_argument("--write-heldout-manifest", action="store_true",
                        help="按当前磁盘重写 benchmarks/heldout-manifest.json")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（调试用）")
    parser.add_argument("--data-root", default=None, help="数据集根目录（默认 benchmarks/tasks）")
    # M2 起：--agent 选跑谁。默认 reference —— M1 的判分可复现口径不因加 LLM 而改变。
    parser.add_argument("--agent", choices=["reference", "llm"], default="reference",
                        help="跑哪个 agent（默认 reference）")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"LLM 模型名（仅 --agent llm；默认 {DEFAULT_MODEL}）")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"OpenAI 兼容端点（仅 --agent llm；默认 {DEFAULT_BASE_URL}）")
    parser.add_argument("--api-key", default=None,
                        help=f"API key（仅 --agent llm）。**默认从环境变量 {API_KEY_ENV} 读，请勿写进命令行历史**")
    parser.add_argument("--trace", default=None, metavar="PATH",
                        help="把每条轨迹落盘成 jsonl（全量记录工具调用与结果，供事后分析 agent 为什么答错）")
    parser.add_argument("--out-records", default=None, metavar="PATH",
                        help="把逐条跑分结果落成 records.jsonl（字段与 B/C 同构，含 agent/judge 署名）。"
                             "**以 .jsonl 结尾 = 完整文件路径**（父目录不存在直接报错，不自动建）；"
                             "**否则 = 目录**，自动落成 <目录>/<agent 名>/records.jsonl（自动建目录，"
                             "两 agent 不互相覆盖）。缺省不写")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须为正整数")

    if args.write_heldout_manifest:
        text = build_heldout_manifest()
        _HELDOUT_MANIFEST.write_text(text, encoding="utf-8", newline="\n")
        print(f"已写 {_HELDOUT_MANIFEST.as_posix()}（{len(heldout_manifest_payload())} 份文件）")
        return 0
    if args.verify_heldout_manifest:
        diffs = verify_heldout_manifest()
        if diffs:
            print("heldout 封存清单与磁盘不一致：", file=sys.stderr)
            for line in diffs:
                print(f"  - {line}", file=sys.stderr)
            print("改了 heldout 必须同步更新清单：--write-heldout-manifest", file=sys.stderr)
            return 2
        print(f"heldout 封存清单与磁盘一致（{len(heldout_manifest_payload())} 份文件）")
        return 0

    # 口径：显式 --split / --heldout-only 要 heldout 就走封存闸；否则默认 train（dev 调试集）。
    # 闸在读盘之前 —— 被拒时 tasks/heldout 连 list 都不该发生，更不许偷偷改成 train。
    if args.heldout_only:
        split = Split.HELDOUT
    elif args.split:
        split = Split(args.split)
    else:
        split = Split.TRAIN
    try:
        check_heldout_allowed(split, allow_heldout=args.allow_heldout)
    except PermissionError as exc:
        print(f"封存闸拒绝：{exc}", file=sys.stderr)
        return 2

    root = _resolve_root(args.data_root)
    tasks = load_tasks(root, split)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    if args.agent == "llm":
        agent = make_llm_agent(model=args.model, base_url=args.base_url, api_key=args.api_key)
        agent_name = agent.name
    else:
        agent = solve
        agent_name = REFERENCE_AGENT_NAME

    print(
        f"# schema={SCHEMA_VERSION} tool_contract={TOOL_CONTRACT_VERSION} "
        f"split={split.value} n={len(tasks)} agent={agent_name}",
        file=sys.stderr,
    )
    results = evaluate(tasks, agent)
    if args.trace:
        path = write_trace(results, args.trace)
        print(f"trace            : {path} ({len(results)} 条)", file=sys.stderr)
    if args.out_records:
        try:
            records_path = write_records(results, args.out_records, agent_name=agent_name)
        except OSError as exc:
            print(f"records 落盘失败: {exc}", file=sys.stderr)
            return 2
        print(f"records          : {records_path} ({len(results)} 条)", file=sys.stderr)
    print(leaderboard(results, agent_name=agent_name))
    return 0


if __name__ == "__main__":  # pragma: no cover - __main__ 走 __main__.py
    raise SystemExit(main())
