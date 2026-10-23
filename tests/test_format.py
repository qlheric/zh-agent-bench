"""format 类任务的可解性测试 —— TASK-006 判据 1。

**这份文件守的是一条铁律，不是某个字段**：真值必须能**仅凭** ``instruction`` +
``initial_data`` 唯一重建。

TASK-005 实测出来的缺陷正是这条的反例：`build.py` 把 ``{"表名": "商品销售表"}``
写进 ``ground_truth``，而指令与 ``initial_data`` 里都无处可寻。参考求解器能 100%
只因它硬编码了同一个常量，于是**任何不作弊的 agent 都 0 分** —— 那道题测的不是
agent 会不会调工具，是「你猜不猜得到我写死的常量」（findings §七.1）。

所以下面每条测试都绕开 ``ground_truth`` 与工具，从 ``initial_data`` 自己重建一遍答案，
再与存储值比。这跟 ``reference.solve`` 无关（那走工具），跟真 LLM agent 的处境一致：
它能看到的也只有指令 + 初始数据。
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pytest

from zh_agent_bench.cli import load_tasks
from zh_agent_bench.harness import evaluate
from zh_agent_bench.judge import judge_task
from zh_agent_bench.reference import solve
from zh_agent_bench.tasks.build import FORMAT_TABLE_NAME, build_dataset
from zh_agent_bench.tasks.schema import Category, Split, Task

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "benchmarks" / "tasks"
SRC_ROOT = REPO_ROOT / "src" / "zh_agent_bench"


# --------------------------------------------------------------------------------------
# 样本：入库的 30 条 format（train + heldout）
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def format_tasks() -> list[Task]:
    tasks = load_tasks(DATA_ROOT, Split.TRAIN) + load_tasks(DATA_ROOT, Split.HELDOUT)
    return [t for t in tasks if t.category is Category.FORMAT]


def _rebuild_from_initial_data(task: Task) -> dict:
    """**只**用 ``initial_data`` 重建一个 agent 可能给出的答案。

    这是「可解性」的形式化：一个不作弊的 agent 拿到的全部信息就是这个函数能看到的这些。
    不调工具、不读 ``ground_truth`` —— 与真 LLM agent 的处境等价。
    """
    dialect = task.initial_data["dialect"]
    rows = list(csv.DictReader(
        io.StringIO(task.initial_data["csv_text"]),
        delimiter=dialect["delimiter"],
        quotechar=dialect["quotechar"],
    ))
    return {"表名": task.initial_data["表名"], "rows": rows}


# --------------------------------------------------------------------------------------
# ① 可解性铁律：真值仅凭 instruction + initial_data 唯一重建
# --------------------------------------------------------------------------------------


def test_ground_truth_is_reconstructible_from_initial_data_only(format_tasks):
    """核心一条：重建值 == 存储值。

    这条测试当年能抓住 TASK-005 的缺陷：那时 ``initial_data`` 没有 ``表名``，
    重建出的 ``{"rows": [...]}`` 与存储的 ``{"表名": "商品销售表", "rows": [...]}``
    永远不相等 —— 差异本身就是「这道题不可解」的证据。
    """
    assert len(format_tasks) == 30, "正式数据集应有 30 条 format"
    for task in format_tasks:
        assert _rebuild_from_initial_data(task) == task.ground_truth["answer"], (
            f"{task.id} 的真值无法仅凭 initial_data 重建 —— 这道题对不作弊的 agent 不可解"
        )


def test_instruction_points_at_the_table_name(format_tasks):
    """指令要点明表名出处。

    只在 initial_data 里放着、指令不提，等于留了个「你得自己翻」；而 benchmark 的
    前提是「读指令 + 初始数据」就够解题。断言的是指令里**出现**该表名（可解析），
    而非逐字等于某句写死的模板 —— 生成器换措辞不该让测试假红。
    """
    for task in format_tasks:
        assert "表名" in task.initial_data
        assert task.initial_data["表名"] in task.instruction, (
            f"{task.id} 的指令没有点明表名「{task.initial_data['表名']}」"
        )


def test_instruction_alone_is_not_the_only_source(format_tasks):
    """反方向的护栏：表名**不只在**指令里，initial_data 必须也带。

    只有指令带的话，多轮上下文里指令被截断/改写就又不可解了。两处都在才是稳的。
    """
    for task in format_tasks:
        assert task.initial_data["表名"], f"{task.id} initial_data 的表名不能为空"
        assert isinstance(task.initial_data["表名"], str)


# --------------------------------------------------------------------------------------
# ② initial_data 含 表名 字段
# --------------------------------------------------------------------------------------


def test_initial_data_carries_the_table_name(format_tasks):
    for task in format_tasks:
        assert "表名" in task.initial_data, f"{task.id} 的 initial_data 缺 表名"
        assert task.initial_data["表名"] == FORMAT_TABLE_NAME, task.id


def test_ground_truth_table_name_is_not_a_second_literal(format_tasks):
    """真值里的表名必须**取自** initial_data，不是另抄一份字面量。

    上一版就是这么写的（``ground_truth`` 写死 ``"商品销售表"``，initial_data 里根本没有），
    两份字面量各改一处就会悄悄漂。这里直接比值：漂了就会被这条抓住。
    """
    for task in format_tasks:
        assert task.ground_truth["answer"]["表名"] == task.initial_data["表名"], task.id


def test_no_table_name_leaks_into_a_way_that_bypasses_initial_data(format_tasks):
    """表名不在 ``ground_truth`` 之外的地方重复出现。

    弱版本的口径检查：答案 JSON 只有 ``表名`` 与 ``rows`` 两个键，多一个键就说明
    生成器又开始往真值里塞只对它自己有意义的字段了。
    """
    for task in format_tasks:
        assert set(task.ground_truth["answer"]) == {"表名", "rows"}, task.id


# --------------------------------------------------------------------------------------
# ③ reference 从 initial_data 读表名（硬编码已删）
# --------------------------------------------------------------------------------------


def test_source_no_longer_hardcodes_the_format_table_name():
    """``grep -r _FORMAT_TABLE_NAME src/`` 应无结果 —— 判据 1 ③ 的自动化版。

    用源码扫描而不是只测行为：行为测试证明不了「没人再写死它」，
    下一个人加一句 ``task_name or "商品销售表"`` 兜底行为测试照样绿。
    """
    hits = [
        path.relative_to(SRC_ROOT).as_posix()
        for path in SRC_ROOT.rglob("*.py")
        if "_FORMAT_TABLE_NAME" in path.read_text(encoding="utf-8")
    ]
    assert not hits, f"源码里还有 _FORMAT_TABLE_NAME 硬编码: {hits}"


def test_reference_reads_table_name_from_initial_data(format_tasks):
    """删掉 initial_data 的表名后，reference 应当答不出来（返回 None）而不是照样答对。

    这是「没有暗号」的行为证明：若 reference 还在别处兜底，它仍会返回正确 JSON，
    这条就会红 —— 而它一旦红，就说明假 100% 又回来了。
    """
    task = format_tasks[0]
    stripped = Task(
        id=task.id,
        category=task.category,
        schema_version=task.schema_version,
        instruction=task.instruction,
        initial_data={k: v for k, v in task.initial_data.items() if k != "表名"},
        tool_names=task.tool_names,
        ground_truth=task.ground_truth,
        answer_spec=task.answer_spec,
        split=task.split,
    )
    assert solve(stripped, _tool_box(stripped)) is None, "删掉表名后 reference 仍答对了 —— 硬编码没删干净"


def test_reference_scores_100_on_all_format_tasks(format_tasks):
    """走真实工具链，30 条 format 全对。

    reference 与真 agent 走的是同一条路（工具 → 结构化结果 → 拼答案），
    所以它 100% 说明「这道题现在真的有解」，而不只是「我改了两行字」。
    """
    results = evaluate(format_tasks, solve)
    missed = [row["task_id"] for row in results if not row["correct"]]
    assert not missed, f"format 未全对: {missed}"
    assert len(results) == 30


def _tool_box(task: Task) -> dict:
    """按任务声明的工具造一个最小 _ToolBox（复用 harness 真实的分发逻辑）。"""
    from zh_agent_bench.harness import _ToolBox

    return _ToolBox(task.tool_names)


# --------------------------------------------------------------------------------------
# ④ 生成器自身：同 seed 两次一致，且与入库产物逐字节一致
# --------------------------------------------------------------------------------------


def test_regenerated_format_tasks_round_trip_through_json(format_tasks):
    """入库的 jsonl 往返一次后与内存任务等价 —— 表名经 jsonl 序列化没走样。"""
    for task in format_tasks:
        assert Task.from_dict(task.to_dict()) == task


def test_regenerating_with_the_pinned_command_gives_solvable_tasks():
    """**现生成的** format 任务也必须可解。

    只测入库产物不够：有人改了生成器却忘了重生成，入库仍是旧的旧测试全绿。
    这里现生成一遍，直接验可解性。
    """
    generated = [t for t in build_dataset(30, 42) if t.category is Category.FORMAT]
    assert len(generated) == 30
    for task in generated:
        assert _rebuild_from_initial_data(task) == task.ground_truth["answer"], task.id
        assert task.initial_data["表名"] in task.instruction, task.id


def test_instruction_has_no_unresolved_placeholder(format_tasks):
    """指令里的表名是插值进去的真实值，不是 ``{表名}`` 这种忘了 f-string 的残留。"""
    for task in format_tasks:
        assert "{" not in task.instruction, f"{task.id} 指令里有疑似未插值的花括号: {task.instruction}"


def test_judge_accepts_the_reconstructed_answer(format_tasks):
    """重建出的答案要真能被判分器判对 —— 「能重建」不等于「判分器吃得下」。

    少了这一步会漏掉一类缺陷：重建对了但类型不对（比如表名成了非字符串），
    ``json_canonical`` 照样判错，agent 看着像做对了却拿 0 分。
    """
    for task in format_tasks[:6]:
        verdict = judge_task(task, _rebuild_from_initial_data(task))
        assert verdict["correct"], f"{task.id} 重建出的答案被判错: {verdict['reason']}"
