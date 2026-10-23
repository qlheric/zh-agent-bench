"""正式数据集质量测试 —— TASK-004 判据 2。

**测的是入库的 `benchmarks/tasks/`（不是内存里现造的数据）**：
生成器的正确性由 `test_tasks.py` 负责，本文件负责「入库产物本身够不够格」。

七条判据逐条对应：
- ① id 全局唯一且形如 ``{category}-{index:04d}``  → `test_ids_are_globally_unique_and_well_formed`
- ② 同 seed 两次独立目录**逐字节一致**            → `test_same_seed_two_dirs_byte_identical`
- ③ heldout 实体池与 train 无交集（**逐池**断言）   → `test_entity_pools_are_disjoint_per_pool`
- ④ 四类 ground_truth 均可被 answer_spec 判分      → `test_every_ground_truth_is_judgable`
- ⑤ instruction 非空且含 CJK                      → `test_instructions_are_non_empty_and_chinese`
- ⑥ jsonl 中文无 ``\\u`` 转义、LF 无CRLF           → `test_jsonl_is_utf8_without_escapes_and_has_lf_only`
- ⑦ 无重复 id、无真实敏感信息                     → `test_no_duplicate_ids` + `test_no_real_world_sensitive_info`
"""

from __future__ import annotations

import json
import re
import tempfile
import unicodedata
from pathlib import Path

import pytest

from zh_agent_bench.cli import load_tasks
from zh_agent_bench.judge import judge
from zh_agent_bench.tasks.build import (
    CITIES,
    COMPANIES,
    DEPARTMENTS,
    PRODUCTS,
    ROWS_PER_TABLE,
    build_dataset,
    write_dataset,
)
from zh_agent_bench.tasks.schema import A_SCENE_CATEGORIES, Category, Split

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "benchmarks" / "tasks"

#: 入库数据集的正式规模。CLI 榜单的 n= 与这些数字挂钩，扩容时这里和 test_harness 同步改。
PER_CATEGORY = 30
SEED = 42
N_HELDOUT = 24
N_TRAIN = 96


@pytest.fixture(scope="module")
def all_tasks():
    """两个 split 合起来的全部入库任务（按 category 枚举序 + id 排序）。"""
    return load_tasks(DATA_ROOT, Split.TRAIN) + load_tasks(DATA_ROOT, Split.HELDOUT)


# --------------------------------------------------------------------------------------
# 规模（判据 1 的入库侧）
# --------------------------------------------------------------------------------------


def test_committed_dataset_has_formal_scale(all_tasks):
    # 规模按 **A 场景**类别数算：TASK-008 起 Category 多了 B 场景的 RETRIEVAL，
    # 但入库的 benchmarks/tasks/ 仍只有 A 场景四类（B 场景由 tasks/qa.py 单独生成到
    # benchmarks/qa/，brainstorm §4.5 要求分榜）。A 场景的条数一个没变。
    assert len(all_tasks) == PER_CATEGORY * len(A_SCENE_CATEGORIES) == 120
    heldout = load_tasks(DATA_ROOT, Split.HELDOUT)
    train = load_tasks(DATA_ROOT, Split.TRAIN)
    assert len(heldout) == N_HELDOUT, f"heldout 应 {N_HELDOUT} 条，实际 {len(heldout)}"
    assert len(train) == N_TRAIN, f"train 应 {N_TRAIN} 条，实际 {len(train)}"
    for category in A_SCENE_CATEGORIES:
        assert sum(1 for t in train if t.category is category) == 24, category
        assert sum(1 for t in heldout if t.category is category) == 6, category


def test_committed_dataset_matches_the_pinned_generation_command():
    """入库产物必须就是 `--seed 42 --per-category 30` 生成的那一份。

    只比逐字节内容不比 mtime —— 有人手改了 jsonl 也会被抓到。
    """
    with tempfile.TemporaryDirectory() as tmp:
        regen = Path(tmp) / "regen"
        write_dataset(build_dataset(PER_CATEGORY, SEED), regen)
        for path in sorted(regen.rglob("*.jsonl")):
            rel = path.relative_to(regen)
            committed = DATA_ROOT / rel
            assert committed.exists(), f"入库缺文件: {rel}"
            assert committed.read_bytes() == path.read_bytes(), (
                f"{rel} 与 --seed {SEED} --per-category {PER_CATEGORY} 的生成结果不一致"
            )


# --------------------------------------------------------------------------------------
# ① id
# --------------------------------------------------------------------------------------


def test_ids_are_globally_unique_and_well_formed(all_tasks):
    ids = [t.id for t in all_tasks]
    assert len(set(ids)) == len(ids), "id 有重复"
    # 只允许重生成后拼接，跨类不可能撞；正则本身就是一道格式闸。
    pattern = re.compile(r"^(query|stats|organize|format)-\d{4}$")
    for task in all_tasks:
        assert pattern.match(task.id), f"id 格式不对: {task.id}"
        prefix = task.id.rsplit("-", 1)[0]
        assert prefix == task.category.value, f"{task.id} 的前缀与 category 不符"


def test_no_duplicate_ids():
    """与 ① 互补：①查格式，这里查**集合层面**没有重复（含跨文件）。"""
    seen: set[str] = set()
    for path in sorted(DATA_ROOT.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            task_id = json.loads(line)["id"]
            assert task_id not in seen, f"id 跨文件重复: {task_id}"
            seen.add(task_id)


def test_split_field_agrees_with_directory(all_tasks):
    """防污染的物理层：heldout 目录里不许混进 train 任务，反之亦然。"""
    for split in (Split.TRAIN, Split.HELDOUT):
        for task in load_tasks(DATA_ROOT, split):
            assert task.split is split, f"{task.id} 在 {split.value} 目录却标着 {task.split.value}"


# --------------------------------------------------------------------------------------
# ② 确定性
# --------------------------------------------------------------------------------------


def test_same_seed_two_dirs_byte_identical():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a", Path(tmp) / "b"
        write_dataset(build_dataset(PER_CATEGORY, SEED), a)
        write_dataset(build_dataset(PER_CATEGORY, SEED), b)
        files_a = sorted(p.relative_to(a) for p in a.rglob("*.jsonl"))
        files_b = sorted(p.relative_to(b) for p in b.rglob("*.jsonl"))
        assert files_a == files_b, "两次生成的文件清单不一致"
        for rel in files_a:
            assert (a / rel).read_bytes() == (b / rel).read_bytes(), f"{rel} 两次生成不一致"


def test_committed_dataset_is_pure_utf8_without_bom(all_tasks):
    """入库文件被 Git/编辑器加过 BOM 的话，``json.loads`` 仍能过但字节比对会挂。"""
    for path in sorted(DATA_ROOT.rglob("*.jsonl")):
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{path.name} 带 UTF-8 BOM"


# --------------------------------------------------------------------------------------
# ③ 防污染：实体池逐池断言
# --------------------------------------------------------------------------------------


def _entities(split: Split, category: Category) -> dict[str, set[str]]:
    """把该 split、该 category 的任务里出现过的实体按池归类。

    逐池而不是笼统断言「没有交集」—— 人名不撞但城市撞了，前者会漏。

    **按类别限定范围是刻意的**：`split_pools` 的 tag 带 ``category.value``
    （build.py），每个类别各切各的池子。所以真正的不交保证是**同类别内**
    train 实体不出现在同类别 heldout 里；跨类别共用词表是设计使然，
    拿它当泄漏来报只会误报（query 的 heldout 人名本来就可能在 organize 的 train 里）。
    """
    out: dict[str, set[str]] = {"name": set(), "city": set(), "product": set(),
                                "department": set(), "company": set()}
    for task in load_tasks(DATA_ROOT, split):
        if task.category is not category:
            continue
        if "公司" in task.initial_data:
            out["company"].add(task.initial_data["公司"])
        for row in task.initial_data.get("rows", []):
            for column, bucket in (("姓名", "name"), ("城市", "city"),
                                   ("商品", "product"), ("部门", "department")):
                if row.get(column):
                    out[bucket].add(row[column])
        # format 类把数据藏在 csv_text 里，实体不落到 rows 上 —— 单独解一遍。
        csv_text = task.initial_data.get("csv_text")
        if csv_text:
            import csv as _csv
            import io as _io

            for row in _csv.DictReader(_io.StringIO(csv_text)):
                out["name"].add(row["姓名"])
                if row["商品"]:
                    out["product"].add(row["商品"])
                if row["城市"]:
                    out["city"].add(row["城市"])
    return out


def test_entity_pools_are_disjoint_per_pool():
    """逐类别、逐池断言 train 与 heldout 的实体不交。"""
    for category in Category:
        train = _entities(Split.TRAIN, category)
        heldout = _entities(Split.HELDOUT, category)
        for bucket in train:
            if not train[bucket] and not heldout[bucket]:
                continue  # 该类别本来就不用这个池（如 organize 无部门列）
            assert train[bucket], f"{category.value}/train 的 {bucket} 池空了，断言失去区分力"
            assert heldout[bucket], f"{category.value}/heldout 的 {bucket} 池空了，断言失去区分力"
            leaked = train[bucket] & heldout[bucket]
            assert not leaked, f"{category.value} 的 {bucket} 池泄漏: {sorted(leaked)}"


def test_heldout_answers_do_not_appear_in_train(all_tasks):
    """答案层防污染：同类别内，heldout 的正确答案不得与 train 的正确答案重合。

    实体池不交还不够 —— 若两个 split 恰好生成出同一个答案（比如平均值撞了），
    在 train 上背答案就能蒙对 heldout。这条把那个漏洞堵上。

    按类别比：不同类别的答案本就可以相同（都是「研发部」），跨类别比必然误报。
    dict 类答案（format）含逐任务随机金额与整表内容，比「表名」那个常量毫无意义，
    故只对标量与列表答案生效。
    """
    for category in (Category.QUERY, Category.STATS, Category.ORGANIZE):
        def keys(split: Split) -> set[str]:
            out = set()
            for task in load_tasks(DATA_ROOT, split):
                if task.category is not category:
                    continue
                out.add(json.dumps(task.ground_truth["answer"], ensure_ascii=False,
                                   sort_keys=True))
            return out

        train_keys, heldout_keys = keys(Split.TRAIN), keys(Split.HELDOUT)
        assert train_keys, f"{category.value} 的 train 答案为空，断言失去区分力"
        overlap = train_keys & heldout_keys
        assert not overlap, f"{category.value} 答案层泄漏: {sorted(overlap)}"


# --------------------------------------------------------------------------------------
# ④ ground_truth 可判分
# --------------------------------------------------------------------------------------


def test_every_ground_truth_is_judgable(all_tasks):
    """把真值自己喂给judge —— 判据要求「均可被 answer_spec 判分」。

    这是全量断言而非抽样：120 条跑一遍 judge 不到 10 毫秒，抽样反而留漏网之鱼。
    真值判不对 = 数据集自己坏了，榜单上的 100% 就无从谈起。
    """
    for task in all_tasks:
        verdict = judge(task.ground_truth["answer"], task.ground_truth, task.answer_spec)
        assert verdict["correct"] is True, f"{task.id} 的真值判不对自己: {verdict['reason']}"


def test_answer_spec_matches_category(all_tasks):
    expected = {
        Category.QUERY: "exact",
        Category.STATS: "decimal_exact",
        Category.ORGANIZE: "ordered_list_exact",
        Category.FORMAT: "json_canonical",
    }
    for task in all_tasks:
        assert task.answer_spec.comparison.value == expected[task.category], task.id


def test_answer_type_matches_kind(all_tasks):
    """kind 说的是「答案长什么样」，字段得真的对得上。"""
    for task in all_tasks:
        answer = task.ground_truth["answer"]
        kind = task.answer_spec.kind.value
        if kind == "string":
            assert isinstance(answer, str) and answer, task.id
        elif kind == "number":
            # 金额路径绝不出现原生 float（build.py 的铁律），落盘后是十进制字符串。
            assert isinstance(answer, str), f"{task.id} 的数值答案不是字符串"
            assert re.match(r"^-?\d+\.\d{2}$", answer), f"{task.id} 金额格式不对: {answer}"
        elif kind == "ordered_list":
            assert isinstance(answer, list) and answer, task.id
            assert all(isinstance(x, str) for x in answer), task.id
        elif kind == "json":
            assert isinstance(answer, dict) and answer, task.id


def test_organize_answers_are_actually_ranked(all_tasks):
    """organize 的答案必须真按销售额降序（并列按姓名升序），不能是随便三个名字。

    这是判据④的加强：judge 只认「有序列表完全一致」，列表本身对不对得单独验。
    """
    for task in load_tasks(DATA_ROOT, Split.HELDOUT) + load_tasks(DATA_ROOT, Split.TRAIN):
        if task.category is not Category.ORGANIZE:
            continue
        rows = task.initial_data["rows"]
        expected = [r["姓名"] for r in sorted(rows, key=lambda r: (-r["销售额"], r["姓名"]))][:3]
        assert task.ground_truth["answer"] == expected, f"{task.id} 排序答案不自洽"


# --------------------------------------------------------------------------------------
# ⑤ instruction
# --------------------------------------------------------------------------------------


def test_instructions_are_non_empty_and_chinese(all_tasks):
    for task in all_tasks:
        assert task.instruction.strip(), f"{task.id} 指令为空"
        assert any("一" <= ch <= "鿿" for ch in task.instruction), (
            f"{task.id} 指令没有中文字符"
        )
        assert "�" not in task.instruction, f"{task.id} 指令含替换字符 U+FFFD"


def test_instructions_are_normalized_nfc(all_tasks):
    """中文全角/半角若混了NFC 与NFD，肉眼一样但字节不同，agent 侧会做无谓的清洗。

    不管的话，将来换个 Unicode 库清洗就会静默改答案。
    """
    for task in all_tasks:
        assert unicodedata.normalize("NFC", task.instruction) == task.instruction, task.id


# --------------------------------------------------------------------------------------
# ⑥ jsonl 编码
# --------------------------------------------------------------------------------------


def test_jsonl_is_utf8_without_escapes_and_has_lf_only():
    for path in sorted(DATA_ROOT.rglob("*.jsonl")):
        raw = path.read_bytes()
        assert b"\r\n" not in raw, f"{path.name} 含 CRLF"
        assert b"\r" not in raw, f"{path.name} 含裸 CR"
        text = raw.decode("utf-8")  # 解不出来就该当场炸
        assert "\\u" not in text, f"{path.name} 用了 \\u 转义（应 ensure_ascii=False）"
        # 真有中文，且不是转义出来的
        assert any("一" <= ch <= "鿿" for ch in text), f"{path.name} 没有中文"
        assert raw.endswith(b"\n"), f"{path.name} 末尾缺换行"


def test_every_jsonl_line_round_trips():
    """to_dict -> 文件 -> from_dict 严格互逆（schema.py 的契约）。"""
    from zh_agent_bench.tasks.schema import Task

    for path in sorted(DATA_ROOT.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            task = Task.from_dict(json.loads(line))
            assert json.loads(json.dumps(task.to_dict(), ensure_ascii=False)) == json.loads(line)


# --------------------------------------------------------------------------------------
# ⑦ 真实敏感信息
# --------------------------------------------------------------------------------------


def test_no_real_world_sensitive_info(all_tasks):
    """粗筛：词表里不得出现身份证/手机号/银行卡/邮箱形态的串。

    这是**兜底**而非主要手段 —— 词表是人工编写的虚构值（build.py 模块docstring 已声明），
    主要防线在那儿。这里防的是将来有人往词表里粘真实数据。
    """
    patterns = [
        re.compile(r"\d{17}[\dXx]"),          # 身份证号
        re.compile(r"1[3-9]\d{9}"),           # 手机号
        re.compile(r"\b\d{16,19}\b"),         # 银行卡号
        re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),  # 邮箱
    ]
    for task in all_tasks:
        blob = json.dumps(task.to_dict(), ensure_ascii=False)
        for pattern in patterns:
            hit = pattern.search(blob)
            assert hit is None, f"{task.id} 疑似含敏感信息: {hit.group() if hit else ''}"


def test_all_entities_come_from_the_curated_pools(all_tasks):
    """反向自证：所有实体都必须能在 build.py 的词表里查到。

    这条比「看起来像虚构的」硬得多 —— 生成器若哪天开始随机拼词，会立刻被抓到。
    """
    from zh_agent_bench.tasks.build import NAMES

    for task in all_tasks:
        if "公司" in task.initial_data:
            assert task.initial_data["公司"] in set(COMPANIES), task.id

    # 跨类别汇总（词表是全类别共用的，这里比的就是全集）
    used: dict[str, set[str]] = {"name": set(), "city": set(), "product": set(),
                                 "department": set()}
    for split in (Split.TRAIN, Split.HELDOUT):
        for category in Category:
            for bucket, values in _entities(split, category).items():
                if bucket != "company":
                    used[bucket] |= values

    for bucket, pool in (("name", set(NAMES)), ("city", set(CITIES)),
                         ("product", set(PRODUCTS)), ("department", set(DEPARTMENTS))):
        assert used[bucket], f"{bucket} 池空了，断言失去区分力"
        unknown = used[bucket] - pool
        assert not unknown, f"{bucket} 池外实体: {sorted(unknown)}"

    # 别的池不许混进人名池（比如把城市名写进了 NAMES）
    assert not (used["name"] & set(CITIES)), "人名池里混进了城市名"
    assert not (used["name"] & set(PRODUCTS)), "人名池里混进了商品名"


def test_word_pools_meet_the_documented_minimums():
    """§2.5 的规模下限。``split_pools`` 对半切，所以池子太短会导致取数时实体复用。"""
    from zh_agent_bench.tasks.build import NAMES

    assert len(NAMES) >= 60, f"人名只有 {len(NAMES)} 个，下限 60"
    for label, pool in (("部门", DEPARTMENTS), ("城市", CITIES),
                        ("公司", COMPANIES), ("商品", PRODUCTS)):
        assert len(pool) >= 30, f"{label}只有 {len(pool)} 个，下限 30"
    # 对半切之后仍要够一整表用
    for label, pool in (("人名", NAMES), ("商品", PRODUCTS), ("城市", CITIES),
                        ("部门", DEPARTMENTS), ("公司", COMPANIES)):
        assert len(pool) // 2 >= ROWS_PER_TABLE, f"{label}池对半切后不足一表 {ROWS_PER_TABLE} 行"


def test_no_duplicate_entities_within_a_pool():
    """池内若有重复名，``take`` 保证不了「任务内键唯一」，query 就会有歧义。"""
    from zh_agent_bench.tasks.build import NAMES

    for label, pool in (("人名", NAMES), ("部门", DEPARTMENTS), ("城市", CITIES),
                        ("公司", COMPANIES), ("商品", PRODUCTS)):
        assert len(set(pool)) == len(pool), f"{label}池内有重复项"