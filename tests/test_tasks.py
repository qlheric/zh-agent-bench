"""TASK-001 不变式测试。

覆盖 TASK-001 §3 判据 1 的 ⑩ 项：
① 四类各 N 条 / id 唯一且形如 {category}-{index:04d}
② 字段集合、类型、枚举值符合 §2.5
③ answer_spec 合法
④ ground_truth.answer 类型与 answer_spec.kind 一致
⑤ train/heldout 无 id 重叠、heldout 精确计数 round(N*0.2)、heldout 实体池不出现在 train
⑥ 同 seed 两次独立输出目录产物逐字节一致
⑦ round-trip 无损
⑧ jsonl 每行可解析、中文无 \\u 转义、无 CRLF、单 LF 结尾
⑨ instruction 非空且含至少一个 CJK
⑩ 非法参数非零退出且不写文件
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from zh_agent_bench.tasks import (
    A_SCENE_CATEGORIES,
    A_SCENE_SCHEMA_VERSION,
    LEGAL_COMPARISONS,
    AnswerKind,
    AnswerSpec,
    Category,
    Comparison,
    Normalization,
    Split,
    Task,
    build_dataset,
    heldout_count,
    write_dataset,
)

PER_CATEGORY = 8
SEED = 42

CJK_RE = re.compile(r"[一-鿿]")
ID_RE = re.compile(r"^(query|stats|organize|format)-\d{4}$")


@pytest.fixture(scope="module")
def dataset() -> list[Task]:
    return build_dataset(per_category=PER_CATEGORY, seed=SEED)


def _read_jsonl(path: Path) -> list[dict]:
    raw = path.read_bytes().decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line]


# ---------------------------------------------------------------- ① 数量与 id


def test_four_categories_each_have_per_category_tasks(dataset):
    # TASK-008 起 Category 多了 B 场景的 RETRIEVAL，而 build_dataset 只造 A 场景四类，
    # 所以这里按 A_SCENE_CATEGORIES 断言 —— 「恰好四类」这条不变式是**生成器的**，
    # 不是枚举的（枚举多一个 B 场景类别不改变 A 场景生成四类这件事）。
    for category in A_SCENE_CATEGORIES:
        rows = [t for t in dataset if t.category is category]
        assert len(rows) == PER_CATEGORY, f"{category.value} 条数不对"


def test_ids_unique_and_well_formed(dataset):
    ids = [t.id for t in dataset]
    assert len(ids) == len(set(ids)), "id 必须全局唯一"
    for t in dataset:
        assert ID_RE.match(t.id), f"id 形如不合规: {t.id}"
        assert t.id.startswith(t.category.value + "-")


# ---------------------------------------------------------------- ② 字段 / 类型 / 枚举


def test_task_is_frozen(dataset):
    with pytest.raises(Exception):
        dataset[0].id = "x"  # type: ignore[misc]


def test_serialized_field_set_and_types(dataset):
    for t in dataset:
        d = t.to_dict()
        # A 场景的序列化字段集**一字不加**：TASK-008 加的 rubric 键在 A 场景不出现，
        # 否则已入库的 120 条 jsonl 会与「固定命令重新生成」的结果对不上（判据锁会失效）。
        assert set(d) == {
            "id", "category", "schema_version", "instruction",
            "initial_data", "tool_names", "ground_truth",
            "answer_spec", "split",
        }
        # A 场景任务盖 1.0（已入库数据生成时的版本），B 场景才用新的 SCHEMA_VERSION。
        assert d["schema_version"] == A_SCENE_SCHEMA_VERSION == "1.0"
        assert d["category"] in {"query", "stats", "organize", "format"}
        assert d["split"] in {"train", "heldout"}
        assert isinstance(d["initial_data"], dict)
        assert isinstance(d["ground_truth"], dict)
        assert list(d["ground_truth"]) == ["answer"]
        assert isinstance(d["tool_names"], list) and all(isinstance(x, str) for x in d["tool_names"])


def test_tool_names_stable_unique_sorted(dataset):
    expected = {
        Category.QUERY: ("table.query",),
        Category.STATS: ("table.stats",),
        Category.ORGANIZE: ("table.organize",),
        Category.FORMAT: ("format.csv_to_json",),
    }
    for t in dataset:
        names = t.tool_names
        assert names == expected[t.category]
        assert len(names) == len(set(names)), "tool_names 不得重复"
        assert list(names) == sorted(names), "tool_names 必须固定排序"


def test_enum_values_are_exactly_as_specified():
    # TASK-008 起 Category 多了 B 场景的 RETRIEVAL。A 场景四类的**取值与顺序**一字未改，
    # 新类别追加在末尾 —— 枚举序是榜单渲染的稳定契约（A 场景四类的相对顺序不能动）。
    assert [c.value for c in Category] == [
        "query", "stats", "organize", "format", "retrieval", "orchestration",
    ]
    assert [c.value for c in A_SCENE_CATEGORIES] == ["query", "stats", "organize", "format"]
    assert [s.value for s in Split] == ["train", "heldout"]
    assert [k.value for k in AnswerKind] == ["string", "number", "ordered_list", "json"]
    # judge 是 TASK-008 加的 B 场景判分方式，orchestration 是 TASK-019 加的 C 场景判分方式，
    # 两者都追加在末尾；A 场景四个算子一字未改。
    assert [c.value for c in Comparison] == [
        "exact", "decimal_exact", "ordered_list_exact", "json_canonical", "judge",
        "orchestration",
    ]
    assert [n.value for n in Normalization] == ["none", "strip"]


def test_amounts_are_integer_cents_not_float(dataset):
    """确定性类型规则：金额一律整数分，原生 float 禁止进入初始数据。"""
    for t in dataset:
        stack: list = [t.initial_data]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                for v in cur.values():
                    stack.append(v)
            elif isinstance(cur, list):
                stack.extend(cur)
            elif isinstance(cur, float):
                pytest.fail(f"{t.id} 初始数据里出现 float: {cur!r}")


def test_amounts_are_in_cents_range(dataset):
    for t in dataset:
        for row in t.initial_data.get("rows", []):
            if "销售额" in row and isinstance(row["销售额"], int):
                assert row["销售额"] >= 100, f"{t.id} 销售额应按分存储"


def test_dates_are_iso8601(dataset):
    iso = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for t in dataset:
        for row in t.initial_data.get("rows", []):
            if "入职日期" in row:
                assert iso.match(row["入职日期"]), f"{t.id} 日期非 ISO 8601"


# ---------------------------------------------------------------- ③ answer_spec 合法


def test_answer_spec_values_legal(dataset):
    for t in dataset:
        spec = t.answer_spec
        assert spec.kind in set(AnswerKind)
        assert spec.comparison in set(Comparison)
        assert spec.normalization in set(Normalization)
        assert LEGAL_COMPARISONS[spec.kind] is spec.comparison
        assert set(spec.to_dict()) == {"kind", "comparison", "normalization"}


def test_answer_spec_per_category_matches_spec(dataset):
    expected = {
        Category.QUERY: ("string", "exact", "strip"),
        Category.STATS: ("number", "decimal_exact", "none"),
        Category.ORGANIZE: ("ordered_list", "ordered_list_exact", "strip"),
        Category.FORMAT: ("json", "json_canonical", "none"),
    }
    for t in dataset:
        assert (t.answer_spec.kind.value, t.answer_spec.comparison.value,
                t.answer_spec.normalization.value) == expected[t.category]


def test_illegal_comparison_rejected():
    with pytest.raises(ValueError):
        AnswerSpec.from_dict({"kind": "string", "comparison": "decimal_exact",
                              "normalization": "strip"})


# ---------------------------------------------------------------- ④ answer 与 kind 一致


def test_answer_type_matches_kind(dataset):
    kind_map = {
        AnswerKind.STRING: str,
        AnswerKind.NUMBER: str,        # 规范十进制字符串
        AnswerKind.ORDERED_LIST: list,
        AnswerKind.JSON: dict,
    }
    for t in dataset:
        answer = t.ground_truth["answer"]
        expected = kind_map[t.answer_spec.kind]
        assert isinstance(answer, expected) and not isinstance(answer, bool), (
            f"{t.id} answer 类型与 kind 不符"
        )


def test_number_answer_is_two_decimal_string(dataset):
    for t in dataset:
        if t.answer_spec.kind is AnswerKind.NUMBER:
            ans = t.ground_truth["answer"]
            assert re.match(r"^-?\d+\.\d{2}$", ans), f"{t.id} 数值答案应为两位小数字符串: {ans}"
            Decimal(ans)  # 必须能被 Decimal 精确解析


def test_stats_answer_equals_recomputed_mean(dataset):
    """独立复算：金额整数分求和 → 折元 → ROUND_HALF_UP 到两位。"""
    for t in dataset:
        if t.category is not Category.STATS:
            continue
        rows = t.initial_data["rows"]
        cents = sum(r["销售额"] for r in rows)
        avg = (Decimal(cents) / Decimal(len(rows))) / Decimal(100)
        # 必须显式传 ROUND_HALF_UP：Decimal.quantize() 默认是 HALF_EVEN，
        # 而 §2.5 钉死的是 HALF_UP，两者会在 .xx5 处分歧。
        assert t.ground_truth["answer"] == str(
            avg.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )


def test_stats_answer_uses_half_up_not_half_even(dataset):
    """钉死舍入方向：恰好 .xx5 的平均值必须向上，不能走银行家舍入。

    先找出所有 HALF_UP 与 HALF_EVEN 会分歧的 stats 任务：若一个都没有，
    说明样本恰好避开了分歧点，这条断言没有区分力 —— 直接跳过而不是假装通过。
    """
    distinguishing = []
    for t in dataset:
        if t.category is not Category.STATS:
            continue
        rows = t.initial_data["rows"]
        cents = sum(r["销售额"] for r in rows)
        avg = (Decimal(cents) / Decimal(len(rows))) / Decimal(100)
        half_up = str(avg.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        half_even = str(avg.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))
        assert t.ground_truth["answer"] == half_up, f"{t.id} 不是 HALF_UP 结果"
        if half_up != half_even:
            distinguishing.append(t.id)
    if not distinguishing:
        pytest.skip("样本里没有恰好落在 .xx5 的 stats 任务，非缺陷")


def test_query_answer_matches_row_in_initial_data(dataset):
    for t in dataset:
        if t.category is not Category.QUERY:
            continue
        name = re.search(r"『(.+?)』", t.instruction).group(1)
        rows = t.initial_data["rows"]
        hits = [r for r in rows if r["姓名"] == name]
        assert len(hits) == 1, f"{t.id} 查询键在任务内必须唯一"
        assert t.ground_truth["answer"] == hits[0]["部门"]


def test_organize_answer_matches_sort_rule(dataset):
    for t in dataset:
        if t.category is not Category.ORGANIZE:
            continue
        rows = t.initial_data["rows"]
        names = [r["姓名"] for r in rows]
        assert len(names) == len(set(names)), f"{t.id} 姓名任务内必须唯一"
        ranked = sorted(rows, key=lambda r: (-r["销售额"], r["姓名"]))
        assert t.ground_truth["answer"] == [r["姓名"] for r in ranked[:3]]


def test_format_answer_parses_from_csv_text(dataset):
    for t in dataset:
        if t.category is not Category.FORMAT:
            continue
        import csv as _csv
        import io as _io

        dialect = t.initial_data["dialect"]
        assert dialect["delimiter"] == "," and dialect["quotechar"] == '"'
        rows = list(
            _csv.DictReader(
                _io.StringIO(t.initial_data["csv_text"]),
                delimiter=dialect["delimiter"],
                quotechar=dialect["quotechar"],
            )
        )
        expected = {"表名": "商品销售表", "rows": rows}
        assert t.ground_truth["answer"] == expected
        assert "\r\n" not in t.initial_data["csv_text"]


def test_json_answer_canonical_comparable(dataset):
    for t in dataset:
        if t.category is not Category.FORMAT:
            continue
        a = json.dumps(t.ground_truth["answer"], sort_keys=True, ensure_ascii=False)
        b = json.loads(json.dumps(t.ground_truth["answer"]))
        assert json.dumps(b, sort_keys=True, ensure_ascii=False) == a


# ---------------------------------------------------------------- ⑤ split / 防污染


def test_no_id_overlap_between_splits(dataset):
    tr = {t.id for t in dataset if t.split is Split.TRAIN}
    ho = {t.id for t in dataset if t.split is Split.HELDOUT}
    assert tr.isdisjoint(ho)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 50, 100, 150])
def test_heldout_count_rule(n):
    assert heldout_count(n) == int(round(Decimal(n) * Decimal("0.2")))


def test_heldout_count_matches_rule_in_dataset(dataset):
    for category in A_SCENE_CATEGORIES:
        ho = sum(1 for t in dataset if t.category is category and t.split is Split.HELDOUT)
        assert ho == heldout_count(PER_CATEGORY)


def _entity_surface(task: Task) -> set[str]:
    """任务里出现的一切实体（人名/公司/商品/城市/部门）。"""
    text = task.instruction + " " + json.dumps(task.initial_data, ensure_ascii=False)
    out: set[str] = set()
    for row in task.initial_data.get("rows", []):
        for key in ("姓名", "部门", "城市", "商品"):
            if key in row:
                out.add(str(row[key]))
    if "公司" in task.initial_data:
        out.add(str(task.initial_data["公司"]))
    for name in re.findall(r"『(.+?)』", task.instruction):
        out.add(name)
    assert text  # 保持引用，避免 flake 误删
    return out


def test_heldout_entities_never_appear_in_train(dataset):
    """防污染的核心断言：heldout 任务用的实体池在 train 里一个都不出现。"""
    for category in A_SCENE_CATEGORIES:
        train_entities: set[str] = set()
        for t in dataset:
            if t.category is category and t.split is Split.TRAIN:
                train_entities |= _entity_surface(t)
        heldout_tasks = [
            t for t in dataset if t.category is category and t.split is Split.HELDOUT
        ]
        assert heldout_tasks, f"{category.value} 应有 heldout 任务"
        for t in heldout_tasks:
            leaked = _entity_surface(t) & train_entities
            assert not leaked, f"{t.id} 的 heldout 实体泄漏进 train: {sorted(leaked)}"


def test_split_pools_are_disjoint():
    from zh_agent_bench.tasks import split_pools

    tr = split_pools(SEED, "t:name", tuple(f"n{i}" for i in range(10)))
    assert set(tr[Split.TRAIN]).isdisjoint(tr[Split.HELDOUT])
    assert len(tr[Split.TRAIN]) + len(tr[Split.HELDOUT]) == 10


# ---------------------------------------------------------------- ⑥ 确定性


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)).replace("\\", "/"): p.read_bytes()
        for p in sorted(root.rglob("*.jsonl"))
    }


def test_same_seed_two_output_dirs_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_dataset(build_dataset(PER_CATEGORY, SEED), a)
    write_dataset(build_dataset(PER_CATEGORY, SEED), b)
    snap_a, snap_b = _snapshot(a), _snapshot(b)
    assert sorted(snap_a) == sorted(snap_b), "文件清单必须一致"
    assert snap_a == snap_b, "同 seed 产物必须逐字节一致"


def test_different_seed_changes_content(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_dataset(build_dataset(PER_CATEGORY, SEED), a)
    write_dataset(build_dataset(PER_CATEGORY, SEED + 1), b)
    assert _snapshot(a) != _snapshot(b)


def test_write_only_touches_managed_files(tmp_path):
    keeper = tmp_path / "keepme.txt"
    keeper.write_text("不要动我", encoding="utf-8")
    root = tmp_path / "tasks"
    write_dataset(build_dataset(PER_CATEGORY, SEED), root)
    stale = root / "train" / "stale.jsonl"
    stale.write_text("{}\n", encoding="utf-8")
    write_dataset(build_dataset(PER_CATEGORY, SEED), root)
    assert keeper.read_text(encoding="utf-8") == "不要动我"
    assert not stale.exists(), "旧的本单产物应被清掉"


# ---------------------------------------------------------------- ⑦ round-trip


def test_round_trip_byte_identical(dataset, tmp_path):
    for t in dataset:
        line = json.dumps(t.to_dict(), ensure_ascii=False, separators=(",", ":"))
        back = Task.from_dict(json.loads(line))
        again = json.dumps(back.to_dict(), ensure_ascii=False, separators=(",", ":"))
        assert line == again, f"{t.id} round-trip 有损"


def test_round_trip_from_files(tmp_path, dataset):
    write_dataset(dataset, tmp_path)
    # 按 id 索引比较，不依赖遍历顺序 —— 文件路径序、split 枚举序、id 序三者互不相同。
    rebuilt: dict[str, Task] = {}
    for p in sorted(tmp_path.rglob("*.jsonl")):
        for d in _read_jsonl(p):
            rebuilt[d["id"]] = Task.from_dict(d)
    assert len(rebuilt) == len(dataset)
    assert set(rebuilt) == {t.id for t in dataset}
    for original in dataset:
        assert _dump_line(original) == _dump_line(rebuilt[original.id]), (
            f"{original.id} 经文件 round-trip 后不一致"
        )


def _dump_line(task: Task) -> str:
    return json.dumps(task.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"


def test_round_trip_through_actual_files_is_byte_identical(tmp_path):
    write_dataset(build_dataset(PER_CATEGORY, SEED), tmp_path)
    before = _snapshot(tmp_path)
    rebuilt = []
    for p in sorted(tmp_path.rglob("*.jsonl")):
        rebuilt.extend(Task.from_dict(d) for d in _read_jsonl(p))
    write_dataset(rebuilt, tmp_path)
    assert _snapshot(tmp_path) == before


# ---------------------------------------------------------------- ⑧ jsonl 格式


def test_jsonl_is_lf_only_and_single_trailing_newline(tmp_path):
    write_dataset(build_dataset(PER_CATEGORY, SEED), tmp_path)
    for p in sorted(tmp_path.rglob("*.jsonl")):
        raw = p.read_bytes()
        assert b"\r" not in raw, f"{p.name} 含 CR"
        assert raw.endswith(b"\n"), f"{p.name} 必须以单个 LF 结尾"
        assert not raw.endswith(b"\n\n"), f"{p.name} 结尾不能有多个 LF"
        assert raw.decode("utf-8").encode("utf-8") == raw, "必须是 utf-8"


def test_jsonl_lines_parse_and_no_unicode_escape(tmp_path):
    write_dataset(build_dataset(PER_CATEGORY, SEED), tmp_path)
    for p in sorted(tmp_path.rglob("*.jsonl")):
        raw = p.read_bytes().decode("utf-8")
        assert "\\u" not in raw, f"{p.name} 含 \\uXXXX 转义"
        for line in raw.splitlines():
            obj = json.loads(line)  # 每行必须可独立解析
            assert obj["id"]


def test_expected_file_layout(tmp_path):
    write_dataset(build_dataset(PER_CATEGORY, SEED), tmp_path)
    files = sorted(str(p.relative_to(tmp_path)).replace("\\", "/") for p in tmp_path.rglob("*.jsonl"))
    assert files == sorted(
        f"{split}/{cat.value}.jsonl"
        for split in ("train", "heldout")
        for cat in A_SCENE_CATEGORIES
    )


# ---------------------------------------------------------------- ⑨ instruction


def test_instruction_non_empty_with_cjk(dataset):
    for t in dataset:
        assert t.instruction.strip(), f"{t.id} instruction 为空"
        assert CJK_RE.search(t.instruction), f"{t.id} instruction 无 CJK 字符"
        assert "\n" not in t.instruction


# ---------------------------------------------------------------- ⑩ 非法参数


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    # PYTHONIOENCODING=utf-8：本机控制台代码页非 UTF-8，子进程默认按 GBK 写 stdout，
    # 父进程 text=True 也按 GBK 解 —— 两边对上时碰巧不炸，但错误信息含中文时必炸
    # （UnicodeDecodeError -> PytestUnhandledThreadExceptionWarning，findings.md §四.6）。
    # 显式钉死编码，两边都按 UTF-8 走。
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, "-m", "zh_agent_bench.tasks.build", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def test_bad_per_category_exits_nonzero_without_writing(tmp_path):
    out = tmp_path / "out"
    proc = _run_cli("--seed", "42", "--per-category", "0", "--output-dir", str(out))
    assert proc.returncode != 0, "非法 per-category 必须非零退出"
    assert not out.exists(), "非法输入不得写文件"


def test_bad_seed_exits_nonzero_without_writing(tmp_path):
    out = tmp_path / "out"
    proc = _run_cli("--seed", "abc", "--per-category", "4", "--output-dir", str(out))
    assert proc.returncode != 0, "非法 seed 必须非零退出"
    assert not out.exists(), "非法输入不得写文件"


def test_cli_success_prints_summary(tmp_path):
    out = tmp_path / "out"
    proc = _run_cli("--seed", "42", "--per-category", "4", "--output-dir", str(out))
    assert proc.returncode == 0, proc.stderr
    for category in A_SCENE_CATEGORIES:
        ho = heldout_count(4)
        assert f"{category.value}" in proc.stdout
        assert f"{4 - ho:>8}{ho:>10}{4:>8}" in proc.stdout, proc.stdout


def test_cli_two_runs_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert _run_cli("--seed", "42", "--per-category", "4", "--output-dir", str(a)).returncode == 0
    assert _run_cli("--seed", "42", "--per-category", "4", "--output-dir", str(b)).returncode == 0
    assert _snapshot(a) == _snapshot(b)
    assert sorted(_snapshot(a)) == sorted(_snapshot(b))


# ---------------------------------------------------------------- 边界值


def test_per_category_one_works():
    tasks = build_dataset(1, SEED)
    for category in A_SCENE_CATEGORIES:
        rows = [t for t in tasks if t.category is category]
        assert len(rows) == 1
        assert rows[0].split is Split.TRAIN  # heldout_count(1) == 0

# ============================================================================
# TASK-019 回归：C 场景扩展后，A 场景的 schema 语义一个字都不能变
# ============================================================================


def test_orchestration_enum_values_do_not_collide_with_a_or_b_scene():
    """C 场景必须是**独立取值**，不能复用 A 的四类或 B 的 retrieval。
    复用会让 is_b_scene / is_c_scene 判不开，分榜也分不开。"""
    assert Category.ORCHESTRATION.value == "orchestration"
    assert Category.ORCHESTRATION not in A_SCENE_CATEGORIES
    assert Category.ORCHESTRATION is not Category.RETRIEVAL
    assert Comparison.ORCHESTRATION is not Comparison.JUDGE
    assert all(c is not Comparison.ORCHESTRATION for c in LEGAL_COMPARISONS.values())


def test_b_scene_and_c_scene_comparisons_stay_pinned_separately():
    from zh_agent_bench.tasks.schema import B_SCENE_COMPARISON, C_SCENE_COMPARISON

    assert B_SCENE_COMPARISON is Comparison.JUDGE
    assert C_SCENE_COMPARISON is Comparison.ORCHESTRATION
    assert B_SCENE_COMPARISON is not C_SCENE_COMPARISON


def test_is_b_scene_and_is_c_scene_are_mutually_exclusive():
    """B 只认 retrieval、C 只认 orchestration，A 四类两边都不是。"""
    for category in Category:
        assert not (category.is_b_scene and category.is_c_scene), category
    assert Category.RETRIEVAL.is_b_scene and not Category.RETRIEVAL.is_c_scene
    assert Category.ORCHESTRATION.is_c_scene and not Category.ORCHESTRATION.is_b_scene
    for category in A_SCENE_CATEGORIES:
        assert not category.is_b_scene and not category.is_c_scene


def test_a_scene_answer_spec_combinations_are_still_exactly_as_pinned():
    """C 场景放 JUDGE / ORCHESTRATION 过闸，不等于把 A 场景的闸松了：
    A 的四种 (kind, comparison) 对应关系与 LEGAL_COMPARISONS 必须仍逐条成立。"""
    assert LEGAL_COMPARISONS == {
        AnswerKind.STRING: Comparison.EXACT,
        AnswerKind.NUMBER: Comparison.DECIMAL_EXACT,
        AnswerKind.ORDERED_LIST: Comparison.ORDERED_LIST_EXACT,
        AnswerKind.JSON: Comparison.JSON_CANONICAL,
    }
    for kind, comparison in LEGAL_COMPARISONS.items():
        assert AnswerSpec(kind=kind, comparison=comparison,
                          normalization=Normalization.NONE).to_dict() == {
            "kind": kind.value, "comparison": comparison.value, "normalization": "none",
        }


@pytest.mark.parametrize("kind", list(AnswerKind))
@pytest.mark.parametrize("comparison", list(Comparison))
def test_a_scene_gate_rejects_every_kind_comparison_crossing(kind, comparison):
    """完整 (kind, comparison) 笛卡尔积：**只有** LEGAL_COMPARISONS 的四条
    加上 JUDGE / ORCHESTRATION 两种开放式比较能过闸。

    枚举在 Python 里按名相等、与构造来源无关，所以用值比较而不是 `is` ——
    用 `is` 的话 `Comparison('judge') is Comparison.JUDGE` 恰好为真，
    一条本该红（string + judge）的组合会被误判成「本来就合法」。
    """
    payload = {"kind": kind.value, "comparison": comparison.value, "normalization": "none"}
    # 合法集合显式写出来，不从实现里反推 —— 反推的话实现改了测试跟着改，等于没测
    legal = (
        comparison is Comparison.JUDGE
        or comparison is Comparison.ORCHESTRATION
        or LEGAL_COMPARISONS[kind] is comparison
    )
    if legal:
        assert AnswerSpec.from_dict(payload).comparison == comparison
    else:
        with pytest.raises(ValueError, match="answer_spec 非法"):
            AnswerSpec.from_dict(payload)


def test_kind_json_with_orchestration_is_the_one_opened_combination():
    """C 场景唯一新增的合法组合：**就是** kind=json + comparison=orchestration。
    它也是整套 schema 里唯一一处「comparison 不由 kind 决定」的 A/B 例外之外的第三个。"""
    spec = AnswerSpec.from_dict(
        {"kind": "json", "comparison": "orchestration", "normalization": "none"}
    )
    assert spec.kind == AnswerKind.JSON
    assert spec.comparison == Comparison.ORCHESTRATION
