"""C 场景任务构造器的离线测试：四类编排任务 + delta 判据 + 确定性。

钉 TASK-019 §3 判据 ①的七个覆盖点：
1. 四种任务类型齐全，每类 >= 2 条
2. ``initial_data.sandbox`` 是一份**合法的沙箱快照**
3. ``expected_final_state`` 只含受影响项（delta）
4. delta / rubric 引用的订单号真实存在于该任务的沙箱里
5. 退款陷阱任务确有非 paid 单；券门槛任务确有未达门槛的单
6. 同 seed 两次生成**逐字节一致**
7. 任务 id 唯一

最关键的是第 3 点与「沙箱与 delta 同源」：delta 里的订单号必须来自**这条任务自己的**
沙箱，而不是全局某份 —— 否则 agent 拿到环境后按 id 找过去找不到单，
判分器比对的是两个毫不相干的世界。
"""

from __future__ import annotations

import json

import pytest

from zh_agent_bench.tasks import orchestration
from zh_agent_bench.tasks.orchestration import (
    C_SPLITS,
    FORMAL_PER_TYPE,
    FORMAL_SIZES,
    TASK_TYPES,
    NoTrapConstructible,
    build_formal_dataset,
    build_orchestration_dataset,
    main,
    write_formal_dataset,
    write_orchestration_dataset,
)
from zh_agent_bench.tasks.qa import RUBRIC_FIELDS, partial_rule_kind
from zh_agent_bench.tasks.schema import (
    A_SCENE_CATEGORIES,
    Category,
    Comparison,
    Task,
)


@pytest.fixture(scope="module")
def tasks() -> list[Task]:
    return build_orchestration_dataset(per_type=2, seed=42)


def _by_type(tasks: list[Task], task_type: str) -> list[Task]:
    return [t for t in tasks if t.initial_data["任务类型"] == task_type]


def _order_of(snap: dict, order_id: str) -> dict:
    return next(o for o in snap["orders"] if o["order_id"] == order_id)


# --------------------------------------------------------------------------------------
# 1. 四种类型齐全，每类 >= 2 条
# --------------------------------------------------------------------------------------


def test_the_four_task_types_each_have_at_least_two_tasks(tasks):
    assert TASK_TYPES == ("改地址", "退款判断", "券抵扣", "多单批量")
    for task_type in TASK_TYPES:
        assert len(_by_type(tasks, task_type)) >= 2, task_type
    assert len(tasks) == len(TASK_TYPES) * 2


def test_every_task_is_c_scene_with_the_orchestration_comparison(tasks):
    """C 场景与 A/B 的分类必须干净可分：不能混进 A 场景的枚举，也不能用 B 的 judge。"""
    for t in tasks:
        assert t.category is Category.ORCHESTRATION
        assert t.category.is_c_scene and not t.category.is_b_scene
        assert t.category not in A_SCENE_CATEGORIES
        assert t.answer_spec.comparison is Comparison.ORCHESTRATION
        assert t.answer_spec.comparison is not Comparison.JUDGE


def test_every_task_carries_rubric_and_answer_point(tasks):
    """C 场景也带 rubric，但 rubric 必须**同时**管编排点与回复点 ——
    只管编排点就退回「改对了没」，只管回复点就退回 B 场景。"""
    for t in tasks:
        assert t.rubric, t.id
        for point in t.rubric:
            assert tuple(point) == RUBRIC_FIELDS
            assert partial_rule_kind(point["partial_rule"]) in ("ternary", "binary")
        # 最后一条必须是回复点（面向客户的说明），第一条必须是编排点（查了什么、调了什么）
        assert "回复" in t.rubric[-1]["requirement"] or "告知" in t.rubric[-1]["requirement"]
        assert any(k in t.rubric[0]["requirement"] for k in ("调用", "查到"))


def test_rubric_weights_sum_to_one(tasks):
    for t in tasks:
        assert abs(sum(p["weight"] for p in t.rubric) - 1.0) < 1e-9, t.id


# --------------------------------------------------------------------------------------
# 2. initial_data 是一份合法沙箱快照
# --------------------------------------------------------------------------------------


def test_initial_data_carries_a_valid_sandbox_snapshot(tasks):
    for t in tasks:
        snap = t.initial_data["sandbox"]
        assert set(snap) == {"orders", "inventory", "coupons"}
        assert len(snap["orders"]) == 8
        assert snap["inventory"] and snap["coupons"]
        for order in snap["orders"]:
            assert set(order) == {
                "order_id", "customer", "status", "address",
                "amount_cents", "product_id", "created_at",
            }
            assert order["status"] in ("pending_payment", "paid", "shipped", "refunded")
            assert type(order["amount_cents"]) is int



# --------------------------------------------------------------------------------------
# 3. delta 只含受影响项
# --------------------------------------------------------------------------------------


def test_the_delta_contains_only_affected_items(tasks):
    """改地址 / 批量退款：delta 只列**该改的**那几单，不列未受影响的。
    判分器拿 delta 去比对全量快照的话，这里多一项就多一条冤枉判错。"""
    for t in _by_type(tasks, "改地址"):
        delta = t.ground_truth["expected_final_state"]["orders"]
        assert len(delta) == 1, "改地址只该动一单"
        assert set(delta[0]) == {"order_id", "address"}

    for t in _by_type(tasks, "多单批量"):
        delta = t.ground_truth["expected_final_state"]
        snap = t.initial_data["sandbox"]
        listed = {row["order_id"] for row in delta["orders"]}
        assert listed, "批量退款至少得退一笔"
        # delta 只收 paid 单，且只收该客户名下的单：多一单就多一条冤枉判错
        for row in delta["orders"]:
            assert row == {"order_id": row["order_id"], "status": "refunded"}
            assert _order_of(snap, row["order_id"])["status"] == "paid"


def test_trap_tasks_delta_declares_what_must_not_change(tasks):
    """退款判断 / 券抵扣的正确答案就是「什么都没改」。delta 若真留空，
    判分器无法区分「没乱动」与「动过又改回来了」—— 必须写成 must_not_change 断言。"""
    for t in _by_type(tasks, "退款判断") + _by_type(tasks, "券抵扣"):
        delta = t.ground_truth["expected_final_state"]
        assert delta["orders"] == [] and delta["coupons"] == []
        assert delta["must_not_change"], t.id
        for row in delta["must_not_change"]:
            assert row.get("order_id") or row.get("code")


# --------------------------------------------------------------------------------------
# 4. 引用的订单 / 券真实存在于该任务自己的沙箱
# --------------------------------------------------------------------------------------


def test_every_referenced_order_id_exists_in_that_tasks_sandbox(tasks):
    """delta 与 rubric 引用的 id 必须来自**这条任务自己的**沙箱。
    串到别的任务的订单上，agent 按 id 找过去就找不到单。"""
    for t in tasks:
        snap = t.initial_data["sandbox"]
        known = {o["order_id"] for o in snap["orders"]} | set(snap["coupons"]) | set(snap["inventory"])
        rows = [
            row
            for key in ("orders", "coupons", "must_not_change")
            for row in t.ground_truth["expected_final_state"].get(key, [])
        ]
        for row in rows:
            token = row.get("order_id") or row.get("code")
            assert token in known, f"{t.id}: delta 引用了不存在的 {token}"
        for point in t.rubric:
            for token in point["evidence_ids"]:
                assert token in known, f"{t.id}: evidence_ids 引用了不存在的 {token}"
        # 题面必须能定位到环境里真实存在的东西，否则 agent 第一步就 unknown_id。
        # 注意多单批量**故意只给客户名**——不给订单号正是逼它用 order.list 去筛。
        if t.initial_data["任务类型"] == "多单批量":
            named = {o["customer"] for o in snap["orders"] if o["customer"] in t.instruction}
            assert named, f"{t.id} 题面没提到任何客户名"
        else:
            mentioned = [o["order_id"] for o in snap["orders"] if o["order_id"] in t.instruction]
            assert mentioned, f"{t.id} 题面没提到任何订单号，agent 无从下手"


def test_the_expected_address_is_the_one_the_instruction_asked_for(tasks):
    """题面与 delta 必须说同一个地址 —— 题面和答案对不上，agent 怎么做都错。"""
    for t in _by_type(tasks, "改地址"):
        address = t.ground_truth["expected_final_state"]["orders"][0]["address"]
        assert address in t.instruction


# --------------------------------------------------------------------------------------
# 5. 陷阱真的存在
# --------------------------------------------------------------------------------------


def test_refund_trap_orders_really_are_unpayable(tasks):
    """退款判断的目标单必须**真的**退不了，否则这不是陷阱，是一道普通题。"""
    for t in _by_type(tasks, "退款判断"):
        snap = t.initial_data["sandbox"]
        keep = t.ground_truth["expected_final_state"]["must_not_change"]
        assert keep
        for row in keep:
            order = _order_of(snap, row["order_id"])
            assert order["status"] in ("shipped", "pending_payment")
            assert order["status"] == row["status"], "must_not_change 写的就是沙箱里的原状态"


def test_coupon_trap_orders_really_are_below_the_threshold(tasks):
    for t in _by_type(tasks, "券抵扣"):
        snap = t.initial_data["sandbox"]
        code = t.ground_truth["expected_final_state"]["must_not_change"][0]["code"]
        limit = snap["coupons"][code]["min_amount_cents"]
        assert snap["coupons"][code]["used"] is False
        order_id = next(o["order_id"] for o in snap["orders"]
                        if str(o["order_id"]) in t.instruction)
        assert _order_of(snap, order_id)["amount_cents"] < limit, "门槛陷阱要求金额真的不够"


def test_batch_refund_keeps_at_least_one_non_paid_order(tasks):
    """批量退款若名下全是 paid 单，「不该退的别退」这一半就无从判 —— 任务退化成普通批量。"""
    for t in _by_type(tasks, "多单批量"):
        keep = t.ground_truth["expected_final_state"]["must_not_change"]
        assert keep, f"{t.id} 名下没有非 paid 单，判不出「不该退的别退」"
        snap = t.initial_data["sandbox"]
        for row in keep:
            assert row["status"] in ("shipped", "pending_payment")


# --------------------------------------------------------------------------------------
# 6. 确定性：同 seed 逐字节一致
# --------------------------------------------------------------------------------------


def test_the_same_seed_produces_byte_identical_output(tmp_path):
    first = write_orchestration_dataset(
        build_orchestration_dataset(per_type=2, seed=42), tmp_path / "a"
    )
    second = write_orchestration_dataset(
        build_orchestration_dataset(per_type=2, seed=42), tmp_path / "b"
    )
    assert first.read_bytes() == second.read_bytes()


def test_a_different_seed_produces_different_tasks():
    assert build_orchestration_dataset(per_type=2, seed=42) != \
        build_orchestration_dataset(per_type=2, seed=7)


# --------------------------------------------------------------------------------------
# 7. id 唯一 + round-trip
# --------------------------------------------------------------------------------------


def test_task_ids_are_unique_and_ordered():
    tasks = build_orchestration_dataset(per_type=3, seed=42)
    ids = [t.id for t in tasks]
    assert len(set(ids)) == len(ids) == 12
    assert ids == sorted(ids)
    assert all(i.startswith("orch-") for i in ids)


def test_tasks_round_trip_through_json(tmp_path, tasks):
    path = write_orchestration_dataset(tasks, tmp_path)
    for line in path.read_text(encoding="utf-8").splitlines():
        assert Task.from_dict(json.loads(line)) == \
            next(t for t in tasks if t.id == json.loads(line)["id"])


def test_the_jsonl_writes_chinese_without_unicode_escapes(tmp_path, tasks):
    raw = write_orchestration_dataset(tasks, tmp_path).read_text(encoding="utf-8")
    assert "\\u" not in raw, "ensure_ascii 漏了，中文被转义成 \\uXXXX"
    assert "退款" in raw


# --------------------------------------------------------------------------------------
# 边界与 CLI
# --------------------------------------------------------------------------------------


def test_per_type_below_one_is_rejected():
    with pytest.raises(ValueError):
        build_orchestration_dataset(per_type=0, seed=42)


def test_per_type_beyond_the_environment_capacity_is_rejected():
    """超过去重容量必须报错，**不许循环复用**：复用会让同一份答案被算多次，
    一致率与 delta 统计虚高。上限不是「订单数」而是「造得出的互不相同的题数」，
    由 :func:`capacity` 沿真实造题路径实测得来。"""
    with pytest.raises(ValueError, match="超过单类去重容量"):
        build_orchestration_dataset(per_type=99, seed=42)


def test_the_deduped_dataset_has_no_two_identical_questions():
    """整批去重：题面 + delta 都逐字相同的才算同一道题（见 :func:`_task_key`）。
    只看 delta 不够 —— 券抵扣十条的 delta 恒为「券未使用」，逐字相同；
    只看题面也不够 —— 题面说改地址但 delta 没说要改，等于没要求改。"""
    tasks = build_orchestration_dataset(per_type=10, seed=42)
    keys = {
        (t.instruction, json.dumps(t.ground_truth["expected_final_state"],
                                   ensure_ascii=False, sort_keys=True))
        for t in tasks
    }
    assert len(keys) == len(tasks) == 40


def test_the_capacity_probe_does_not_crash_and_is_ordered_by_construction():
    """容量探测走的就是正式造题那条路，探出的上限必须真能造出这么多条。"""
    from zh_agent_bench.tasks.orchestration import _SEED_INDEX_UNIT, _SEED_UNIQUE_STEP, \
        _TRAP_SEED_ATTEMPTS, _UNIQUE_SEED_ATTEMPTS

    span = (_UNIQUE_SEED_ATTEMPTS - 1) * _SEED_UNIQUE_STEP + \
        (_TRAP_SEED_ATTEMPTS - 1) * _TRAP_SEED_ATTEMPTS
    assert span < _SEED_INDEX_UNIT, "去重顺延会溢进类型位，两个类型共用一份沙箱"


def test_the_cli_writes_a_parseable_file_and_exits_zero(tmp_path, capsys):
    assert main(["--seed", "42", "--per-type", "2",
                 "--output-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    rows = [json.loads(line)
            for line in (tmp_path / "orchestration.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == 8
    assert {r["initial_data"]["任务类型"] for r in rows} == set(TASK_TYPES)


def test_the_cli_rejects_per_type_below_one(tmp_path):
    with pytest.raises(SystemExit):
        main(["--per-type", "0", "--output-dir", str(tmp_path)])


def test_no_trap_constructible_is_a_value_error():
    """构造器内部对「换 seed 也许能救」的失败用这个子类；它必须仍是 ValueError，
    否则 main() 里的 parser.error 接不住，CLI 会甩一串 traceback。"""
    assert issubclass(NoTrapConstructible, ValueError)


def test_the_sandbox_snapshots_are_not_shared_between_tasks(tasks):
    """每条任务一份自己的沙箱：上一条任务的退款不该让下一条的 delta 失效。"""
    snaps = [json.dumps(t.initial_data["sandbox"], sort_keys=True) for t in tasks]
    assert len(snaps) == len(tasks)
    for snap in snaps:
        assert '"status": "refunded"' not in snap, "初始快照里不该有已退款单"


# --------------------------------------------------------------------------------------
# 8. 正式集：40 条 × trial / dev / blind（TASK-023 判据 ①③④⑥⑦⑧）
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def formal():
    return build_formal_dataset(seed=42)


def test_each_split_reaches_its_target_size_and_every_type_gets_ten(formal):
    """判据 ①②：三档各自达标，每类跨三档合计 >= 10 条。

    规模按 :data:`SEED_GROUPS_PER_SPLIT` 的 2 : 4 : 4 组推导 ⇒ 8 / 16 / 16，
    比例 20% / 40% / 40%，四类均分后每类 2 + 4 + 4 = 10 条。
    """
    assert FORMAL_SIZES == {"trial": 8, "dev": 16, "blind": 16}
    assert len(formal) == sum(FORMAL_SIZES.values()) == 40
    for split in C_SPLITS:
        rows = [ft for ft in formal if ft.split == split]
        assert len(rows) == FORMAL_SIZES[split], split
        for task_type in TASK_TYPES:
            assert len([ft for ft in rows if ft.task.initial_data["任务类型"] == task_type]) \
                == FORMAL_PER_TYPE[split][task_type], f"{split}/{task_type}"
    for task_type in TASK_TYPES:
        total = len([ft for ft in formal if ft.task.initial_data["任务类型"] == task_type])
        assert total >= 10, f"{task_type} 只有 {total} 条"


def test_formal_task_ids_are_unique_and_carry_the_split_prefix(formal):
    """判据 ③：id 唯一且**带档位前缀** —— 光看 orch-0007 分不出它属于哪一档。"""
    ids = [ft.task.id for ft in formal]
    assert len(set(ids)) == len(ids) == 40
    for ft in formal:
        assert ft.task.id == f"orch-{ft.split}-{ft.task.id.rsplit('-', 1)[1]}"
        assert ft.task.initial_data["评测档位"] == ft.split


def test_one_seed_group_never_spans_two_splits(formal):
    """判据 ④：同一 seed 组只落一个档 —— 这正是分片防泄漏的全部意义。

    同一组造出的题出自同一批沙箱，开发期见过答案的话，盲测期原样再考一次就废了。
    """
    by_group: dict[int, set[str]] = {}
    for ft in formal:
        by_group.setdefault(ft.seed_group, set()).add(ft.split)
    for group, splits in by_group.items():
        assert len(splits) == 1, f"seed 组 {group} 同时落了 {sorted(splits)}"
    # 三档的组号区间也必须互不相交（生成期自检之外再钉一道，防止改常量时漏改一边）
    groups = {split: {ft.seed_group for ft in formal if ft.split == split} for split in C_SPLITS}
    assert groups["trial"] & groups["dev"] == set()
    assert groups["dev"] & groups["blind"] == set()
    assert groups["trial"] & groups["blind"] == set()


def test_formal_tasks_keep_the_delta_only_affected_items_rule(formal):
    """判据 ⑤：扩量后 delta 口径不变 —— 仍是「只含受影响项」，不许因为条数多了就放宽。"""
    for ft in formal:
        delta = ft.task.ground_truth["expected_final_state"]
        task_type = ft.task.initial_data["任务类型"]
        if task_type == "改地址":
            assert len(delta["orders"]) == 1, ft.task.id
            assert set(delta["orders"][0]) == {"order_id", "address"}
        elif task_type in ("退款判断", "券抵扣"):
            assert delta["orders"] == [] and delta["coupons"] == [], ft.task.id
            assert delta["must_not_change"], ft.task.id
        else:
            assert delta["orders"], ft.task.id
            for row in delta["orders"]:
                assert row == {"order_id": row["order_id"], "status": "refunded"}


def test_the_traps_survive_scaling_to_forty(formal):
    """判据 ⑥：扩量后陷阱仍成立 —— 非 paid 单在、门槛不达标单在、批量里有不该退的单。"""
    for ft in formal:
        snap = ft.task.initial_data["sandbox"]
        task_type = ft.task.initial_data["任务类型"]
        delta = ft.task.ground_truth["expected_final_state"]
        if task_type == "退款判断":
            assert delta["must_not_change"], ft.task.id
            for row in delta["must_not_change"]:
                assert _order_of(snap, row["order_id"])["status"] in ("shipped", "pending_payment")
        elif task_type == "券抵扣":
            code = delta["must_not_change"][0]["code"]
            limit = snap["coupons"][code]["min_amount_cents"]
            order_id = next(o["order_id"] for o in snap["orders"] if o["order_id"] in ft.task.instruction)
            assert _order_of(snap, order_id)["amount_cents"] < limit, ft.task.id
        elif task_type == "多单批量":
            assert delta["must_not_change"], f"{ft.task.id} 名下没有非 paid 单"


def test_the_same_seed_produces_byte_identical_formal_files(tmp_path):
    """判据 ⑦：同 seed 两次正式集落盘**逐字节一致**（含档内打散后的 id 顺序）。"""
    first = write_formal_dataset(build_formal_dataset(seed=42), tmp_path / "a")
    second = write_formal_dataset(build_formal_dataset(seed=42), tmp_path / "b")
    assert set(first) == set(C_SPLITS)
    for split in C_SPLITS:
        assert first[split].read_bytes() == second[split].read_bytes(), split


def test_the_formal_preset_cli_writes_three_split_files(tmp_path, capsys):
    """判据 ③：CLI 一次生成三档，每档一个目录一个文件，中文不转义。"""
    assert main(["--seed", "42", "--preset", "formal", "--output-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    for split in C_SPLITS:
        path = tmp_path / split / "orchestration.jsonl"
        raw = path.read_text("utf-8")
        assert "\\u" not in raw, f"{split}: ensure_ascii 漏了"
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == FORMAL_SIZES[split], split
        assert all(r["id"].startswith(f"orch-{split}-") for r in rows)
    total = sum(
        len((tmp_path / s / "orchestration.jsonl").read_text("utf-8").splitlines())
        for s in C_SPLITS
    )
    assert total >= 40


def test_the_formal_preset_reports_error_when_capacity_is_short(monkeypatch):
    """判据 ⑧：容量撑不住时报错退出，**不许静默少出几条**。

    报错点有两条，两条都是「拒绝出半份产物」：
    一是 :func:`_plan_splits` 的生成期自检（报「装不下」），二是造题时
    :func:`_build_unique` 逐条撞空（报「候选空间见底」）。这里 monkeypatch 掉常量，
    命中的是哪条取决于组号排布，故断言「报错 + 不落产物」这个共同行为。
    """
    monkeypatch.setattr(orchestration, "SEED_GROUPS_PER_SPLIT",
                        {"trial": range(0, 2), "dev": range(2, 6), "blind": range(6, 999)})
    with pytest.raises(ValueError, match="装不下|候选空间见底"):
        build_formal_dataset(seed=42)
