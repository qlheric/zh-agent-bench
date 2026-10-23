"""sandbox 的离线测试：**全程不联网、不读时钟**（纯内存模拟环境）。

钉 TASK-018 §2.5 的全部要点：初始化确定性、两个读工具、两个写工具的副作用、
退款状态机、库存查询、``unknown_id``、金额/日期类型、闭包隔离、快照深拷贝。

关键点是**副作用那条**：写工具返回的订单必须与 sandbox 内部是**两个对象**
—— 若返回的是内部引用，agent 拿到返回值后自己改一笔就能污染环境，
而 snapshot 照样「看起来对」，这类 bug 测不出来。
"""

from __future__ import annotations

import copy
import json

import pytest

from zh_agent_bench.sandbox import (
    INVALID_STATE,
    STATUSES_ALLOWED,
    TOOL_SPECS,
    UNKNOWN_ID,
    Sandbox,
    make_sandbox,
    snapshot,
    tool_functions,
)


@pytest.fixture
def sb() -> Sandbox:
    return make_sandbox()


@pytest.fixture
def tools(sb):
    return tool_functions(sb)


def _paid_order(sb: Sandbox) -> dict:
    """把第一条订单强制置为 ``paid``，让退款路径有稳定入口。"""
    order = sb.orders[0]
    order["status"] = "paid"
    return order


# --------------------------------------------------------------------------------------
# 1. 初始化确定性
# --------------------------------------------------------------------------------------


def test_two_sandboxes_with_the_same_seed_are_byte_identical():
    """判分靠逐字节比对快照，环境不确定的话分数没法复现。"""
    a = json.dumps(snapshot(make_sandbox(42)), ensure_ascii=False, sort_keys=True)
    b = json.dumps(snapshot(make_sandbox(42)), ensure_ascii=False, sort_keys=True)
    assert a == b


def test_a_different_seed_changes_the_world_but_not_its_shape():
    """换 seed 只改「是谁/多少钱/哪天」，不该改环境的结构。"""
    other = snapshot(make_sandbox(7))
    mine = snapshot(make_sandbox(42))
    assert other != mine
    assert set(other["inventory"]) == set(mine["inventory"])
    assert [set(o) for o in other["orders"]] == [set(o) for o in mine["orders"]]
    assert set(other["coupons"]) == set(mine["coupons"])


def test_initial_state_is_well_formed(sb):
    assert len(sb.orders) == 8
    assert len({o["order_id"] for o in sb.orders}) == 8          # 订单号唯一
    assert all(o["status"] in STATUSES_ALLOWED for o in sb.orders)
    assert all(isinstance(s, int) and s >= 0 for s in sb.inventory.values())


# --------------------------------------------------------------------------------------
# 2. 读工具
# --------------------------------------------------------------------------------------


def test_order_list_returns_every_order(sb, tools):
    out = tools["order.list"]({})
    assert "error" not in out
    assert [o["order_id"] for o in out["orders"]] == [o["order_id"] for o in sb.orders]


def test_order_list_filters_by_customer(sb, tools):
    target = sb.orders[0]["customer"]
    out = tools["order.list"]({"customer": target})
    assert out["orders"], "样本里应当至少有一位客户命中"
    assert all(o["customer"] == target for o in out["orders"])
    # 空结果不是错误：客户确实没有单
    assert "error" not in tools["order.list"]({"customer": "查无此人"})


def test_order_get_returns_the_same_order_that_list_showed(sb, tools):
    target = sb.orders[2]
    out = tools["order.get"]({"order_id": target["order_id"]})
    assert out["order"] == target


def test_inventory_check_reports_the_live_stock(sb, tools):
    product_id = sorted(sb.inventory)[0]
    out = tools["inventory.check"]({"product_id": product_id})
    assert out == {"product_id": product_id, "stock": sb.inventory[product_id]}


# --------------------------------------------------------------------------------------
# 3. 金额与日期口径
# --------------------------------------------------------------------------------------


def test_amounts_are_integer_cents_and_dates_are_iso(sb):
    """沿用 A 场景口径：金额禁 float（会毁掉精确判分），日期 ISO 8601。"""
    snap = snapshot(sb)
    for order in snap["orders"]:
        assert type(order["amount_cents"]) is int      # type 而非 isinstance：bool 要挡掉
        assert len(order["created_at"]) == 10
        assert order["created_at"][4] == "-" and order["created_at"][7] == "-"
    for spec in snap["coupons"].values():
        assert type(spec["discount_cents"]) is int
        assert type(spec["min_amount_cents"]) is int


# --------------------------------------------------------------------------------------
# 4. 写工具：副作用
# --------------------------------------------------------------------------------------


def test_change_address_mutates_the_sandbox_and_shows_up_in_the_snapshot(sb, tools):
    target = sb.orders[0]
    out = tools["order.change_address"](
        {"order_id": target["order_id"], "new_address": "上海市黄浦区南京东路 1 号"}
    )
    assert "error" not in out
    assert out["order"]["address"] == "上海市黄浦区南京东路 1 号"
    assert target["address"] == "上海市黄浦区南京东路 1 号"        # 就地改了
    after = {o["order_id"]: o["address"] for o in snapshot(sb)["orders"]}
    assert after[target["order_id"]] == "上海市黄浦区南京东路 1 号"


def test_refund_flips_the_status_to_refunded(sb, tools):
    target = _paid_order(sb)
    out = tools["order.refund"]({"order_id": target["order_id"], "reason": "客户申请"})
    assert out["order"]["status"] == "refunded"
    after = {o["order_id"]: o["status"] for o in snapshot(sb)["orders"]}
    assert after[target["order_id"]] == "refunded"


def test_a_write_result_is_a_copy_not_a_live_reference(sb, tools):
    """返回值必须是**副本**。返回内部引用的话，agent 拿到回包再改一笔就污染了环境，
    而 snapshot 看着照样「对」—— 这是本单最容易漏、后果最隐蔽的一类漏洞。"""
    target = _paid_order(sb)
    refunded = tools["order.refund"]({"order_id": target["order_id"], "reason": "客户申请"})
    refunded["order"]["status"] = "paid"          # agent 拿着回包自己改回去
    refunded["order"]["amount_cents"] = 1
    assert target["status"] == "refunded"
    assert target["amount_cents"] != 1
    listed = tools["order.get"]({"order_id": target["order_id"]})["order"]
    listed["address"] = "篡改"
    assert target["address"] != "篡改"


def test_coupon_use_marks_the_coupon_and_needs_the_minimum_amount(sb, tools):
    code, spec = sorted(sb.coupons.items())[0]
    order = max(sb.orders, key=lambda o: o["amount_cents"])
    order["amount_cents"] = spec["min_amount_cents"]     # 正好卡在门槛上（≥ 通过）
    out = tools["coupon.use"]({"code": code, "order_id": order["order_id"]})
    assert out["discount_cents"] == spec["discount_cents"]
    assert snapshot(sb)["coupons"][code]["used"] is True


# --------------------------------------------------------------------------------------
# 5. 退款状态机
# --------------------------------------------------------------------------------------


def test_refunding_the_same_order_twice_is_invalid_state(sb, tools):
    target = _paid_order(sb)
    first = tools["order.refund"]({"order_id": target["order_id"], "reason": "客户申请"})
    assert first["order"]["status"] == "refunded"
    second = tools["order.refund"]({"order_id": target["order_id"], "reason": "再退一次"})
    assert second["error"]["code"] == INVALID_STATE
    assert "重复" in second["error"]["message"]


def test_refusing_a_refund_must_leave_the_world_untouched(sb, tools):
    """被拒的写操作是**不该有任何副作用**的 —— 这条最容易被实现漏掉。"""
    target = _paid_order(sb)
    assert "error" not in tools["order.refund"](
        {"order_id": target["order_id"], "reason": "客户申请"}
    )
    before = json.dumps(snapshot(sb), ensure_ascii=False, sort_keys=True)
    tools["order.refund"]({"order_id": target["order_id"], "reason": "再退一次"})
    after = json.dumps(snapshot(sb), ensure_ascii=False, sort_keys=True)
    assert after == before          # 第二次退款没把 refunded 又写成别的


@pytest.mark.parametrize("status", ["pending_payment", "shipped"])
def test_refunding_an_unpaid_or_shipped_order_is_invalid_state(sb, tools, status):
    target = sb.orders[0]
    target["status"] = status
    out = tools["order.refund"]({"order_id": target["order_id"], "reason": "客户申请"})
    assert out["error"]["code"] == INVALID_STATE
    assert status in out["error"]["message"]
    assert target["status"] == status      # 没被改成 refunded


# --------------------------------------------------------------------------------------
# 6. unknown_id / invalid_input
# --------------------------------------------------------------------------------------


def test_unknown_ids_are_reported_as_unknown_id(tools):
    assert tools["order.get"]({"order_id": "SO-没有这个"})["error"]["code"] == UNKNOWN_ID
    assert tools["inventory.check"]({"product_id": "P-9999"})["error"]["code"] == UNKNOWN_ID
    assert tools["order.change_address"](
        {"order_id": "SO-没有这个", "new_address": "某地"}
    )["error"]["code"] == UNKNOWN_ID
    assert tools["order.refund"](
        {"order_id": "SO-没有这个", "reason": "客户申请"}
    )["error"]["code"] == UNKNOWN_ID


def test_missing_or_blank_fields_are_invalid_input(tools):
    assert tools["order.get"]({})["error"]["code"] == "invalid_input"
    assert tools["order.get"]({"order_id": "   "})["error"]["code"] == "invalid_input"
    assert tools["inventory.check"]({"product_id": 1001})["error"]["code"] == "invalid_input"
    assert tools["order.change_address"](
        {"order_id": "SO-20260001", "new_address": ""}
    )["error"]["code"] == "invalid_input"


def test_an_unknown_id_is_not_also_an_invalid_state(tools):
    """两个 code 各管一段，混用会让判分器的分支判错。"""
    assert tools["order.refund"]({"order_id": "SO-没有这个", "reason": "x"})[
        "error"]["code"] == UNKNOWN_ID


# --------------------------------------------------------------------------------------
# 7. 闭包隔离与快照深拷贝
# --------------------------------------------------------------------------------------


def test_two_sandboxes_do_not_share_state():
    """两个 sandbox 各自的工具互不可见 —— 否则跑第二条任务时第一条已污染环境，
    分数会随执行顺序变，同一模型两次跑出两个分。"""
    a, b = make_sandbox(), make_sandbox()
    tools_a, tools_b = tool_functions(a), tool_functions(b)
    target = a.orders[0]
    tools_a["order.change_address"](
        {"order_id": target["order_id"], "new_address": "只属于 a 的地址"}
    )
    tools_a["order.refund"]({"order_id": target["order_id"], "reason": "a 退的"})

    assert tools_b["order.get"]({"order_id": target["order_id"]})["order"]["address"] != \
        "只属于 a 的地址"
    assert b.orders[0]["address"] != "只属于 a 的地址"
    assert snapshot(b) == snapshot(make_sandbox())


def test_tools_do_not_hand_out_the_sandbox_itself(sb, tools):
    """工具只回结构化结果，不回环境对象 —— 回环境就能绕开工具直接改，
    副作用与顺序依赖就都测不到了。"""
    returned = tools["order.list"]({})["orders"][0]
    assert isinstance(returned, dict)
    assert not isinstance(returned, Sandbox)
    assert "inventory" not in returned and "coupons" not in returned


def test_snapshot_is_a_deep_copy(sb):
    """判分器常会先排序 / 补字段再比；直接返回内部对象的话，判分器一个
    ``sort()`` 就把环境顺序改了，后面再判第二条时环境已不是 agent 走完的样子。"""
    snap = snapshot(sb)
    snap["orders"][0]["status"] = "refunded"
    snap["orders"].reverse()
    snap["inventory"]["P-1001"] = -1
    snap["coupons"].popitem()
    assert sb.orders[0]["status"] != "refunded"
    assert [o["order_id"] for o in sb.orders] == \
        [o["order_id"] for o in snapshot(sb)["orders"]]
    assert snapshot(sb)["inventory"]["P-1001"] != -1
    assert len(sb.coupons) == len(snapshot(make_sandbox())["coupons"])


def test_snapshot_owns_nothing_shared_with_the_environment():
    """比 ``test_snapshot_is_a_deep_copy`` 更狠一层：连「值相同但不是同一份」都钉住。"""
    sb = make_sandbox()
    snap = snapshot(sb)
    assert snap["orders"] is not sb.orders
    assert snap["orders"][0] is not sb.orders[0]
    assert snap["inventory"] is not sb.inventory
    assert snap["coupons"] is not sb.coupons
    assert snap == copy.deepcopy(snapshot(make_sandbox(42)))


# --------------------------------------------------------------------------------------
# TASK-021：TOOL_SPECS 与闭包实读键对齐
# --------------------------------------------------------------------------------------


def test_tool_specs_cover_exactly_the_six_sandbox_tools():
    """specs 与 tool_functions 必须同一套工具 —— 多一个少一个都是雷。"""
    from zh_agent_bench.sandbox import TOOL_SPECS, tool_descriptions

    assert set(TOOL_SPECS) == set(tool_functions(make_sandbox()))
    assert set(TOOL_SPECS) == {
        "order.list", "order.get", "order.change_address",
        "order.refund", "inventory.check", "coupon.use",
    }
    assert set(tool_descriptions()) == set(TOOL_SPECS)


def test_every_spec_is_a_legal_openai_object_schema():
    """发出去的 schema 不合法会被 provider 直接拒收（HTTP 400），不会告诉你哪一条错。"""
    legal = {"object", "array", "string", "number", "integer", "boolean", "null"}
    for tool_id, spec in TOOL_SPECS.items():
        assert spec["type"] == "object", tool_id
        assert isinstance(spec["properties"], dict), tool_id
        assert isinstance(spec["required"], list), tool_id
        # required 里的键必须在 properties 里，否则模型无从构造这个参数
        assert set(spec["required"]) <= set(spec["properties"]), tool_id
        for name, sub in spec["properties"].items():
            assert sub["type"] in legal, f"{tool_id}.{name} 的 type 非法"
            assert isinstance(sub.get("description", ""), str)


def test_tool_specs_required_matches_what_the_closures_read():
    """**specs 的 required 必须与 tool_functions 闭包实际 payload.get 的键一致。**

    这是本单最该被钉住的一条：schema 说必填、闭包不读（或反过来），症状是模型传了参数
    却被报 invalid_input / 少了参数却一路通过 —— 离病因很远极难查。
    做法是黑盒反查：对每个 required 键去掉它，闭包必须报 invalid_input。
    """
    sb = make_sandbox()
    tools = tool_functions(sb)
    # 造一个「全部必填键都齐全」的可调用 payload；逐个删键后必须报 invalid_input。
    payloads = {
        "order.list": {"customer": "张伟"},                       # 无必填
        "order.get": {"order_id": "SO-20260001"},
        "order.change_address": {"order_id": "SO-20260001", "new_address": "新地址 1 号"},
        "order.refund": {"order_id": "SO-20260001", "reason": "不想要了"},
        "inventory.check": {"product_id": "P-1001"},
        "coupon.use": {"code": "SAVE50", "order_id": "SO-20260001"},
    }
    assert set(payloads) == set(TOOL_SPECS)

    for tool_id, spec in TOOL_SPECS.items():
        full = payloads[tool_id]
        for key in spec["required"]:
            partial = {k: v for k, v in full.items() if k != key}
            result = tools[tool_id](partial)
            assert result.get("error", {}).get("code") == "invalid_input", (
                f"{tool_id} 声明 {key} 必填，但去掉它闭包不报错 —— schemas 与实现漂了"
            )
        # 补一句：非 required 的键去掉不该致命（否则就是隐式必填）
        for key in set(full) - set(spec["required"]):
            reduced = {k: v for k, v in full.items() if k != key}
            result = tools[tool_id](reduced)
            assert result.get("error", {}).get("code") != "invalid_input", (
                f"{tool_id} 的 {key} 没声明 required，却被闭包当必填 —— schema 说可选、"
                f"实现说必填，模型会因此一直重试"
            )


def test_order_list_customer_is_optional_so_no_required_entry():
    """order.list 的 customer 是可选过滤条件；写进 required 会诱导模型瞎填。"""
    assert TOOL_SPECS["order.list"]["required"] == []
    assert "customer" in TOOL_SPECS["order.list"]["properties"]
    # 不传 customer 必须能列全部，而不是报 invalid_input
    assert "orders" in tool_functions(make_sandbox())["order.list"]({})
