"""C 场景地基：有状态的模拟电商环境（TASK-018）。

A 场景工具是纯函数（查表）、B 场景只读（检索），两者都测不出「多工具编排」。
C 场景要测的是 **agent 会不会按顺序把世界改对**：先查单再退款、先查库存再下单。
所以这里的工具**有状态、有副作用、有顺序依赖**，跑完拿 :func:`snapshot`
的最终状态判分，而不是拿某个工具的返回值判分。

契约（TASK-018 §2.5）：
- 金额一律整数分（沿用 A 场景口径，``int``）；日期 ISO 8601 字符串。
- 错误统一 ``{"error": {"code": ..., "message": ...}}``。
  **code 只有本模块新增的 ``unknown_id`` / ``invalid_state`` 两个**
  （``tools/_errors.py`` 的 ``ErrorCode`` 枚举在禁区内，改不得，所以此处自造同形结构，
  不自创第三个 code）。
- 确定性：同 seed 两次 :func:`make_sandbox` 的状态逐字节一致。
- :func:`tool_functions` 返回**绑定到该实例**的闭包，agent 拿不到 ``Sandbox`` 对象本身
  —— 拿得到就能直接 ``sb.orders.append(...)`` 绕过工具，副作用就测不到了。
- :func:`snapshot` 返回**深拷贝**，外部改它不影响 sandbox（判分器拿到快照后
  不该有办法反过来污染环境）。

本模块**只提供地基**：不写任务集、不写判分器（后续单）。
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable

#: 本模块新增的两个错误 code。取值固定，判分器靠它做机器可判定的分支。
UNKNOWN_ID = "unknown_id"        # 找不到订单 / 商品 / 优惠券
INVALID_STATE = "invalid_state"  # 当前状态不允许该操作（重复退款、对未支付订单退款…）


def _error(code: str, message: str) -> dict[str, Any]:
    """与 ``tools._errors.make_error`` 逐字段同形的错误结构。"""
    return {"error": {"code": code, "message": message}}


def _require(payload: Any, key: str) -> str | None:
    """取必填字符串字段；缺失或类型不对返回 ``None``（调用方转 invalid_input）。"""
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


# --------------------------------------------------------------------------------------
# 环境模型
# --------------------------------------------------------------------------------------

@dataclass
class Sandbox:
    """一个有状态的模拟电商环境。工具直接改它，跑完拿 snapshot 判分。"""

    orders: list[dict] = field(default_factory=list)
    inventory: dict[str, int] = field(default_factory=dict)
    coupons: dict[str, dict] = field(default_factory=dict)


_PRODUCTS = ("P-1001", "P-1002", "P-1003", "P-1004")
_CUSTOMERS = ("张伟", "李娜", "王强", "赵敏", "陈晨")
_CITIES = ("上海市浦东新区", "北京市海淀区", "杭州市西湖区", "广州市天河区", "成都市武侯区")
_STREETS = ("锦绣路 128 号 3 单元 502 室", "中关村大街 27 号 1802 室",
            "文三路 90 号 B 座 7 层", "天河北路 233 号 1201 室", "人民南路 45 号 2 栋 901 室")
_STATUSES = ("paid", "paid", "paid", "shipped", "pending_payment")

#: 订单状态取值。``refund`` 只接受 ``paid``。
STATUSES_ALLOWED: tuple[str, ...] = ("pending_payment", "paid", "shipped", "refunded")

_COUPONS = {
    "SAVE50": {"discount_cents": 5000, "min_amount_cents": 100000},
    "SAVE20": {"discount_cents": 2000, "min_amount_cents": 50000},
}


def make_sandbox(seed: int = 42) -> Sandbox:
    """确定性初始化。**只依赖 seed** —— 不读时钟、不读环境变量、不联网。

    同 seed 两次调用，:func:`snapshot` 的 JSON 逐字节一致；不同 seed 只改
    「具体是谁 / 多少钱 / 哪天」这类无关字段，工具的状态机行为不变。
    """
    rng = random.Random(seed)
    sb = Sandbox()
    sb.inventory = {pid: rng.randint(3, 40) for pid in _PRODUCTS}
    sb.coupons = {code: {**spec, "used": False} for code, spec in _COUPONS.items()}
    sb.orders = [
        {
            "order_id": f"SO-2026{index:04d}",
            "customer": rng.choice(_CUSTOMERS),
            "status": rng.choice(_STATUSES),
            "address": f"{rng.choice(_CITIES)}{rng.choice(_STREETS)}",
            "amount_cents": rng.randrange(5000, 900000, 100),
            "product_id": rng.choice(_PRODUCTS),
            "created_at": f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)}",
        }
        for index in range(1, 9)
    ]
    return sb


# --------------------------------------------------------------------------------------
# 内部查找
# --------------------------------------------------------------------------------------

def _find(sb: Sandbox, order_id: str) -> dict | None:
    return next((o for o in sb.orders if o["order_id"] == order_id), None)


# --------------------------------------------------------------------------------------
# 工具契约（input_schema）
# --------------------------------------------------------------------------------------

#: 6 个沙箱工具的 input_schema。**唯一真相源在这里** —— agents 层不许另写一份
#: （复述必然漂移，而漂移的 schema 会让模型传错参、agent 一直撞 unknown_id/invalid_input，
#: 症状离病因很远）。
#:
#: 每个 ``required`` 与 :func:`tool_functions` 里闭包实际 ``payload.get`` 的键**逐一对齐**：
#: 必填的进 required，可选的（只有 ``order.list.customer``）不进 —— 标进 required
#: 会诱导模型瞎填，反而拉低分数。
#: 有测试对着 :func:`tool_functions` 逐键反查（``test_tool_specs_required_matches_what_the_closures_read``），
#: 改 schema 忘了改闭包（或反过来）当场红。
TOOL_SPECS: dict[str, dict[str, Any]] = {
    "order.list": {
        "type": "object",
        "properties": {
            "customer": {"type": "string", "description": "客户名；不传则列出全部订单"},
        },
        "required": [],
    },
    "order.get": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "订单号，形如 SO-20260001"},
        },
        "required": ["order_id"],
    },
    "order.change_address": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "要改地址的订单号"},
            "new_address": {"type": "string", "description": "新的收货地址全文"},
        },
        "required": ["order_id", "new_address"],
    },
    "order.refund": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "要退款的订单号；仅 paid 状态可退"},
            "reason": {"type": "string", "description": "退款原因"},
        },
        "required": ["order_id", "reason"],
    },
    "inventory.check": {
        "type": "object",
        "properties": {
            "product_id": {"type": "string", "description": "商品号，形如 P-1001"},
        },
        "required": ["product_id"],
    },
    "coupon.use": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "优惠券码，形如 SAVE50"},
            "order_id": {"type": "string", "description": "要用券的订单号；金额需达券门槛"},
        },
        "required": ["code", "order_id"],
    },
}


def tool_descriptions() -> dict[str, str]:
    """工具 id → 中文一句话说明（function schema 的 ``description`` 用）。

    单独抽出来而不是让 :data:`TOOL_SPECS` 兼任：schema 是**给模型看的数据契约**，
    描述是**给人看的说明**，混在一个 dict 里会让「改说明」和「改契约」变成同一件事。
    """
    return {
        "order.list": "列订单，可按客户名过滤",
        "order.get": "按订单号取单条订单详情",
        "order.change_address": "修改某订单的收货地址",
        "order.refund": "对已支付订单退款（未支付/已发货/已退款会被拒）",
        "inventory.check": "查某商品的当前库存数量",
        "coupon.use": "在某订单上使用优惠券（金额需达券门槛）",
    }


# --------------------------------------------------------------------------------------
# 工具（全部 fn(payload: dict) -> dict）
# --------------------------------------------------------------------------------------

def tool_functions(sb: Sandbox) -> dict[str, Callable[[dict], dict]]:
    """返回绑定到 ``sb`` 的工具函数字典。

    闭包只闭包 ``sb`` 本身、闭包外不发对象出去，agent 因此**没有别的路**改世界 ——
    要改就必须走工具，副作用与顺序依赖才会真的被工具调用记录下来。
    """
    # 读工具
    def order_list(payload: Any = None) -> dict:
        """【读】列订单。``{"customer": "张伟"}`` 可选，按客户过滤。"""
        if not isinstance(payload, dict):
            return _error("invalid_input", "payload 必须是对象")
        customer = payload.get("customer")
        if customer is not None and not isinstance(customer, str):
            return _error("invalid_input", "customer 必须是字符串")
        rows = sb.orders
        if isinstance(customer, str) and customer.strip():
            rows = [o for o in rows if o["customer"] == customer.strip()]
        return {"orders": [dict(o) for o in rows]}

    def order_get(payload: Any) -> dict:
        """【读】按 order_id 取单条订单。找不到报 ``unknown_id``。"""
        order_id = _require(payload, "order_id")
        if order_id is None:
            return _error("invalid_input", "缺少必填字段 order_id")
        order = _find(sb, order_id)
        if order is None:
            return _error(UNKNOWN_ID, f"订单不存在：{order_id}")
        return {"order": dict(order)}

    def inventory_check(payload: Any) -> dict:
        """【读】查商品库存。找不到报 ``unknown_id``。"""
        product_id = _require(payload, "product_id")
        if product_id is None:
            return _error("invalid_input", "缺少必填字段 product_id")
        if product_id not in sb.inventory:
            return _error(UNKNOWN_ID, f"商品不存在：{product_id}")
        return {"product_id": product_id, "stock": sb.inventory[product_id]}

    # 写工具（副作用就地改 sb）
    def order_change_address(payload: Any) -> dict:
        """【写】改收货地址。副作用：改 ``order.address``，返回改后的订单。"""
        order_id = _require(payload, "order_id")
        new_address = _require(payload, "new_address")
        if order_id is None or new_address is None:
            return _error("invalid_input", "缺少必填字段 order_id / new_address")
        order = _find(sb, order_id)
        if order is None:
            return _error(UNKNOWN_ID, f"订单不存在：{order_id}")
        order["address"] = new_address
        return {"order": dict(order)}

    def order_refund(payload: Any) -> dict:
        """【写】退款。副作用：``status -> "refunded"``。

        只允许 ``status == "paid"``：已退款的报 ``invalid_state``（不能退两次，
        不能拿退款当收入），未支付 / 已发货的同样报 ``invalid_state``。
        """
        order_id = _require(payload, "order_id")
        reason = _require(payload, "reason")
        if order_id is None or reason is None:
            return _error("invalid_input", "缺少必填字段 order_id / reason")
        order = _find(sb, order_id)
        if order is None:
            return _error(UNKNOWN_ID, f"订单不存在：{order_id}")
        if order["status"] == "refunded":
            return _error(INVALID_STATE, f"订单已退款，不能重复退款：{order_id}")
        if order["status"] != "paid":
            return _error(
                INVALID_STATE,
                f"只有已支付订单可退款，当前状态 {order['status']}：{order_id}",
            )
        order["status"] = "refunded"
        return {"order": dict(order), "reason": reason}

    def coupon_use(payload: Any) -> dict:
        """【写】用券。副作用：``used -> True``。

        券不存在报 ``unknown_id``；已用过报 ``invalid_state``；订单金额未达门槛报
        ``invalid_state``（门槛不足是状态不允许，不是输入非法）。
        """
        code = _require(payload, "code")
        order_id = _require(payload, "order_id")
        if code is None or order_id is None:
            return _error("invalid_input", "缺少必填字段 code / order_id")
        coupon = sb.coupons.get(code)
        if coupon is None:
            return _error(UNKNOWN_ID, f"优惠券不存在：{code}")
        order = _find(sb, order_id)
        if order is None:
            return _error(UNKNOWN_ID, f"订单不存在：{order_id}")
        if coupon["used"]:
            return _error(INVALID_STATE, f"优惠券已使用：{code}")
        if order["amount_cents"] < coupon["min_amount_cents"]:
            return _error(
                INVALID_STATE,
                f"订单金额 {order['amount_cents']} 分未达券 {code} 的"
                f" {coupon['min_amount_cents']} 分门槛",
            )
        coupon["used"] = True
        return {"coupon": dict(coupon), "discount_cents": coupon["discount_cents"]}

    return {
        "order.list": order_list,
        "order.get": order_get,
        "order.change_address": order_change_address,
        "order.refund": order_refund,
        "inventory.check": inventory_check,
        "coupon.use": coupon_use,
    }


# --------------------------------------------------------------------------------------
# 最终状态（判分入口）
# --------------------------------------------------------------------------------------

def snapshot(sb: Sandbox) -> dict:
    """返回 sandbox 的**深拷贝**最终状态，供判分器比对。

    深拷贝不是为了「安全洁癖」：判分器通常会先排序 / 筛选 / 补字段再比，
    直接返回内部对象的话，判分器一个 ``snap["orders"].sort()`` 就把环境顺序改了 ——
    后面再判第二条任务时，环境已经不是 agent 走完的样子了。
    """
    return {
        "orders": copy.deepcopy(sb.orders),
        "inventory": dict(sb.inventory),
        "coupons": copy.deepcopy(sb.coupons),
    }
