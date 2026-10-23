"""C 场景任务构造器：多工具编排（TASK-019；TASK-023 起补正式规模 + 三 split）。

A 场景工具是纯函数、B 场景只读检索 —— 两者都测不出「会不会按顺序调多个有状态的工具」。
C 场景跑在 TASK-018 的有状态沙箱上，任务必须**多步骤 + 有陷阱**：
「先 get 看清状态再决定退不退」「先查门槛再决定用不用券」，
否则 agent 不查直接调用，判分器也照样给分 —— 那测的是胆量，不是编排。

**正式规模与三 split（TASK-023）**：试标阶段的 8 条只够验通路。正式集按
:func:`build_formal_dataset` 生成 **40 条（每类 10 条）**，切成
``trial / dev / blind`` 三档，切分单位是**环境 seed 分组**（同一 seed 造出的任务
只能落在同一档）—— 与 B 场景「同一 ``base_id`` 不跨 split」同一口径，防开发期
见过答案的题在盲测期原样再考一次。

契约（TASK-019 §2.5）：
- ``initial_data = {"sandbox": snapshot(initial_sandbox), "任务类型": ...}`` ——
  沙箱初始快照与任务里提到的订单 id **来自同一份环境**（同一 seed 造一次），
  不然 agent 拿到一份环境、按 id 找过去却找不到单。
- ``ground_truth = {"expected_final_state": <delta>, "example_answer": ...}`` ——
  delta **只含受影响的那几单 + 那张券**（findings §十七.3），不含未受影响项。
  判分器比对 delta 而不是全量快照，正是为了不把「agent 顺手做了件对的事」判成错。
- ``answer_spec = {kind: json, comparison: orchestration}``。
- rubric 沿用 :mod:`tasks.qa` 的七字段规范（编排点 + 回复点）。

**本模块只出题**：不判分（TASK-020）、不接 harness、不跑分。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..sandbox import Sandbox, make_sandbox, snapshot
from .schema import (
    C_SCENE_COMPARISON,
    SCHEMA_VERSION,
    AnswerKind,
    AnswerSpec,
    Category,
    Normalization,
    Split,
    Task,
)
from .qa import RUBRIC_FIELDS, partial_rule_kind

# --------------------------------------------------------------------------------------
# 契约常量
# --------------------------------------------------------------------------------------

#: rubric 七个必填字段（与 B 场景同一套规范，顺序即 jsonl 字段序，稳定输出靠它）。
#: 从 :mod:`tasks.qa` 导入而不是抄一份 —— 抄第二份必然漂，漂了之后
#: 「B、C 两场景 rubric 规范一致」这句话就是假的。

#: 四种编排任务类型，固定顺序 —— 任务 id 与产物行序都按它。
TASK_TYPES: tuple[str, ...] = ("改地址", "退款判断", "券抵扣", "多单批量")

#: 类型名的 ASCII 别名（stdout 汇总表用；Windows 控制台未必渲染中文，同 qa.py 的理由）。
TYPE_LABELS: dict[str, str] = {
    "改地址": "address",
    "退款判断": "refund-judge",
    "券抵扣": "coupon",
    "多单批量": "batch-refund",
}

#: 每条任务给的工具集。按类型裁剪：给全套等于让 agent 靠试错撞对，
#: 而「该用哪几个工具」本身就是编排能力的一部分（findings：工具契约要求按需暴露）。
ALL_TOOLS: tuple[str, ...] = (
    "order.list", "order.get", "order.change_address",
    "order.refund", "inventory.check", "coupon.use",
)


def _p(requirement: str, weight: float, evidence: list[str], partial: str,
       alts: list[str] | None = None, critical: bool = False) -> dict[str, Any]:
    """一个评分点。``criterion_id`` 由 :func:`_materialize` 按位置补上，不手写。"""
    return {
        "requirement": requirement,
        "weight": weight,
        "evidence_ids": evidence,
        "partial_rule": partial,
        "accepted_alternatives": alts or [],
        "critical": critical,
    }


class NoTrapConstructible(ValueError):
    """这个环境里造不出该类型的**陷阱**（差一个非 paid 单 / 一张够得着的券…）。

    与其它 ``ValueError`` 分开，是因为它可以安全地换 seed 重试 —— 而 rubric 校验失败、
    delta 非法那些错误换 seed 也救不了，必须炸出来给人看。
    """


@dataclass(frozen=True)
class Spec:
    """一条 C 场景任务要落的全部内容。

    ``initial``/``expected`` 是**已经算好的**快照与 delta —— 构造器自己跑一遍工具
    把世界改到目标态，再从前后快照里取差异。自己跑一遍而不是手写期望值：
    手写的期望值与实现不一致时，出题器和判分器会一起错，而测试全绿。
    """

    task_type: str
    instruction: str
    initial: dict[str, Any]                 # snapshot(initial_sandbox)
    expected: dict[str, Any]                # delta：只含受影响的订单 / 券
    example_answer: str
    points: tuple[dict[str, Any], ...]
    tools: tuple[str, ...]


# --------------------------------------------------------------------------------------
# 场景构造：每类从沙箱里挑出具体的订单 / 券
# --------------------------------------------------------------------------------------

def _order(sb: Sandbox, order_id: str) -> dict:
    return next(o for o in sb.orders if o["order_id"] == order_id)


#: 改地址任务的新地址池。25 个两两不同的地址，够撑起每类 10~20 条正式集。
#:
#: **为什么不用沙箱自己的 :data:`sandbox._CITIES` × :data:`sandbox._STREETS` 自由组合**：
#: 那 5×5=25 种组合里有几种会与沙箱某单的原地址逐字相同，抽到就得顺延，
#: 而「顺延」让 :func:`capacity` 算不出一个确定的数。写死一份不含任何沙箱原地址的
#: 池子，容量就是 ``len(_ADDRESS_SLOTS)`` 这个常量，不用跑一遍才知道。
_ADDRESS_SLOTS: tuple[str, ...] = tuple(
    f"{city}{street}（{floor} 层）"
    for city, floor in (
        ("上海市黄浦区", "56"), ("北京市朝阳区", "33"), ("杭州市上城区", "18"),
        ("广州市越秀区", "42"), ("成都市锦江区", "27"),
    )
    for street in (
        "南京东路 100 号 环球金融中心", "建国门外大街 1 号 国贸写字楼",
        "延安路 8 号 银泰百货", "天河路 208 号 粤海天河城",
        "红星路三段 1 号 IFS 国际金融中心",
    )
)


def _build_address(sb: Sandbox, rng: random.Random, index: int) -> Spec:
    """改地址：``order.get`` → ``order.change_address``，1 步写。

    **候选按 index 取，不用 ``rng.choice``**：随机挑在扩量后必然撞重复
    （实测每类 10 条里 3 条是「同一个订单、改成同一个地址」—— 同一份答案被算
    三次，一致率虚高）。改成「先枚举候选、再按 index 取」后，重复在结构上不可能，
    容量也变成一个**可算的数**（见 :func:`capacity`）。

    **候选是 ``(订单, 新地址)`` 的组合，不是单个订单**：只按订单取的话上限是 8
    （沙箱固定 8 单），撑不起「每类 10 条」；按组合取则是 ``8 × 24 = 192``。
    同一单不同地址给出的是不同的题（不同题面、不同 delta），算重复的话
    「改地址」就只剩一单一条，A 场景那种「同一模板改个数」也认不出它。
    """
    targets = sorted(sb.orders, key=lambda o: o["order_id"])
    pairs = [
        (order, addr)
        for order in targets
        for addr in _ADDRESS_SLOTS
        if addr != order["address"]      # 新地址必须与原地址不同，否则「改了等于没改」
    ]
    if not pairs:
        raise NoTrapConstructible("沙箱里没有「订单 × 新地址」的合法组合")
    target, new_address = pairs[index % len(pairs)]
    oid = target["order_id"]
    expected = {
        "orders": [{"order_id": oid, "address": new_address}],
        "coupons": [],
    }
    return Spec(
        task_type="改地址",
        instruction=f"订单 {oid} 的收货人要求把收货地址改为「{new_address}」，请帮他改好。",
        initial=snapshot(sb),
        expected=expected,
        example_answer=f"已为订单 {oid} 将收货地址修改为「{new_address}」，请注意查收。",
        points=(
            _p(f"调用 order.get 查到了订单 {oid}", 0.3, [oid],
               "查到了该订单并确认存在 → 1；直接改地址没先查 → 0.5；查错订单 → 0"),
            _p(f"调用 order.change_address 把 {oid} 的地址改成指定的新地址", 0.4, [oid],
               "地址与指定完全一致 → 1；地址大致对但有出入 → 0.5；没改 → 0", critical=True),
            _p("回复中确认改单成功，并复述了新地址", 0.3, [oid],
               "确认成功且地址复述准确 → 1；只说改好了不复述 → 0.5；没说成功 → 0"),
        ),
        tools=("order.get", "order.change_address", "order.list"),
    )


def _build_refund_judge(sb: Sandbox, rng: random.Random, index: int) -> Spec:
    """退款判断：``order.get`` 看状态 → 只有 ``paid`` 能退。**陷阱**：非 paid 单。

    delta 里 orders 是**空列表** —— 正确答案就是「什么都没改」，该保持原样的那单
    写在 ``must_not_change`` 里。这是四类里最能筛出「不查就调」agent 的一类：
    直接调 refund 会被沙箱拒绝（``invalid_state``），世界没被改坏，但编排点给 0。

    候选按 index 取（同 :func:`_build_address` 的理由）：随机挑在扩量后必然撞重复。
    """
    blocked = sorted(
        (o for o in sb.orders if o["status"] in ("shipped", "pending_payment")),
        key=lambda o: o["order_id"],
    )
    if not blocked:
        raise NoTrapConstructible("没有 shipped/pending_payment 订单，退款陷阱造不出来")
    target = blocked[index % len(blocked)]
    oid = target["order_id"]
    return Spec(
        task_type="退款判断",
        instruction=f"客户 {target['customer']} 要求给订单 {oid} 办理退款，请帮他处理。",
        initial=snapshot(sb),
        # 陷阱任务的正确答案是「什么都没改」—— 但 delta 留空会让判分器无从下手：
        # 「该保持原状」必须写成机器能比的断言，否则「没退」和「退错了又被改回来」同分。
        expected={"orders": [], "coupons": [],
                  "must_not_change": [{"order_id": oid, "status": target["status"]}]},
        example_answer=(
            f"订单 {oid} 当前状态为 {target['status']}，只有已支付订单才能办理退款，"
            f"因此这笔没有退款。订单状态未做任何变更。"
        ),
        points=(
            _p(f"调用 order.get 查到了订单 {oid} 并确认其状态为 {target['status']}",
               0.35, [oid],
               "查了且状态说对 → 1；查了但没在回复里说状态 → 0.5；没查 → 0", critical=True),
            _p("识别出该订单不可退款（未支付 / 已发货），没有强行调用退款",
               0.35, [oid],
               "明确不退款且未改动订单 → 1；没退款但没说原因 → 0.5；调用了退款 → 0"),
            _p("回复中告知客户不能退款及具体原因", 0.3, [oid],
               "说清不能退且点明原因 → 1；只说不能退 → 0.5；没说不能退 → 0"),
        ),
        tools=("order.get", "order.refund", "order.list"),
    )


def _build_coupon(sb: Sandbox, rng: random.Random, index: int) -> Spec:
    """券抵扣：门槛/未用先查清 → ``coupon.use``。**陷阱**：门槛不达标的单。

    只在「本环境确实造得出低于门槛的订单」的券里挑：``SAVE20`` 门槛 50000 分，
    某些 seed 下 8 单里最小的一单都高于它 —— 盲挑就会挑到造不出陷阱的券。

    券与订单都按 index 取（候选按 (券码, 订单号) 排序后展平）：随机挑是这一类
    重复最严重的一类 —— 沙箱只有 ``SAVE50`` / ``SAVE20`` 两张券，盲挑 10 次会有
    3 次落回同一张券同一笔小额单，同一份答案被算三次。
    """
    def below(code: str) -> list[dict]:
        """金额低于该券门槛的订单，按订单号排序（排序即确定性）。"""
        limit = sb.coupons[code]["min_amount_cents"]
        return sorted(
            (o for o in sb.orders if o["amount_cents"] < limit),
            key=lambda o: o["order_id"],
        )

    usable = sorted(c for c in sb.coupons if below(c))
    if not usable:
        raise NoTrapConstructible(
            f"没有任何一张券能造出「未达门槛」的订单"
            f"（最低订单 {min(o['amount_cents'] for o in sb.orders)} 分 vs 最低门槛 "
            f"{min(c['min_amount_cents'] for c in sb.coupons.values())} 分）"
        )
    pairs = [(c, o) for c in usable for o in below(c)]
    code, target = pairs[index % len(pairs)]
    spec_coupon = sb.coupons[code]
    oid = target["order_id"]
    return Spec(
        task_type="券抵扣",
        instruction=f"客户 {target['customer']} 想在订单 {oid} 上使用优惠券 {code}，请帮他用上。",
        initial=snapshot(sb),
        # 与「退款判断」同理：正确答案就是「什么都没改」，但 delta 不能真留空 ——
        # must_not_change 把「券仍是未使用」写成机器能比的断言，判分器才判得了「没乱用券」。
        expected={"orders": [], "coupons": [],
                  "must_not_change": [{"code": code, "used": False}]},
        example_answer=(
            f"订单 {oid} 的金额为 {target['amount_cents']} 分，未达到优惠券 {code} 所需的"
            f" {spec_coupon['min_amount_cents']} 分门槛，因此无法使用该券，优惠券保持未使用。"
        ),
        points=(
            _p(f"查到订单 {oid} 的金额并与优惠券 {code} 的 {spec_coupon['min_amount_cents']} 分门槛比对",
               0.35, [oid, code],
               "两边数字都对上 → 1；只报了其中一边 → 0.5；都没查 → 0", critical=True),
            _p(f"判断出订单未达门槛，没有调用 coupon.use 把券用掉",
               0.35, [code],
               "明确没用券且券仍为未使用 → 1；没用券但没说原因 → 0.5；把券用掉了 → 0"),
            _p("回复中告知客户该券用不了及未达门槛的原因", 0.3, [oid, code],
               "说清用不了并点明门槛 → 1；只说用不了 → 0.5；没说 → 0"),
        ),
        tools=("order.get", "inventory.check", "coupon.use"),
    )


def _build_batch_refund(sb: Sandbox, rng: random.Random, index: int) -> Spec:
    """多单批量：``order.list`` 按客户筛 → 逐单 ``refund``，只有 paid 单能退。

    这类是**唯一真正测多步写**的：N 步 get + M 步 refund，且必须漏掉非 paid 单。

    合格客户按排序后取数：``max`` 只挑「paid 单最多」的那一个客户，扩量后 10 条里
    会反复是同一个人名 —— 同一份答案被算多次。改成**对所有合格客户排序后按 index 取**，
    容量变成可算的数（见 :func:`capacity`）。
    """
    paid = [o for o in sb.orders if o["status"] == "paid"]
    if not paid:
        raise NoTrapConstructible("沙箱里没有 paid 订单，批量退款造不出来")
    # 合格客户 = 「名下同时有 paid 与非 paid 单」：多退几笔才叫批量，而只有同时有
    # 非 paid 单，「不该退的别退」这一半才判得出来。
    eligible = [
        c for c in sorted({o["customer"] for o in sb.orders})
        if len({o["status"] == "paid" for o in sb.orders if o["customer"] == c}) > 1
    ]
    if not eligible:
        raise NoTrapConstructible(
            "没有「同时有 paid 与非 paid 单」的客户，「不该退的别退」判不出来"
        )
    customer = eligible[index % len(eligible)]
    mine = [o for o in sb.orders if o["customer"] == customer]
    to_refund = sorted(o["order_id"] for o in mine if o["status"] == "paid")
    keep = sorted(o["order_id"] for o in mine if o["status"] != "paid")
    return Spec(
        task_type="多单批量",
        instruction=f"客户 {customer} 要求把名下所有已支付的订单全部退款，请你批量处理。",
        initial=snapshot(sb),
        # delta 里 keep 那几单**明确列出目标状态** —— 判分器据此判「不该退的没退」。
        # 不列的话「没动」和「动了但没列」在 delta 里长得一样，判分器无从区分。
        expected={
            "orders": [{"order_id": oid, "status": "refunded"} for oid in to_refund],
            "coupons": [],
            "must_not_change": [{"order_id": o["order_id"], "status": o["status"]}
                                for o in mine if o["status"] != "paid"],
        },
        example_answer=(
            f"客户 {customer} 名下共 {len(mine)} 笔订单，其中已支付 {len(to_refund)} 笔"
            f"（{'、'.join(to_refund)}）已全部退款；另有 {len(keep)} 笔"
            f"（{'、'.join(keep)}）当前不是已支付状态，按规则未予退款。"
        ),
        points=(
            _p(f"调用 order.list 按客户 {customer} 筛出其名下全部订单，而不是逐单盲查",
               0.25, to_refund + keep,
               "用 order.list 按客户筛 → 1；逐单 get 也能查全 → 0.5；没查全 → 0"),
            _p("逐单确认状态后，只对已支付订单调用 refund", 0.35, to_refund,
               "全部该退的退了且没多退 → 1；少退了一笔 → 0.5；多退了不该退的 → 0", critical=True),
            _p("对不该退的订单保持原状态、未调用退款", 0.2, [o for o in keep],
               "非已支付的都没动 → 1；动了一笔 → 0"),
            _p("回复中分别列出已退款与未退款的订单号并说明原因", 0.2, to_refund + keep,
               "两类都列全且说清原因 → 1；只列了一类 → 0.5；都没列 → 0"),
        ),
        tools=("order.list", "order.get", "order.refund"),
    )


_BUILDERS = {
    "改地址": _build_address,
    "退款判断": _build_refund_judge,
    "券抵扣": _build_coupon,
    "多单批量": _build_batch_refund,
}


# --------------------------------------------------------------------------------------
# 实例化
# --------------------------------------------------------------------------------------

def _materialize(points: Sequence[dict[str, Any]], task_id: str) -> tuple[dict[str, Any], ...]:
    """按位置补上 ``criterion_id``，其余字段原样透传。"""
    return tuple(
        {"criterion_id": f"{task_id}-c{i}",
         **{k: p[k] for k in RUBRIC_FIELDS if k != "criterion_id"}}
        for i, p in enumerate(points, start=1)
    )


def _validate_rubric(rubric: Sequence[dict[str, Any]], sandbox_snapshot: dict[str, Any]) -> None:
    """结构校验。不通过就抛 —— 出题期就该炸，不该留到判分时才发现 rubric 不可判。

    与 B 场景的 ``validate_rubric`` 同一套七字段规范，但 ``evidence_ids`` 的
    「存在性」查的是**沙箱里的订单号 / 券码 / 商品号**（不是知识库 doc_id）——
    C 场景的证据是环境对象本身。
    """
    for i, point in enumerate(rubric, start=1):
        missing = [f for f in RUBRIC_FIELDS if f not in point]
        if missing:
            raise ValueError(f"第 {i} 个评分点缺字段: {missing}")
        cid = point["criterion_id"]
        if not isinstance(point["requirement"], str) or not point["requirement"].strip():
            raise ValueError(f"{cid}: requirement 必须是非空字符串")
        if not isinstance(point["weight"], float) or not 0 < point["weight"] <= 1:
            raise ValueError(f"{cid}: weight 必须是 (0, 1] 的 float")
        if not isinstance(point["critical"], bool):
            raise ValueError(f"{cid}: critical 必须是 bool")
        if partial_rule_kind(point["partial_rule"]) == "undecidable":
            raise ValueError(
                f"{cid}: partial_rule 必须写明 0.5 的触发情形，或显式宣告本点为二值（→ 0）"
            )
        unknown = [x for x in point["evidence_ids"] if not _exists(x, sandbox_snapshot)]
        if unknown:
            raise ValueError(f"{cid}: evidence_ids 指向沙箱中不存在的对象: {unknown}")

    total = sum(p["weight"] for p in rubric)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"权重之和应为 1.0，实际 {total!r}")


def _exists(token: str, sandbox_snapshot: dict[str, Any]) -> bool:
    """订单号 / 券码 / 商品号是否出现在沙箱快照里。"""
    if token in sandbox_snapshot["inventory"] or token in sandbox_snapshot["coupons"]:
        return True
    return any(o["order_id"] == token for o in sandbox_snapshot["orders"])


def build_task(task_id: str, spec: Spec, split: Split, stage: str = "") -> Task:
    """把一条 :class:`Spec` 实例化成 :class:`Task`（并就地校验 rubric 与 delta）。

    :param stage: 三级 split 名（``trial / dev / blind``）。``schema.Split`` 只有
        ``TRAIN / HELDOUT`` 两值，改不了（禁区），所以三级 split 名写进
        ``initial_data["评测档位"]`` —— 与 B 场景同一口径：不占 schema 的两值枚举，
        也不会让人误以为 blind 就是 ``HELDOUT``。
    """
    initial = spec.initial
    for section in ("orders", "inventory", "coupons"):
        if section not in initial:
            raise ValueError(f"initial_data.sandbox 缺 {section} 段")
    rubric = _materialize(spec.points, task_id)
    _validate_rubric(rubric, initial)
    _validate_delta(spec.expected, initial)
    initial_data: dict[str, Any] = {"sandbox": initial, "任务类型": spec.task_type}
    if stage:
        initial_data["评测档位"] = stage
    return Task(
        id=task_id,
        category=Category.ORCHESTRATION,
        schema_version=SCHEMA_VERSION,
        instruction=spec.instruction,
        initial_data=initial_data,
        tool_names=spec.tools,
        ground_truth={
            "expected_final_state": spec.expected,
            "example_answer": spec.example_answer,
        },
        answer_spec=AnswerSpec(
            kind=AnswerKind.JSON,
            comparison=C_SCENE_COMPARISON,
            normalization=Normalization.NONE,
        ),
        split=split,
        rubric=rubric,
    )


def _validate_delta(expected: dict[str, Any], initial: dict[str, Any]) -> None:
    """delta 必须**只含受影响项**（findings §十七.3），且引用的对象必须真实存在。

    这条校验是判分口径的地基：delta 里混进未受影响项，判分器就会拿它去比对
    agent 没被要求改的字段，「顺手做了件对的事」反而被判成错。

    三段的口径（TASK-020 判分器照这读）：
    - ``orders`` / ``coupons``：**该改成的目标状态**，逐项判「改对了没」。
    - ``must_not_change``：**该保持原样的对象及其目标状态**。陷阱任务
      （退款判断 / 券抵扣）的正确答案就是「什么都没改」，若 delta 全空，判分器
      无法区分「没乱动」和「动过又改回来了」—— 所以「别动」必须写成可比的断言。
    - 三段全空 = 无效 delta：判分器既无目标态也无不变式，出题人自己没想清楚要什么。
    """
    known = {o["order_id"] for o in initial["orders"]}
    known |= set(initial["inventory"]) | set(initial["coupons"])
    for key in ("orders", "coupons", "must_not_change"):
        rows = expected.get(key, [])
        if not isinstance(rows, list):
            raise ValueError(f"expected_final_state.{key} 必须是列表，实际 {type(rows).__name__}")
        for row in rows:
            unknown = sorted(set(row) - {"order_id", "code", "address", "status", "used"})
            if unknown:
                raise ValueError(f"delta 行含未定义字段: {unknown}")
            token = row.get("order_id") or row.get("code")
            if token not in known:
                raise ValueError(f"delta 指向沙箱中不存在的对象: {token}")
    if not any(expected.get(k) for k in ("orders", "coupons", "must_not_change")):
        raise ValueError(
            "delta 为空：既无应改项也无 must_not_change，判分器无从判对错。"
            "陷阱任务请用 must_not_change 写明「哪些东西必须保持原样」"
        )


#: 容量探测的扫描上限。真正的上限远小于它（改地址撞订单数、退款判断撞非 paid 单数），
#: 设成它只是为了让「探测跑满也没撞重复」这种意外有个确定的返回值。
_CAPACITY_PROBE = 64


#: 去重时最多试多少个候选种子。40 轮实测够用（券抵扣这种候选最窄的一类
#: 也能在十几轮内找到新题），留余量应对分布不均。
_UNIQUE_SEED_ATTEMPTS = 40

#: 去重顺延步长。必须 > :data:`_TRAP_SEED_ATTEMPTS` 轮顺延的总跨度
#: （``39 * 40 = 1560``），否则两次去重尝试会踩进同一片种子、等于白试一次。
_SEED_UNIQUE_STEP = 2000

#: seed 空间的三级位宽，逐级从严到宽。**每级位宽必须大于下一级的最大跨度**，
#: 否则顺延会溢进上位、让两条不同 (类型, 下标) 的任务落进同一份沙箱 ——
#: 而同一份沙箱上两条题可能互相依赖（批量退款退掉的那单，正是改地址要改的那单）。
#:
#: - 一个下标的跨度 = ``39 * 2000 + 1560 = 79560`` ⇒ 位宽 100000
#: - 一个类型的跨度 = ``63 * 100000 = 6300000`` ⇒ 位宽 10000000
_SEED_INDEX_UNIT = 100_000
_SEED_TYPE_UNIT = 10_000_000
_SEED_ROOT_UNIT = 1_000_000_000


def _seed_of(seed: int, type_index: int, index: int) -> int:
    """沙箱 seed 派生：``(seed, 类型序号, 下标)`` → int。

    :func:`sandbox.make_sandbox` 只吃 int，所以这里必须产出 int，不能直接塞
    ``f"{seed}:orch:{task_type}"`` 那样的人读 seed —— 那是 :class:`random.Random`
    的用法，两者恰好都能收，但签名只写了 int。

    **三级位宽互不重叠**（见 :data:`_SEED_INDEX_UNIT` 一行的理由）：给定
    ``0 <= index < _CAPACITY_PROBE`` 与 ``_TRAP_SEED_ATTEMPTS`` 轮顺延，
    不同 (类型, 下标) 组合的 seed 区间不相交。
    """
    return seed * _SEED_ROOT_UNIT + type_index * _SEED_TYPE_UNIT + index * _SEED_INDEX_UNIT


def _task_key(spec: Spec) -> tuple[str, str]:
    """两条任务「其实是同一道题」的判据：**题面 + 必须产生的动作**都逐字相同。

    只看 delta 不够（券抵扣的 delta 恒为「券未使用」，十条题的 delta 逐字相同）；
    只看题面也不够（题面说改地址但 delta 没说要改，等于没要求改）。
    两样都相同 ⇒ agent 看到的题、要做的动作、判分器的期望三者全同，
    算两条就是同一份答案被计两次。

    这就是券抵扣能撑到十条而不用改沙箱的根据：**它那一类真正的候选空间是
    ``(券, 未达门槛的订单)`` 对**，换种子时这个对的取值就在变。
    """
    return (spec.instruction, json.dumps(spec.expected, ensure_ascii=False, sort_keys=True))


def _build_unique(task_type: str, seed: int, type_index: int, index: int,
                  seen: set[tuple]) -> Spec:
    """造第 ``index`` 条任务，顺延种子直到它的题面/动作**没被 :data:`seen` 用过**。

    顺延而不是复用：复用会让同一份答案被算多次，一致率与 delta 统计虚高。

    :param seen: **就地累积**，调用方复用同一个 set 就能保证整批不重复。
    """
    base = _seed_of(seed, type_index, index)
    for attempt in range(_UNIQUE_SEED_ATTEMPTS):
        try:
            spec = _build_with_trap(task_type, base + attempt * _SEED_UNIQUE_STEP, index)
        except ValueError:
            continue          # 该类型在这个 seed 邻域造不出陷阱，换邻域继续
        key = _task_key(spec)
        if key not in seen:
            seen.add(key)
            return spec
    raise ValueError(
        f"类型「{task_type}」在 seed {base} 起的 {_UNIQUE_SEED_ATTEMPTS} 个候选里"
        f"都造不出新题（要么造不出陷阱，要么题面与前面重复）—— "
        f"这是该类候选空间见底的信号，不是 seed 运气问题"
    )


def capacity(task_type: str, seed: int = 42) -> int:
    """这一类在该 seed 下**最多能造多少条互不重复的任务**。

    为什么要有这个函数：试标阶段的 ``per_type <= 6`` 用的是 ``len(orders) - 2``
    这个**代理值** —— 它的假设是「多条任务共用一份沙箱会互相污染」。实测（60 个
    sub-seed × 4 类）这个假设是错的：**每条任务本来就用独立 seed 造独立沙箱**，
    4 类 60 次全部造出陷阱，从未失败。真正的上限不是订单数，而是**去重后的候选空间**
    —— 某一类在这个沙箱形状下能造出的**互不相同**的题有多少。

    探测走的是与正式生成**完全相同**的那条路（同一个 :func:`_build_unique`），
    否则测出来的上限与实际能造出的条数对不上。
    """
    type_index = TASK_TYPES.index(task_type)
    seen: set[tuple] = set()
    for index in range(_CAPACITY_PROBE):
        try:
            _build_unique(task_type, seed, type_index, index, seen)
        except ValueError:
            return index      # 候选空间见底，这条 index 拿不到新题
    return _CAPACITY_PROBE


def build_orchestration_dataset(per_type: int = 2, seed: int = 42) -> list[Task]:
    """生成 C 场景任务。返回按 id 升序的稳定列表，总条数 = ``per_type * 4``。

    **每条任务一份自己的沙箱**（同 seed 但同实例内会互相污染：上一条任务退掉的单，
    下一条任务就退不到了 —— 同一批任务里出现「上一条的答案让这条的陷阱失效」）。
    沙箱由 ``make_sandbox(seed)`` 造，seed 随类型与下标派生，故同 seed 两次产物逐字节一致。

    **整批去重**（:func:`_build_unique` + 全批共享的 ``seen``）：跨类型去重也不做 ——
    四类题的题面与动作空间本就互不相交，真撞上了也只能是巧合，且同一条 order_id
    出现在两类题里是正常的（改地址改的单可能正好是批量退款要退的单）。

    ``per_type`` 超过 :func:`capacity` 时**报错**而不是循环复用：复用会让同一份答案
    被反复算，一致率与 delta 统计被同一样本计多次（与 :func:`qa.build_qa_dataset`
    同一条理由）。上限的推导见 :func:`capacity`。
    """
    if per_type < 1:
        raise ValueError(f"per_type 必须 >= 1，收到 {per_type}")
    limits = {t: capacity(t, seed) for t in TASK_TYPES}
    max_per_type = min(limits.values())
    if per_type > max_per_type:
        detail = "、".join(f"{t}={limits[t]}" for t in TASK_TYPES)
        raise ValueError(
            f"per_type={per_type} 超过单类去重容量 {max_per_type}（各类上限：{detail}）——"
            f"再大就得让多条任务共用一份答案，同一份答案被算多次会让一致率与 delta 统计虚高"
        )

    tasks: list[Task] = []
    seen: set[tuple] = set()
    seq = 0
    for type_index, task_type in enumerate(TASK_TYPES):
        for index in range(per_type):
            spec = _build_unique(task_type, seed, type_index, index, seen)
            seq += 1
            tasks.append(build_task(f"orch-{seq:04d}", spec, Split.TRAIN))
    tasks.sort(key=lambda t: t.id)
    return tasks


# --------------------------------------------------------------------------------------
# 正式集：40 条 × trial / dev / blind 三档
# --------------------------------------------------------------------------------------

#: 三档切分。顺序即产物目录的顺序。
C_SPLITS: tuple[str, ...] = ("trial", "dev", "blind")

#: 切分的**分组单位 = 环境 seed 组**。三档各占一段**互不相交的组号区间**，
#: 一个组出一套「四类各一条」—— 同一组造出的所有任务只落同一档，
#: 这就是契约要的「按环境 seed 分组切分」。
#:
#: **为什么用组号而不是直接切 seed 值**：沙箱 seed 由 ``(根 seed, 档内组序号)`` 派生
#: （见 :func:`_seed_of`），组号连续 ⇒ 三档的种子区间也连续 ⇒
#: 「某一档的全部任务都出自一段连续的种子区间」肉眼可审。
#: 而直接按 seed 值切，会被 :func:`_build_with_trap` 的顺延打散成不连续的集合。
#:
#: **比例**：2 : 4 : 4 ⇒ 8 / 16 / 16 条，占 20% / 40% / 40%，
#: 与契约的参考比例逐字对上。四类均分 ⇒ 每类 10 条（2 + 4 + 4），刚好压住下界。
SEED_GROUPS_PER_SPLIT: dict[str, range] = {
    "trial": range(0, 2),
    "dev": range(2, 6),
    "blind": range(6, 10),
}

#: 每档的总条数（组数 × 任务类型数）。**由 :data:`SEED_GROUPS_PER_SPLIT` 推导**，
#: 不另写一份 —— 另写一份的话，改了组数忘了改条数，两边就会各说各话。
FORMAL_SIZES: dict[str, int] = {
    split: len(SEED_GROUPS_PER_SPLIT[split]) * len(TASK_TYPES) for split in C_SPLITS
}

#: 每档每类的条数（= 该档的组数，四类均分）。
FORMAL_PER_TYPE: dict[str, dict[str, int]] = {
    split: {tt: len(SEED_GROUPS_PER_SPLIT[split]) for tt in TASK_TYPES}
    for split in C_SPLITS
}

#: 组的总数，也就是每类能造的总条数下标空间（10 个组号 ⇒ 下标 0..9）。
TOTAL_SEED_GROUPS = sum(len(g) for g in SEED_GROUPS_PER_SPLIT.values())


@dataclass(frozen=True)
class FormalTask:
    """正式集的一条任务。``split`` 是三级档位名（不是 schema 的两值枚举）。"""

    task: Task
    split: str
    seed_group: int


def _plan_splits() -> dict[str, list[int]]:
    """档位 → 组号列表。自检（区间不重叠、组数够）在生成期炸，不留到产物里才发现。

    **刻意不在模块级调用**：容量自检要真造一遍题，而造题函数定义在本模块更靠后的位置，
    模块级执行会 ``NameError``。改成 :func:`build_formal_dataset` 每次调用时自检 ——
    报错时机一样早（都在写盘之前），但不会把 import 变成一次隐式的全量生成。
    """
    groups = [g for split in C_SPLITS for g in SEED_GROUPS_PER_SPLIT[split]]
    if len(set(groups)) != len(groups):
        raise ValueError(
            f"三档的组号区间有重叠 —— 同一 seed 组会同时落两档，等于没切: {groups}"
        )
    plan = {split: list(SEED_GROUPS_PER_SPLIT[split]) for split in C_SPLITS}
    limits = {t: capacity(t) for t in TASK_TYPES}
    short = sorted(t for t, v in limits.items() if v < TOTAL_SEED_GROUPS)
    if short:
        detail = "、".join(f"{t}={limits[t]}" for t in short)
        raise ValueError(
            f"这些类型的去重容量装不下 {TOTAL_SEED_GROUPS} 条（每组一条）: {detail} —— "
            f"调 SEED_GROUPS_PER_SPLIT 或加候选"
        )
    return plan


def build_formal_dataset(seed: int = 42) -> list[FormalTask]:
    """生成 40 条正式任务，按 ``trial / dev / blind`` 三档切分。

    **一个组出一套「四类各一条」**：组内四类共用同一个组序号作下标，但**类型位不同
    ⇒ 沙箱不同**（见 :func:`_seed_of` 的三级位宽），所以同组的四道题不会互相依赖；
    而同一档的组之间各造各的，整批共享一个 ``seen`` 去重。

    **档内按行随机打散**：组号已经保证「同一组不跨档」，剩下的自由度是「档内 id 顺序」。
    打散让 id 不按组连号 —— 标注员连着标十条不会发现「这十条出自同一组种子」，
    也就不会把「同源」的偏差混进分数里。

    :param seed: 根 seed，与 :func:`build_orchestration_dataset` 同义（决定挑哪些订单 / 券）。
    """
    rng = random.Random(f"{seed}:orch-formal")
    tasks: list[FormalTask] = []
    seen: set[tuple] = set()
    plan = _plan_splits()
    for split in C_SPLITS:
        rows: list[FormalTask] = []
        for group in plan[split]:
            # 全局组序号（三档连续编号）当造题下标 ⇒ 每类的 10 条各不相同，
            # 且下标 < TOTAL_SEED_GROUPS = 10 远小于容量探测上限，一一对得上。
            index = _group_ordinal(split, group)
            for type_index, task_type in enumerate(TASK_TYPES):
                spec = _build_unique(task_type, seed, type_index, index, seen)
                task = build_task(
                    f"orch-{split}-{len(rows) + 1:04d}", spec, Split.TRAIN, stage=split
                )
                rows.append(FormalTask(task, split, group))
        rng.shuffle(rows)
        tasks.extend(rows)
    return tasks


def _group_ordinal(split: str, group: int) -> int:
    """组号 → 全局组序号（0 起，三档按 :data:`C_SPLITS` 顺序连续编号）。"""
    base = 0
    for earlier in C_SPLITS:
        if earlier == split:
            return base + SEED_GROUPS_PER_SPLIT[split].index(group)
        base += len(SEED_GROUPS_PER_SPLIT[earlier])
    raise ValueError(f"未知档位 {split!r}")   # pragma: no cover —— C_SPLITS 是模块常量


def write_formal_dataset(tasks: Sequence[FormalTask], output_dir: Path) -> dict[str, Path]:
    """写成 ``<output_dir>/{trial,dev,blind}/orchestration.jsonl`` 三个目录。原子写。

    每个档位一个文件（而不是一个带 split 字段的大文件）：盲测集要能整目录封存、
    不给开发流程读，单文件做不到那个「不打开」的保证（与 B 场景同一理由）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in C_SPLITS:
        rows = [ft.task for ft in tasks if ft.split == split]
        target = output_dir / split / ORCH_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(_dump_line(t) for t in sorted(rows, key=lambda t: t.id))
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8", newline="\n")
        os.replace(tmp, target)
        paths[split] = target
    return paths


def _formal_summary(tasks: Sequence[FormalTask], seed: int, output_dir: Path) -> None:
    """正式集的 stdout 汇总表。表头用 ASCII（Windows 控制台代码页未必能渲染中文）。"""
    print(f"preset=formal seed={seed} total={len(tasks)} output={output_dir}")
    print(f"{'split':<10}{'type':<14}{'total':>8}{'points':>9}{'groups':>8}")
    for split in C_SPLITS:
        rows = [ft for ft in tasks if ft.split == split]
        for task_type in TASK_TYPES:
            sub = [ft for ft in rows if ft.task.initial_data["任务类型"] == task_type]
            pts = sum(len(ft.task.rubric or ()) for ft in sub)
            print(f"{split:<10}{TYPE_LABELS[task_type]:<14}{len(sub):>8}{pts:>9}"
                  f"{len({ft.seed_group for ft in sub}):>8}")
        pts = sum(len(ft.task.rubric or ()) for ft in rows)
        print(f"{split:<10}{'SUBTOTAL':<14}{len(rows):>8}{pts:>9}"
              f"{len({ft.seed_group for ft in rows}):>8}")
    print(f"{'TOTAL':<10}{'':14}{len(tasks):>8}"
          f"{sum(len(ft.task.rubric or ()) for ft in tasks):>9}"
          f"{len({ft.seed_group for ft in tasks}):>8}")


#: 造陷阱时最多换多少个种子。40 轮不是概率上的自信 —— 券门槛造不出陷阱只在
#: 「最小一单都高于最低门槛」时发生，占比很小；40 轮全败说明环境本身不适合当任务环境。
_TRAP_SEED_ATTEMPTS = 40


def _build_with_trap(task_type: str, base: int, index: int) -> Spec:
    """按 ``base`` 造一条任务；本沙箱造不出该类型的陷阱就顺延再试（步长 :data:`_TRAP_SEED_ATTEMPTS`）。

    为什么需要重试而不是直接报错：沙箱的订单金额是 ``randrange(5000, 900000)``，
    8 单里最小的一单偶尔会高过**所有**券的门槛 —— 那时「券门槛」这个陷阱在该
    环境里物理上不成立。硬报错的话，8 个 sub-seed 里撞上一个就整批生成失败，
    构造器在正常用法下就得靠反复试 --seed 才跑得通。

    顺延是**确定性的**（只在撞上 ``NoTrapConstructible`` 时按 ``_TRAP_SEED_ATTEMPTS``
    步长跳），所以同 seed 仍然逐字节一致；而 rubric / delta 的非法**不走这条路**，
    那些是真 bug，必须炸出来。

    步长（:data:`_TRAP_SEED_ATTEMPTS` = 40）必须大于 1：顺延 +1 会踩到相邻下标的任务
    base（同一个 index 顺延出来的种子，正好是 ``index + 1`` 的种子，
    两条任务撞成同一份环境、白占一条样本）。

    ``index`` 一路透传给 builder —— 各 builder 按 index 取候选而非随机挑，
    顺延后重试仍取同一个 index，答案不会因为换种子而漂。
    """
    for attempt in range(_TRAP_SEED_ATTEMPTS):
        sub_seed = base + attempt * _TRAP_SEED_ATTEMPTS
        try:
            return _BUILDERS[task_type](make_sandbox(sub_seed), random.Random(sub_seed), index)
        except NoTrapConstructible:
            continue
    raise ValueError(
        f"类型「{task_type}」在 seed {base}..{base + _TRAP_SEED_ATTEMPTS - 1} 上"
        f"都造不出陷阱 —— 环境规模/分布需要调整，不是 seed 运气问题"
    )


# --------------------------------------------------------------------------------------
# 落盘 + CLI
# --------------------------------------------------------------------------------------

ORCH_FILENAME = "orchestration.jsonl"


def _dump_line(task: Task) -> str:
    return json.dumps(task.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"


def write_orchestration_dataset(tasks: Sequence[Task], output_dir: Path) -> Path:
    """写成单文件 ``<output_dir>/orchestration.jsonl``。原子写（临时文件 + ``os.replace``）。

    单文件而非 train/heldout 分目录：C 场景试标阶段规模小，且判分要**整批一起看**
    （同一类任务的陷阱要横向对比），拆开反而不好使。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / ORCH_FILENAME
    payload = "".join(_dump_line(t) for t in sorted(tasks, key=lambda t: t.id))
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(tmp, target)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m zh_agent_bench.tasks.orchestration",
        description="生成 zh-agent-bench 的 C 场景任务（有状态沙箱上的多工具编排）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，决定挑哪些订单 / 券")
    parser.add_argument("--per-type", type=int, default=2, help="每种编排类型生成多少条（>=1）")
    parser.add_argument(
        "--preset", choices=("trial", "formal"), default="trial",
        help="trial=单文件 N 条（试标/校准用）；formal=40 条三档（正式评测用）",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/orchestration/tasks"),
                        help="输出目录（写 orchestration.jsonl）")
    args = parser.parse_args(argv)

    # 边界校验必须发生在任何写盘动作之前 —— 非法输入不该留半份产物。
    if args.per_type < 1:
        parser.error(f"--per-type 必须 >= 1，收到 {args.per_type}")
        return 2  # pragma: no cover —— parser.error 不返回

    if args.preset == "formal":
        # formal 不用 --per-type：规模由 :data:`SEED_GROUPS_PER_SPLIT` 定死，
        # 给个 per_type 只会让人以为它能改条数。仍照常校验以免非法值静默通过。
        try:
            tasks = build_formal_dataset(seed=args.seed)
        except ValueError as exc:
            parser.error(str(exc))
            return 2  # pragma: no cover
        paths = write_formal_dataset(tasks, args.output_dir)
        _formal_summary(tasks, args.seed, args.output_dir)
        print("wrote " + " ".join(f"{k}={v}" for k, v in paths.items()))
        return 0

    try:
        tasks = build_orchestration_dataset(per_type=args.per_type, seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover
    path = write_orchestration_dataset(tasks, args.output_dir)

    print(f"per_type={args.per_type} seed={args.seed} total={len(tasks)} output={path}")
    print(f"{'type':<14}{'total':>8}{'points':>9}{'orders':>9}{'delta':>8}")
    for task_type in TASK_TYPES:
        rows = [t for t in tasks if t.initial_data["任务类型"] == task_type]
        points = sum(len(t.rubric or ()) for t in rows)
        orders = sum(len(t.ground_truth["expected_final_state"]["orders"]) for t in rows)
        print(f"{TYPE_LABELS[task_type]:<14}{len(rows):>8}{points:>9}{orders:>9}{'':>8}")
    print(f"{'TOTAL':<14}{len(tasks):>8}"
          f"{sum(len(t.rubric or ()) for t in tasks):>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
