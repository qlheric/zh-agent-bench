"""任务 schema —— 全项目的数据契约根。

本模块只定义「任务长什么样」与「答案怎么被机器精确判分」（answer_spec）。
后续 TASK-002（工具契约）、TASK-003（judge）都依赖这里的枚举与字段，不得漂移。

设计要点：
- 全部 dataclass 用 ``frozen=True``：Task 生成后不可变，避免构造器与 judge 各自改一份。
- 枚举一律 ``str`` 子类，``json.dumps`` 直接序列化即得小写标识（``Category.QUERY -> "query"``）。
- tuple 在序列化时统一转 list，保证 round-trip 无损。
- 本模块运行时仅用 stdlib。

**A / B 两场景（TASK-008 起）**：
- **A 场景**（``query``/``stats``/``organize``/``format``）客观判分，答案确定，
  四个比较算子直接比。本模块对它们的枚举与字段语义**一个字都没改**。
- **B 场景**（``retrieval``）知识检索 QA，答案主观开放 → 改走
  ``Comparison.JUDGE``：**逐条 rubric 打点、程序聚合**（G = Σ w·s / Σ w，s ∈ {0, 0.5, 1}）。
  judge 不是「拿答案和参考答案比语义等价」—— 那种做法会误杀合法答案、奖励复述参考答案
  （gpt-6-astra 头脑风暴 P0-1），所以参考答案只作**示例**，评分依据是 :attr:`Task.rubric`。

brainstorm §4.5 要求 A、B **分榜**：A 的程序判分与 B 的估计判分可靠性不同，
直接加权会制造精确的假象 —— 所以 :class:`Category` 一枚举并存，由场景各自取用。

**C 场景（TASK-019）**：``orchestration`` 多工具编排，跑在有状态沙箱上。
它也带 ``rubric``，但判分看的是**最终状态的 delta**（哪些订单/券被改成了什么），
不只是答案文本 —— 所以单列一个 :attr:`Comparison.ORCHESTRATION`，绝不与 B 的
``judge`` 混用（judge 只读答案文本，看不到沙箱被改成什么样）。
A、B 两场景的字段与枚举语义**一个字都没改**。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1.1"
"""schema 版本号。为后续演进留位；改动字段语义时必须 bump。

TASK-008 加了 :attr:`Task.rubric` 与 :attr:`Category.RETRIEVAL`、:attr:`Comparison.JUDGE`。
加字段 + 加枚举是**非破坏性**变更，故 bump 到 1.1。
"""

A_SCENE_SCHEMA_VERSION = "1.0"
"""**A 场景已入库数据**生成时用的版本号，冻结在 1.0。

为什么要单独一个常量：``benchmarks/tasks/**`` 是在 1.0 下生成并提交的，
测试 ``test_committed_dataset_matches_the_pinned_generation_command`` 钉着
「用固定命令重新生成 = 逐字节等于入库产物」。若把 :data:`SCHEMA_VERSION` 一改，
A 场景重新生成就会与入库的 120 条产生 120 处 ``schema_version`` 差异 ——
要么重写已入库基准数据（数据是证据，不该随手改），要么让那条锁失效。
两个都不该。所以 A 场景任务继续盖 1.0，B 场景任务盖 1.1，各自自报家门。
"""


class Category(str, Enum):
    """任务子类。值即 jsonl 中的小写标识。"""

    QUERY = "query"
    STATS = "stats"
    ORGANIZE = "organize"
    FORMAT = "format"
    RETRIEVAL = "retrieval"   # B 场景（TASK-008）：知识检索 QA，rubric + 中文 judge
    ORCHESTRATION = "orchestration"   # C 场景（TASK-019）：多工具编排，delta + 编排 + 回复

    @property
    def is_b_scene(self) -> bool:
        """是否 B 场景。B 场景才带 ``rubric``、才走 judge 判分。"""
        return self is Category.RETRIEVAL

    @property
    def is_c_scene(self) -> bool:
        """是否 C 场景。C 场景也带 ``rubric``，但判分看「沙箱最终状态的 delta」。

        与 :attr:`is_b_scene` **互斥**：B 的 rubric 只对答案文本打分，看不到沙箱被改成
        什么样；C 的 rubric 要同时管编排点（该不该退的没退）与回复点。两套判分口径
        混用会让「世界改对了但没说清」和「说清了但没改」拿同一分。
        """
        return self is Category.ORCHESTRATION


#: A 场景的四类。``tasks/build.py`` 按它生成（它只认 A 场景），B 场景由 ``tasks/qa.py`` 生成。
A_SCENE_CATEGORIES: tuple[Category, ...] = (
    Category.QUERY, Category.STATS, Category.ORGANIZE, Category.FORMAT,
)



class Split(str, Enum):
    """数据集划分。"""

    TRAIN = "train"
    HELDOUT = "heldout"


class AnswerKind(str, Enum):
    """答案类型（判分时先按它判断怎么解析答案）。"""

    STRING = "string"
    NUMBER = "number"
    ORDERED_LIST = "ordered_list"
    JSON = "json"


class Comparison(str, Enum):
    """比较算子。``answer_spec.comparison`` 的取值域。"""

    EXACT = "exact"                       # 字符串逐字相等
    DECIMAL_EXACT = "decimal_exact"       # 双方 Decimal 化后精确相等
    ORDERED_LIST_EXACT = "ordered_list_exact"  # 按序逐元素相等，长度必须一致
    JSON_CANONICAL = "json_canonical"     # 解析为 JSON 对象后 sort_keys 再比较，不比原文
    JUDGE = "judge"                       # B 场景：逐条 rubric 打点、程序聚合（不由 judge.py 实现）
    ORCHESTRATION = "orchestration"       # C 场景：沙箱状态 delta + 编排点 + 回复点（TASK-020 实现）


#: B 场景 retrieval 任务唯一合法的比较方式。
B_SCENE_COMPARISON = Comparison.JUDGE

#: C 场景 orchestration 任务唯一合法的比较方式。
#: **必须与 B_SCENE_COMPARISON 分开钉**：C 要看沙箱的最终状态，judge 只看答案文本。
#: 复用 ``JUDGE`` 会让判分器拿不到沙箱快照，「世界改对了」这一半就没人判了。
C_SCENE_COMPARISON = Comparison.ORCHESTRATION



class Normalization(str, Enum):
    """比较前的规范化策略。"""

    NONE = "none"
    STRIP = "strip"


#: (kind, comparison) 的合法组合。schema 层就把非法组合挡住，避免生成出无法判分的数据。
LEGAL_COMPARISONS: dict[AnswerKind, Comparison] = {
    AnswerKind.STRING: Comparison.EXACT,
    AnswerKind.NUMBER: Comparison.DECIMAL_EXACT,
    AnswerKind.ORDERED_LIST: Comparison.ORDERED_LIST_EXACT,
    AnswerKind.JSON: Comparison.JSON_CANONICAL,
}


@dataclass(frozen=True)
class AnswerSpec:
    """判分契约：judge 靠它知道「怎么把 agent 答案和 ground truth 比」。

    comparison 不能隐含在 category 里，必须显式落到每一类任务上。
    """

    kind: AnswerKind
    comparison: Comparison
    normalization: Normalization

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "comparison": self.comparison.value,
            "normalization": self.normalization.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnswerSpec":
        kind = AnswerKind(data["kind"])
        comparison = Comparison(data["comparison"])
        spec = cls(kind=kind, comparison=comparison, normalization=Normalization(data["normalization"]))
        expected = LEGAL_COMPARISONS.get(kind)
        # ``JUDGE``（B）与 ``ORCHESTRATION``（C）是**仅有的两个**不由 kind 决定的例外：
        # 答案是否开放式与它被序列化成 string 还是 json 无关。
        # 放它们过这道闸，其余 (kind, comparison) 组合仍逐字受 LEGAL_COMPARISONS 约束 ——
        # A 场景四种 (kind, comparison) 的对应关系一个字没动。
        if comparison in (Comparison.JUDGE, Comparison.ORCHESTRATION):
            return spec
        if expected is not None and comparison != expected:
            raise ValueError(
                f"answer_spec 非法：kind={kind.value} 只允许 comparison={expected.value}，"
                f"收到 {comparison.value}"
            )
        return spec


@dataclass(frozen=True)
class Task:
    """一条评测任务。

    字段契约见 TASK-001 §2.5，构造器与测试都以此为准。

    :attr:`rubric` 是 TASK-008 加的**唯一新字段**，且是 B 场景专属：A 场景恒为
    ``None``，序列化时**不出现这个键** —— 见 :meth:`to_dict` 的说明。
    """

    id: str
    category: Category
    schema_version: str
    instruction: str
    initial_data: dict[str, Any]
    tool_names: tuple[str, ...]
    ground_truth: dict[str, Any]
    answer_spec: AnswerSpec
    split: Split
    #: B 场景的评分要点清单，每点一个 dict（见 :mod:`tasks.qa`）。
    #: **A 场景必须为 ``None``** —— 有 rubric 就意味着要走 judge 打分，而 A 场景是客观判分。
    rubric: tuple[dict[str, Any], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为 jsonl 一行的对象（tuple -> list，枚举 -> 小写 str）。

        ``rubric`` 为 ``None`` 时**整个键不写**，而不是写 ``null``。原因：
        ``benchmarks/tasks/**`` 那 120 条已入库 A 场景数据是在本字段存在之前生成的，
        判据钉着「用固定命令重新生成 = 逐字节等于入库产物」。若 A 场景也吐出
        ``"rubric": null``，120 行全部对不上，要么重写已入库基准数据、要么让那条锁失效。
        缺省键由 :meth:`from_dict` 的 ``data.get("rubric")`` 兜住，round-trip 仍无损。
        """
        payload = {
            "id": self.id,
            "category": self.category.value,
            "schema_version": self.schema_version,
            "instruction": self.instruction,
            "initial_data": self.initial_data,
            "tool_names": list(self.tool_names),
            "ground_truth": dict(self.ground_truth),
            "answer_spec": self.answer_spec.to_dict(),
            "split": self.split.value,
        }
        if self.rubric is not None:
            payload["rubric"] = [dict(point) for point in self.rubric]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """jsonl 一行 -> Task。与 to_dict 严格互逆（round-trip 判据 ⑦）。

        ``rubric`` 用 ``data.get`` 而不是 ``data[...]``：1.0 的 A 场景数据里没有这个键，
        但那**不**代表它是可选的 —— 缺了键仍要读出 ``rubric is None``（判据 ③）。
        """
        return cls(
            id=data["id"],
            category=Category(data["category"]),
            schema_version=data["schema_version"],
            instruction=data["instruction"],
            initial_data=data["initial_data"],
            tool_names=tuple(data["tool_names"]),
            ground_truth=data["ground_truth"],
            answer_spec=AnswerSpec.from_dict(data["answer_spec"]),
            split=Split(data["split"]),
            rubric=tuple(data["rubric"]) if data.get("rubric") is not None else None,
        )