"""任务构造器 —— 程序化生成中文办公任务与客观答案。

运行期仅用 stdlib（不引第三方依赖），不联网：所有中文词表都是内置的、人工审核过的
虚构办公词表（人名/部门/城市/公司/商品），不使用真实个人信息。

三个命门（对应 TASK-001 的验收判据）：

1. **确定性**：同 seed 两次运行产物逐字节一致。所有随机性都来自 ``random.Random(字符串种子)``
   （CPython 对 str 种子走 sha512，不受 PYTHONHASHSEED 影响），行序按 id 升序写。
2. **防污染**：实体池先整体切成 train 段 / heldout 段**两半不相交**，再各自生成任务；
   所以 heldout 用的人名/公司/商品组合绝不出现在 train。这是「按池分组整体切分」，
   不是按行随机切。
3. **客观答案可精确判分**：金额一律整数分 ``int`` 存储，统计答案用 ``Decimal`` +
   ``ROUND_HALF_UP`` 算完序列化成字符串，原生 float 不进入金额与答案字段。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from .schema import (
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
)

# --------------------------------------------------------------------------------------
# 人工审核的中文办公词表（虚构、贴近真实办公语境，不含真实个人敏感信息）
# --------------------------------------------------------------------------------------

NAMES: tuple[str, ...] = (
    "王小明", "李静", "张伟", "王芳", "刘洋", "陈杰", "杨帆", "赵敏", "黄磊", "周涛",
    "吴倩", "徐强", "孙丽", "胡军", "朱婷", "高翔", "林峰", "何欣", "郭鹏", "马超",
    "罗静怡", "梁宇", "宋佳", "唐磊", "韩雪", "冯涛", "于洋", "董明", "程琳", "曹阳",
    "袁晓东", "邓颖", "许斌", "傅敏", "沈括", "曾毅", "彭宇", "吕娜", "苏航", "蒋涛",
    "蔡雯", "贾立", "丁磊", "魏琳", "薛峰", "叶青", "阎志", "余强", "潘宁", "杜衡",
    "戴俊", "夏雨", "钟琪", "汪涛", "田甜", "任健", "姜蕊", "范明", "方磊", "石岩",
    "姚静", "谭峰", "廖辉", "邹凯", "熊勇", "金兰", "陆明", "郝嘉", "孔睿", "白洁",
    "崔健", "康宁", "毛毅", "邱枫", "秦朗", "江涛", "翁颖", "樊磊", "温华", "季平",
)

#: 规模下限（TASK-004 §2.5）：人名 >= 60，其余 >= 30。
#: ``split_pools`` 会把池子**整体对半切**，所以 32 的池每侧留 16 个，
#: 足够 ``take(pool, start, ROWS_PER_TABLE=6)`` 在任务内取到 6 个互不重复的实体。
DEPARTMENTS: tuple[str, ...] = (
    "研发部", "市场部", "销售部", "人力资源部", "财务部", "客户服务部",
    "产品部", "运营部", "法务部", "行政部", "采购部", "数据部",
    "战略发展部", "质量管理部", "供应链管理部", "品牌公关部",
    "信息安全部", "技术支撑部", "商务拓展部", "培训发展部",
    "资产管理部", "项目管理部", "知识产权部", "审计监察部",
    "总务后勤部", "国际业务部", "产品设计部", "用户运营部",
    "数据治理部", "硬件工程部", "解决方案部", "渠道服务部",
)

CITIES: tuple[str, ...] = (
    "北京", "上海", "广州", "深圳", "杭州", "成都",
    "南京", "武汉", "西安", "苏州", "青岛", "重庆",
    "天津", "长沙", "郑州", "合肥", "厦门", "福州",
    "济南", "大连", "宁波", "无锡", "昆明", "贵阳",
    "沈阳", "哈尔滨", "石家庄", "太原", "南宁", "南昌",
)

COMPANIES: tuple[str, ...] = (
    "杭州云栖科技有限公司",
    "上海澜舟信息技术有限公司",
    "北京星辰智造有限公司",
    "深圳前海数科有限公司",
    "成都锦官云创有限公司",
    "广州南沙智联有限公司",
    "苏州工业园云算有限公司",
    "南京玄武数科有限公司",
    "武汉光谷云启有限公司",
    "西安高新数智有限公司",
    "青岛蓝湾智造有限公司",
    "重庆两江云科有限公司",
    "天津滨海智算有限公司",
    "长沙麓山云图有限公司",
    "郑州中原数联有限公司",
    "合肥巢湖智造有限公司",
    "厦门海沧云创有限公司",
    "福州闽江数科有限公司",
    "济南泉城智联有限公司",
    "大连金普云启有限公司",
    "宁波甬江数智有限公司",
    "无锡太湖云算有限公司",
    "昆明滇池数科有限公司",
    "贵阳数谷智造有限公司",
    "沈阳浑南云联有限公司",
    "哈尔滨松江数智有限公司",
    "石家庄冀中云创有限公司",
    "太原晋阳数科有限公司",
    "南宁邕江智联有限公司",
    "南昌赣江云算有限公司",
    "珠海横琴数科有限公司",
    "东莞松山湖智造有限公司",
)

#: 办公/数码语境下的通用商品词（虚构品类，不含任何品牌名）。
PRODUCTS: tuple[str, ...] = (
    "智能会议平板", "无线降噪耳机", "人体工学椅", "机械键盘", "便携固态硬盘",
    "高清摄像头", "桌面显示器", "智能音箱", "高速扫描仪", "激光打印机",
    "企业级路由器", "服务器机柜", "空气净化器", "净水器", "全自动咖啡机",
    "家用投影仪", "平板支架", "多功能扩展坞", "电竞鼠标", "显示器挂架",
    "桌面文档柜", "人体工学鼠标垫", "无线演示器", "视频会议终端", "指纹考勤机",
    "碎纸机", "电子白板", "台式工作站", "笔记本支架底座", "USB-C 扩展坞",
    "降噪门禁一体机", "智能工牌", "双面复印纸",
)

TABLE_NAMES: tuple[str, ...] = (
    "员工信息表", "销售明细表", "商品销售表", "考勤记录表", "订单汇总表",
)

#: format 任务的表名。**initial_data 与 ground_truth 都从这里取同一个值** ——
#: TASK-005 实测踩出来的坑：当初只把它写进 ground_truth，指令与 initial_data 都没有出处，
#: 于是任何不作弊的 agent 都答不对（它测的是「你猜不猜得到我写死的常量」，
#: 不是能力）。findings §七.1 的可解性铁律：真值必须能仅凭 instruction + initial_data 重建。
FORMAT_TABLE_NAME = "商品销售表"

#: 全年 36 个工作日（每月 3 个）。取数是 ``(start + i) % len(DATES)``，
#: 扩池纯粹是为了让 30 类任务的日期分布铺满一年，而不是每条任务都在同几个月里打转。
DATES: tuple[str, ...] = (
    "2026-01-05", "2026-01-15", "2026-01-26",
    "2026-02-03", "2026-02-11", "2026-02-25",
    "2026-03-04", "2026-03-16", "2026-03-27",
    "2026-04-07", "2026-04-15", "2026-04-23",
    "2026-05-06", "2026-05-14", "2026-05-28",
    "2026-06-08", "2026-06-17", "2026-06-29",
    "2026-07-06", "2026-07-15", "2026-07-24",
    "2026-08-05", "2026-08-13", "2026-08-26",
    "2026-09-02", "2026-09-16", "2026-09-28",
    "2026-10-09", "2026-10-20", "2026-10-29",
    "2026-11-04", "2026-11-18", "2026-11-27",
    "2026-12-03", "2026-12-15", "2026-12-24",
)

#: 每次任务表里的行数上限（每类共用）。切片去重要求 行数 <= 池子长度。
ROWS_PER_TABLE = 6

#: organize 子类要排出的名次。
TOP_N = 3


# --------------------------------------------------------------------------------------
# 切分规则（固定、写死、被测试逐值断言）
# --------------------------------------------------------------------------------------


def heldout_count(n: int) -> int:
    """heldout 条数 = ``round(N * 0.2)``。

    用 ``Decimal`` 乘再 ``round()``，避免 0.2 的二进制浮点误差；
    ``round()`` 对 Decimal 走 ROUND_HALF_EVEN（银行家舍入），规则钉死在这里。
    """
    return int(round(Decimal(n) * Decimal("0.2")))


def split_pools(seed: int, tag: str, pool: Sequence[str]) -> dict[Split, list[str]]:
    """把一个实体池整体切成 train 段 / heldout 段**两半不相交**。

    这是防污染的全部机制：heldout 任务只能从 heldout 段取实体，
    train 任务只能从 train 段取 ⇒ 两个 split 的实体集合必然无交集。
    池内按 seed 打乱后再切，保证不同 seed 下 train/heldout 的实体组合不同。
    """
    items = list(pool)
    rng = random.Random(f"{seed}:{tag}")
    rng.shuffle(items)
    half = len(items) // 2
    if half == 0:
        raise ValueError(f"实体池 {tag} 太短，无法切成两半")
    return {Split.HELDOUT: items[:half], Split.TRAIN: items[half:]}


def take(pool: Sequence[str], start: int, count: int) -> list[str]:
    """从池中循环取 ``count`` 个元素；只要 ``count <= len(pool)`` 就保证互不重复。

    跨任务复用池子是允许的（同一 split 内）；只要任务内键唯一即可判分。
    """
    if count > len(pool):
        raise ValueError(f"需要 {count} 个实体，但池子只有 {len(pool)} 个")
    n = len(pool)
    return [pool[(start + i) % n] for i in range(count)]


# --------------------------------------------------------------------------------------
# 四个子类：初始数据 + 客观答案 + answer_spec
# --------------------------------------------------------------------------------------

#: 每个 category 对应的 answer_spec。comparison 不隐含在 category 里，显式落表。
ANSWER_SPECS: dict[Category, AnswerSpec] = {
    Category.QUERY: AnswerSpec(AnswerKind.STRING, Comparison.EXACT, Normalization.STRIP),
    Category.STATS: AnswerSpec(AnswerKind.NUMBER, Comparison.DECIMAL_EXACT, Normalization.NONE),
    Category.ORGANIZE: AnswerSpec(AnswerKind.ORDERED_LIST, Comparison.ORDERED_LIST_EXACT, Normalization.STRIP),
    Category.FORMAT: AnswerSpec(AnswerKind.JSON, Comparison.JSON_CANONICAL, Normalization.NONE),
}


def _amount_cents(rng: random.Random, low: int, high: int) -> int:
    """生成整数分金额。禁止原生 float —— 浮点会破坏精确判分。"""
    return rng.randrange(low, high)


def _build_query(
    rng: random.Random,
    names: Sequence[str],
    depts: Sequence[str],
    cities: Sequence[str],
    company: str,
    start: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """数据查询：查某人部门。查询键（姓名）在任务内唯一，答案确定值。"""
    picked = take(names, start, ROWS_PER_TABLE)
    target_idx = rng.randrange(ROWS_PER_TABLE)
    rows = [
        {
            "姓名": name,
            "部门": depts[(start + i) % len(depts)],
            "城市": cities[(start + i) % len(cities)],
            "入职日期": DATES[(start + i) % len(DATES)],
        }
        for i, name in enumerate(picked)
    ]
    target = rows[target_idx]
    initial_data = {"表名": "员工信息表", "公司": company, "rows": rows}
    instruction = f"查一下『{target['姓名']}』的部门是什么"
    return initial_data, {"answer": target["部门"]}, instruction


def _build_stats(
    rng: random.Random,
    names: Sequence[str],
    prods: Sequence[str],
    cities: Sequence[str],
    company: str,
    start: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """统计：算平均值。金额整数分存储，答案 Decimal + ROUND_HALF_UP 的十进制字符串。"""
    picked = take(names, start, ROWS_PER_TABLE)
    rows = [
        {
            "姓名": name,
            "商品": prods[(start + i) % len(prods)],
            "城市": cities[(start + i) % len(cities)],
            "销售额": _amount_cents(rng, 5_000, 20_000_000),
        }
        for i, name in enumerate(picked)
    ]
    # 平均值：先在「分」上求和，再折元，最后 ROUND_HALF_UP 到两位小数。
    total_cents = sum(r["销售额"] for r in rows)
    avg_yuan = (Decimal(total_cents) / Decimal(len(rows))) / Decimal(100)
    answer = str(avg_yuan.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    initial_data = {"表名": "销售明细表", "公司": company, "rows": rows}
    instruction = f"计算「销售明细表」中「销售额」的平均值，保留两位小数"
    return initial_data, {"answer": answer}, instruction


def _build_organize(
    rng: random.Random,
    names: Sequence[str],
    cities: Sequence[str],
    company: str,
    start: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """整理：按销售额降序取前 3 名姓名，并列按姓名升序（次级键写进指令，答案唯一确定）。"""
    picked = take(names, start, ROWS_PER_TABLE)
    # 取值范围刻意压窄 —— 制造并列，强制次级键真正生效。
    base = rng.randrange(100_000, 400_000)
    rows = [
        {
            "姓名": name,
            "城市": cities[(start + i) % len(cities)],
            "销售额": base + rng.randrange(0, 3) * 1_000,
        }
        for i, name in enumerate(picked)
    ]
    ranked = sorted(rows, key=lambda r: (-r["销售额"], r["姓名"]))
    answer = [r["姓名"] for r in ranked[:TOP_N]]
    initial_data = {"表名": "销售明细表", "公司": company, "rows": rows}
    instruction = f"按销售额降序排出前 {TOP_N} 名的姓名（并列按姓名升序）"
    return initial_data, {"answer": answer}, instruction


def _build_format(
    rng: random.Random,
    names: Sequence[str],
    prods: Sequence[str],
    cities: Sequence[str],
    company: str,
    start: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """格式转换：CSV -> JSON。CSV 方言固定（delimiter=','、quotechar='"'、UTF-8、首行表头、空字段为 ""）。

    **表名进 ``initial_data``**（findings §七.1）：答案 JSON 的顶层 ``表名`` 键取自
    ``initial_data["表名"]``，与 ``ground_truth`` 同一个常量 —— 任务因此对不作弊的 agent 可解。
    """
    picked = take(names, start, ROWS_PER_TABLE)
    headers = ["姓名", "商品", "城市", "销售额"]
    rows = []
    for i, name in enumerate(picked):
        rows.append(
            {
                "姓名": name,
                "商品": prods[(start + i) % len(prods)],
                # 第 2 行刻意留空，验证空字段序列化为 "" 的约定
                "城市": cities[(start + i) % len(cities)] if i != 2 else "",
                "销售额": str(_amount_cents(rng, 1_000, 9_000_000)),
            }
        )
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers, delimiter=",", quotechar='"',
                            lineterminator="\n")  # 强制 LF，避免 CRLF 混进 initial_data
    writer.writeheader()
    writer.writerows(rows)
    initial_data = {
        "表名": FORMAT_TABLE_NAME,
        "公司": company,
        "csv_text": buf.getvalue(),
        "dialect": {"delimiter": ",", "quotechar": '"', "encoding": "utf-8", "空字段": ""},
    }
    # 表名从 initial_data 取，与它同源，不再是只有求解器才知道的暗号。
    answer_obj = {"表名": initial_data["表名"], "rows": rows}
    instruction = f"把这份 CSV 转成 JSON（`表名` 用 initial_data 里的「{FORMAT_TABLE_NAME}」）"
    return initial_data, {"answer": answer_obj}, instruction


# --------------------------------------------------------------------------------------
# 组装
# --------------------------------------------------------------------------------------


def _split_of(category: Category, n: int, index: int, h_count: int) -> Split:
    """id 升序排列后，前 ``h_count`` 条进 heldout，其余进 train。

    id 在 split 内部连续（query-0001..query-0006 进 train），全局唯一，
    扩容时只会在尾部追加，旧 id 不变。
    """
    return Split.HELDOUT if index < h_count else Split.TRAIN


def build_category(
    category: Category,
    per_category: int,
    seed: int,
) -> list[Task]:
    """生成某个 category 的全部任务（train + heldout 混在一个列表里，按 id 升序）。"""
    h_count = heldout_count(per_category)
    pools = {
        "name": split_pools(seed, f"{category.value}:name", NAMES),
        "prod": split_pools(seed, f"{category.value}:prod", PRODUCTS),
        "dept": split_pools(seed, f"{category.value}:dept", DEPARTMENTS),
        "city": split_pools(seed, f"{category.value}:city", CITIES),
        "comp": split_pools(seed, f"{category.value}:comp", COMPANIES),
    }
    tool_names = {
        Category.QUERY: ("table.query",),
        Category.STATS: ("table.stats",),
        Category.ORGANIZE: ("table.organize",),
        Category.FORMAT: ("format.csv_to_json",),
    }[category]

    tasks: list[Task] = []
    for index in range(per_category):
        split = _split_of(category, per_category, index, h_count)
        rng = random.Random(f"{seed}:{category.value}:{index}")
        start = index * ROWS_PER_TABLE
        names = pools["name"][split]
        prods = pools["prod"][split]
        depts = pools["dept"][split]
        cities = pools["city"][split]
        comps = pools["comp"][split]
        company = comps[index % len(comps)]

        if category is Category.QUERY:
            initial_data, gt, instruction = _build_query(rng, names, depts, cities, company, start)
        elif category is Category.STATS:
            initial_data, gt, instruction = _build_stats(rng, names, prods, cities, company, start)
        elif category is Category.ORGANIZE:
            initial_data, gt, instruction = _build_organize(rng, names, cities, company, start)
        elif category is Category.FORMAT:
            initial_data, gt, instruction = _build_format(rng, names, prods, cities, company, start)
        else:  # pragma: no cover —— 枚举封闭，走不到
            raise ValueError(f"未知 category: {category}")

        tasks.append(
            Task(
                id=f"{category.value}-{index + 1:04d}",
                category=category,
                schema_version=A_SCENE_SCHEMA_VERSION,
                instruction=instruction,
                initial_data=initial_data,
                tool_names=tool_names,
                ground_truth=gt,
                answer_spec=ANSWER_SPECS[category],
                split=split,
            )
        )
    return tasks


def build_dataset(per_category: int = 8, seed: int = 42) -> list[Task]:
    """生成全部 A 场景四类任务；返回按 (category 枚举序, id) 稳定排序的列表。

    只遍历 :data:`A_SCENE_CATEGORIES`，**不是** ``Category`` 全枚举（TASK-008 起）：
    ``Category`` 多了个 B 场景的 ``RETRIEVAL``，而 ``build_category`` 只会造 A 场景四类
    的任务（它按 category 取词表、分发工具、真值算法都是 A 场景专属）。
    B 场景由 ``tasks/qa.py`` 单独生成 —— brainstorm §4.5 要求 A、B 分榜，
    两套任务本来就不该挤在同一个生成器里。
    """
    tasks: list[Task] = []
    for category in A_SCENE_CATEGORIES:
        tasks.extend(build_category(category, per_category, seed))
    tasks.sort(key=lambda t: (A_SCENE_CATEGORIES.index(t.category), t.id))
    return tasks


# --------------------------------------------------------------------------------------
# 落盘
# --------------------------------------------------------------------------------------

#: 本单管理的产物文件名后缀。清旧产物时只碰这个后缀 + train/heldout 两个子目录。
_SUFFIX = ".jsonl"


def _dump_line(task: Task) -> str:
    """一行 JSON：utf-8 原样中文（ensure_ascii=False）、紧凑分隔符、单个 \\n 结尾。"""
    return json.dumps(task.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"


def write_dataset(tasks: Sequence[Task], output_dir: Path) -> list[Path]:
    """把任务写进 ``<output_dir>/{train,heldout}/<category>.jsonl``。

    - 只清 ``<output_dir>/train`` 与 ``<output_dir>/heldout`` 下以 ``.jsonl`` 结尾的旧文件，
      不删目录、不碰目录外任何东西。
    - 原子写：先写同目录临时文件，再 ``os.replace`` 覆盖。
    """
    written: list[Path] = []
    for split in (Split.TRAIN, Split.HELDOUT):
        split_dir = output_dir / split.value
        split_dir.mkdir(parents=True, exist_ok=True)
        for stale in sorted(split_dir.glob(f"*{_SUFFIX}")):
            stale.unlink()
    for category in A_SCENE_CATEGORIES:
        for split in (Split.TRAIN, Split.HELDOUT):
            rows = [t for t in tasks if t.category is category and t.split is split]
            rows.sort(key=lambda t: t.id)
            target = output_dir / split.value / f"{category.value}{_SUFFIX}"
            payload = "".join(_dump_line(t) for t in rows)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(payload, encoding="utf-8", newline="\n")
            os.replace(tmp, target)
            written.append(target)
    return written


def _validate_args(per_category: int, seed: int) -> None:
    """边界校验：在任何写盘动作之前抛错，保证非法输入不落文件。"""
    if per_category < 1:
        raise ValueError(f"--per-category 必须 >= 1，收到 {per_category}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"--seed 必须为整数，收到 {seed!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m zh_agent_bench.tasks.build",
        description="生成 zh-agent-bench 的任务数据集（train/heldout）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，决定实体池切分与任务内容")
    parser.add_argument("--per-category", type=int, default=8, help="每个 category 生成多少条任务（>=1）")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/tasks"),
        help="输出根目录（其下建 train/ 与 heldout/）",
    )
    args = parser.parse_args(argv)

    try:
        _validate_args(args.per_category, args.seed)
    except ValueError as exc:
        parser.error(str(exc))  # argparse 以 exit code 2 非零退出，且发生在写盘之前
        return 2  # pragma: no cover —— parser.error 不返回

    tasks = build_dataset(per_category=args.per_category, seed=args.seed)
    write_dataset(tasks, args.output_dir)

    print(f"per_category={args.per_category} seed={args.seed} output={args.output_dir}")
    print(f"{'category':<10}{'train':>8}{'heldout':>10}{'total':>8}")
    for category in A_SCENE_CATEGORIES:
        tr = sum(1 for t in tasks if t.category is category and t.split is Split.TRAIN)
        ho = sum(1 for t in tasks if t.category is category and t.split is Split.HELDOUT)
        print(f"{category.value:<10}{tr:>8}{ho:>10}{tr + ho:>8}")
    tot_tr = sum(1 for t in tasks if t.split is Split.TRAIN)
    tot_ho = sum(1 for t in tasks if t.split is Split.HELDOUT)
    # 汇总行标签用 ASCII：Windows 控制台代码页未必能正确渲染中文，
    # 而判据要求「stdout 打印行数汇总」，标签必须人人可读。
    print(f"{'TOTAL':<10}{tot_tr:>8}{tot_ho:>10}{tot_tr + tot_ho:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())