# zh-agent-bench

**中文 agent 工具调用基准** —— 三个场景，一套客观判分口径，clone 即跑。

被测对象是 agent，不是 benchmark：题目由程序生成（标准答案确定），
判分不依赖另一个模型打主观分，**同 seed 连跑两次榜单逐字节一致**。
运行时零第三方依赖，只用 Python ≥ 3.12 标准库。

| 场景 | 考什么 | 规模 | 判分 |
|---|---|---|---|
| **A 办公数据** | 读指令 → 调对工具 → 给出可精确比对的答案 | 120 条（heldout 24） | 程序 `exact` 比对 |
| **B 知识检索** | 从企业知识库取证，并**承认不知道** | 390 条（dev 120 / trial 60 / blind 150） | 中文 rubric judge，主榜 `G ≥ 0.8` |
| **C 多工具编排** | 在有状态订单沙箱里改地址 / 退款 / 用券 | 40 条（trial 8 / dev 16 / blind 16） | 状态 delta + 回复质量 |

**为什么中文 agent 需要它**：公开榜单多在英文、封闭题、无状态。
这个基准的三件事在中文圈里是缺的 —— 中文办公数据题、中文知识库检索里
**「证据不足 / 不可回答」作为正确答案**（而不是扣分）、以及**多工具写操作后核对状态**。

---

## Quickstart

```bash
git clone <本仓库地址>
cd zh-agent-bench
uv sync
```

### 跑 A 场景（不需要 key，10 秒内）

**默认跑 train（96 条 dev 调试集）**；heldout 是封存口径，要显式加 `--allow-heldout`：

```bash
uv run python -m zh_agent_bench                      # 默认 = train（dev 调试集）
uv run python -m zh_agent_bench --heldout-only --allow-heldout   # 官方口径（封存集）
```

不加 `--allow-heldout` 就 exit 2，**不会静默改跑 train** —— 静默降级会让人拿一份
dev 分数当成官方成绩发出去，没人看得出来。

```
# schema=1.1 tool_contract=1.0 split=heldout n=24 agent=reference-solver
====================================================
zh-agent-bench  M1 leaderboard
====================================================
agent            : reference-solver
tool contract    : 1.0
score            : 100.00%
correct          : 24/24
```

这是**参考求解器**在跑，证明「任务可解、判分链路通」，不是被测 agent 的成绩。

### 跑被测 agent（A 场景，需 key）

```bash
export DEEPSEEK_API_KEY=sk-xxx        # Windows: set DEEPSEEK_API_KEY=sk-xxx
uv run python -m zh_agent_bench --agent llm --heldout-only --allow-heldout \
    --out-records benchmarks/office/heldout
```

`--out-records` 给**目录**时按 agent 名派生子目录（→ `benchmarks/office/heldout/llm-agent-deepseek-chat/records.jsonl`）；
给 `.jsonl` 结尾的**文件路径**时原样写。

> ⚠️ **先确认 key 生效再跑**。无 key 时本命令**不报错、exit 0**，
> 会把 24 条全判错落进产物目录 —— 而榜单下一次就会把这堆错当分数读出去。
> 见 [已知短板](#已知短板--边界) 第 1 条。

### 跑 B / C 场景并出统一榜（需 key）

```bash
# B 场景（知识检索）—— 默认落 benchmarks/qa/dev/<agent>/
uv run python -m zh_agent_bench.eval_run --split dev

# C 场景（多工具编排）—— 默认落 benchmarks/orchestration/dev/<agent>/
uv run python -m zh_agent_bench.eval_run --split orchestration --preset dev

# 一张表出 A/B/C 三场景（一 agent 一行）
uv run python -m zh_agent_bench.leaderboard --scenario all --b-split dev --c-preset dev
```

榜单只做聚合、不跑 agent —— 扫 `benchmarks/<场景>/<档>/*/records.jsonl`，**一 agent 一行**，
两个 agent 跑同一档不会互相覆盖。`--agent <目录名>` 是过滤器（默认全出）。

### 跑测试

```bash
uv run pytest -q
```

---

## 榜单

真数字见 [`benchmarks/leaderboard/leaderboard.md`](benchmarks/leaderboard/leaderboard.md)（唯一真相源，由上面最后一条命令生成）。

| agent | A 场景 | B 场景 | B G | B F | B T | C 场景 | 样本数 | 降级/失败 |
|---|---|---|---|---|---|---|---|---|
| `llm-agent-deepseek-chat` | 24/24 = 100.00% | 24/120 = 20.00% | 0.5108 | 0.7299 | 1.0000 | 12/16 = 75.00% | 160 | 2（B 2 条 agent 失败） |
| `llm-agent-gpt-6-astra` | 24/24 = 100.00% | 73/120 = 60.83% | 0.7744 | 0.6296 | 1.0000 | 12/16 = 75.00% | 160 | 10（B 9，C 1） |

**B 场景是两模型差距最大的地方**：deepseek-chat **20%** vs gpt-6-astra **60.83%**（3 倍）；A（100%）和 C（75%）两者一致。差别全在「肯不肯说『库里没有这条』」——这正是 B 场景要测的能力。

> **产物自带署名**：布局是 `benchmarks/<场景>/<档>/<agent>/records.jsonl`，
> `records.jsonl` 每行带 `agent`/`judge` 字段。两个 agent 跑同一档不会互相覆盖，榜单**一 agent 一行**；
> `--agent <目录名>` 只是过滤器（默认全出）。旧布局（路径里没有 agent 目录）的产物显示「未发现产物」，
> **不静默当成 0**。

**「未跑」≠ 0 分**：某场景没有产物时该列显式写「未跑」，
0 分会被读成「跑了得零分」，那是两条完全不同的结论。

---

## 三个场景

### A 场景 · 办公数据（客观判分）

**考什么**：单轮工具调用 —— 把中文指令翻译成一次正确的表格查询/统计/排序/格式转换。
4 类 × 30 条，train 96 / heldout 24。

| 类别 | 任务 | 答案类型 | 比较算子 |
|---|---|---|---|
| `query` | 查某人部门 | 字符串 | `exact`（去首尾空白） |
| `stats` | 算某列平均值 | 十进制字符串 | `decimal_exact` |
| `organize` | 按销售额降序取前 3 名 | 有序姓名列表 | `ordered_list_exact` |
| `format` | CSV 转 JSON | JSON 对象 | `json_canonical`（键序无关，数组顺序有关） |

**怎么判**：4 个确定性工具返回**结构化结果**（不预制答案），
判分器按上表的算子逐条比对。**铁律：烂答案绝不抛异常** —— 一律判错并写明理由，
否则一批烂答案会把整批评测挂掉。

**规模**：heldout 24 条是榜单口径；train 96 条供调试。heldout 已封存 —— 默认命令跑 train，
要跑官方集得显式加 `--allow-heldout`（见 [已知短板](#已知短板--边界) 第 6 条）。

**防污染**：实体池先整体对半切成 train 段 / heldout 段再各自生成任务 ——
同类别内 heldout 的人名、城市、公司、商品绝不与 train 相交。榜单只用 heldout。

**金额铁律**：一切金额以**整数分**存储，答案用 `Decimal` + `ROUND_HALF_UP` 算出两位小数。
原生 `float` 不进入金额与答案字段。

### B 场景 · 知识检索（rubric judge）

**考什么**：从企业知识库取证并作答，且**该说「不知道」时要说不知道**。
4 类各 30 条（dev 120 条）：单跳 / 多跳 / 证据不足 / 不可回答。

后两类是这场景的核心 —— 标准答案就是「库里没有这条 / 先澄清」，
**硬答反而判错**。这是 agent 真实失败的高发区（见榜单短板）。

**怎么判**：每题带**逐条 rubric 评分点** + 知识库原文，
中文 judge 只对每个评分点做三档判断（支持 / 矛盾 / 证据不足），
**聚合由程序做** —— 让模型直接给一个总分，丢分点就永远查不出来了。

| 维度 | 含义 | dev 实测 |
|---|---|---|
| `G` | rubric 加权得分（主榜依据） | 0.511 |
| `F` | 忠实度：断言有没有被证据支持 | 0.730 |
| `T` | 工具协议合规（工具用对了吗） | 1.000 |

**主榜门槛 `G ≥ 0.8`，冻结在代码里**（`judge.PASS_THRESHOLD_G`）。
阈值一旦按榜单成绩回调，分数就不再可比 —— 观察到 0.8 偏严（多跳题 G=0.775 被判失败），留待观察。

**规模**：trial 60 / dev 120 / blind 150，按环境 seed 组隔离。

**防污染**：三档由不同 seed 组生成，题面与槽位不重叠。

### C 场景 · 多工具编排（状态 delta）

**考什么**：在**有状态**订单沙箱（orders / inventory / coupons）里做写操作，
4 类各 10 条 = 40 条。带陷阱：非 `paid` 的单不能退、券门槛不达标不能抵。

**怎么判**：**状态 delta 两段** —— 只比对「受影响订单/券的目标状态」，
不比对全量快照（否则「顺手做对的事」会被判错）：

1. `orders` / `coupons` 该变成什么；
2. `must_not_change` 什么必须**不**变 —— 陷阱任务（不该退的退了）靠这一段才判得出错。

再加一段**回复质量**（`reply_ok`，门槛 `G ≥ 0.8`，与 B 的门槛分开冻结）。
主榜 = `delta_ok ∧ reply_ok`。

**规模**：trial 8 / dev 16 / blind 16，按环境 seed 组隔离。

**防污染**：每题一份**独立沙箱**（sub_seed 由 `根 seed + 类型位 + 下标位` 整数派生）。
上一题退掉的单会让下一题的陷阱物理性失效，分数随执行顺序变 —— 所以绝不复用。

---

## 判分口径（门槛写在这里，不藏在代码里）

| 场景 | 主榜口径 | 门槛来源 |
|---|---|---|
| A | `exact` 逐字相等（按上表四种算子） | 无阈值 |
| B | `G ≥ 0.8` | `judge.PASS_THRESHOLD_G` |
| C | `delta_ok ∧ reply_ok`，其中 `reply_ok` 要求 `G ≥ 0.8` | `judge.PASS_THRESHOLD_C` |

- **降级与失败单列，不计入正确率**。judge 崩了不是 agent 答错，混进正确率会让榜单替 judge 背锅。
- **榜单的 C 列是现算的**，不读产物里那个 `correct` 字段 —— 口径写在榜单这一层，
  读上一任写下的结论等于把口径藏起来，将来改判分口径的人不会想起来回来改榜单。

---

## 校准证据

判分器本身可不可信，要有数，不能只说「我们校准过了」。

### B 场景（中文 rubric judge）

| 检验 | 结果 | 怎么测的 |
|---|---|---|
| **双标一致性** | 标注者间一致率 **88.9%**（248/279 个评分点），**κ = 0.801** | 两个独立标注者（`gpt-6-astra` / `deepseek-chat`）复用同一套 judge prompt，只换输出契约 |
| **judge vs 共识** | 一致 **86.4%**、MAE 0.019；通过翻转 6 条、大幅分歧 1 条 | judge 单独判，与两人共识（中点）比 |
| **位置偏置** | **5.0%**（ΔG 0.025）—— 未检出偏置 | rubric 逆序重排，看 G 翻转率 |
| **长度偏置** | **1.7%**（ΔG 0.009）—— 未检出偏置 | 追加不含新事实的客套话，看 G 是否上升 |
| **抗答非所问** | gap **0.835**、抗空话 **100%** | 用本任务证据 id 做骨架构造空话答案 |
| **校准集自检** | example / corrupted **各 100%** | 已知答案的构造样本 |

κ > 0.8 属「高度一致」。分歧最大的一条是 `qa-trial-kb-g1-06`
（judge 0.850 vs 共识 0.550）—— 单点分歧属正常范围，但已记录在案。

### C 场景（状态 delta 判分器）

**oracle 上界 40/40 = 100%**（真 key）。oracle 是「知道答案的正确解」，
验证的是「判分器认不认正确解」，不是「会不会解题」——
与 A 场景的参考求解器同性质。5 倍规模下上界仍无缺口。

---

## 已知短板 / 边界

1. **无 key 时跑分命令不报错，exit 0，产物全错。**
   `python -m zh_agent_bench --agent llm` 与 `eval_run` 都会把 API 失败转成
   「答不出来」（设计如此：评测工具最不该犯的错是让一批跑挂的任务挂掉整批），
   但代价是**一份全错的产物能悄无声息地覆盖上一份好产物**。
   跑之前先 `python -c "import os; print(bool(os.environ.get('DEEPSEEK_API_KEY')))"` 确认。

2. **deepseek-chat 在 B 场景只有 20%（24/120），gpt-6-astra 是 60.83%（73/120）—— 这是模型差距，不是判分器冤。**
   deepseek 分类型（从 `benchmarks/qa/dev/llm-agent-deepseek-chat/records.jsonl` 统计，榜单本身不列分类型）：
   单跳 6/30、多跳 7/30、证据不足 11/30、**不可回答 0/30**。
   「不可回答」全灭说明模型倾向硬答、不肯说「库里没有这条」——
   这正是这场景要测的能力，也是本基准与「只考检索」的基准的差别所在。
   A（单轮查询，两者都 100%）→ C（多工具编排，两者都 75%）→ B（取证与承认证据不足，**差 3 倍**）：
   **B 是唯一的区分点**，另两个场景目前分不出这两个模型。

3. **A 场景 100% 是 deepseek-chat 的真实成绩**（A 是单轮表格查询 + `exact` 比对，全对合理）。
   产物落 `benchmarks/office/heldout/<agent>/records.jsonl`，agent 名从目录名与记录里的 `agent` 字段两处可读。

4. **C 场景 40 条规模小。** 每类 10 条（trial 8 + dev 16 + blind 16），
   按 20/40/40 比例切分而不是每档 ≥10（数学上互斥，保比例优先）。
   单类准确率的置信区间宽，**只看整档趋势，别单看某一类的百分比**。

5. **blind 已封存（开发流程碰不到）。** 四个入口（`--split blind` / `--preset blind` /
   `--b-split blind` / `--c-preset blind`）默认 **exit 2**，要显式加 `--allow-blind` 才放行。
   `benchmarks/blind-manifest.json` 记着盲测产物的 sha256，`--verify-blind-manifest` 可核验是否被改过；
   改盲测产物必须同步更新清单，否则校验红。

6. **A 场景的 heldout 也已封存。** 入口（`--heldout-only` / `--split heldout`）默认 **exit 2**，
   要显式加 `--allow-heldout`；**默认命令跑的是 train**（dev 调试集），被拒时**不静默降级**。
   `benchmarks/heldout-manifest.json` 记着 heldout 任务集与产物的 sha256，
   `--verify-heldout-manifest` 可核验（`--write-heldout-manifest` 更新）。
   > 闸只加在 CLI 上 —— 作为库调 `load_tasks(..., Split.HELDOUT)` 不经闸（见「接自己的 agent」）。

7. **多 agent 横评已支持，但「同 agent 换 judge」不进横评表。**
   布局 `benchmarks/<场景>/<档>/<agent>/` 让两个 agent 互不覆盖、榜单一行一个 agent；
   但同一 agent 换 judge（如 gpt-6-astra 双判）的产物是**旧布局**、
   单独放在 `benchmarks/<档>-results-gpt-judge/records.jsonl`，不在 `<档>/<agent>/` 下 ——
   横评表要的是「一 agent 一行」，换 judge 属于同一个 agent 的另一种量法，
   混进去会被误读成另一个 agent。榜单**完全不扫这个目录**
   （它只扫 `<档>/<agent>/`，而它连 `<档>/` 下都没躺），
   所以它既不进榜也不标「未发现产物」—— **两份的对比需人工对读 `records.jsonl`**。

8. **judge 与被测 agent 在 B/C 场景是同一个模型。** 同模型自评有偏置，
   偏差实验是后续单的事。当前模型 id 已记在各自报告里，至少分得清是谁判的。

---

## 目录结构

```
benchmarks/
  tasks/          A 场景数据集（train / heldout，按类别分文件）
  office/         A 场景跑分产物（heldout/<agent>/records.jsonl）
  qa/             B 场景任务集（trial/dev/blind）+ 跑分产物（<档>/<agent>/）
  orchestration/  C 场景任务集 + 沙箱跑分产物（<档>/<agent>/）
  leaderboard/    统一榜单产物（leaderboard.md / .json）
  calibration/    A 场景 judge 校准集（example + corrupted，不是能力分）
src/zh_agent_bench/
  tasks/          任务契约与数据集生成
  tools/          4 个确定性中文办公工具 + 契约校验
  sandbox.py      C 场景有状态沙箱
  judge.py        三场景判分器（含 rubric judge 与门槛常量）
  harness.py      轨迹 → 判分 → 榜单闭环
  agents/         reference（确定性求解器）/ llm（function-calling 循环）
  leaderboard.py  A/B/C 统一榜单
  eval_run.py     B / C 场景跑分器
  cli.py          A 场景跑分入口（`python -m zh_agent_bench`）
```

设计口径与已知坑写在各源文件顶部的 docstring 与本文的「判分口径」「已知短板」两节里。
模块级文档在各源文件顶部的 docstring 里。

### 重新生成 A 场景数据集

```bash
uv run python -m zh_agent_bench.tasks.build --seed 42 --per-category 30 \
    --output-dir benchmarks/tasks
```

同 seed 逐字节一致。词表是人工编写的虚构办公实体，不含真实个人信息。

---

## 接自己的 agent

实现一个函数，两个参数，返回最终答案：

```python
def solve(task, tools) -> answer:
    # task      —— Task 实例，看 instruction / initial_data / tool_names
    # tools     —— 本任务声明的工具集合，形如 {"table.query": callable}
    # 返回       —— 字符串 / 数字 / 列表 / dict，按 task.answer_spec 的类型给
    ...
```

`tools[name](payload)` 返回工具的**结构化结果字典**，不做任何预格式化 ——
防的是「工具直接把答案喂回去」。参考实现见 `src/zh_agent_bench/reference.py`。

**不要读 `task.ground_truth`**，那是给判分器用的。

```python
from pathlib import Path

from zh_agent_bench.cli import load_tasks
from zh_agent_bench.harness import evaluate, leaderboard
from zh_agent_bench.tasks.schema import Split

tasks = load_tasks(Path("benchmarks/tasks"), Split.HELDOUT)  # 要 Path，传 str 会炸
results = evaluate(tasks, solve)          # solve 必须是 (task, tools) -> answer
print(leaderboard(results))
```

---

## 许可

MIT，见 [`LICENSE`](LICENSE)。

数据集词表为虚构办公实体，不含真实个人信息。
