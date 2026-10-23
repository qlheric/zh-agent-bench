# judge 校准报告（最小闭环）

- 任务数：16（--per-type 4 --seed 42）
- 答案数：32（每条任务 2 个：example + 错误变体）
- 主榜门槛 G：0.8

## 混淆表（正类 = 判对）

|  | judge 判对 | judge 判错 |
|---|---|---|
| **预期对** | TP 11 | FN 5 |
| **预期错** | FP 3 | TN 13 |

一致率：75.00%（(11+13)/32）

> FP = 坏答案被判对（最危险）；FN = 好答案被判错（会误伤 agent）。

## 按类型切片

| type | n | TP | TN | FP | FN | agreement |
|---|---|---|---|---|---|---|
| 单跳 | 8 | 4 | 4 | 0 | 0 | 100.00% |
| 多跳 | 8 | 4 | 4 | 0 | 0 | 100.00% |
| 证据不足 | 8 | 1 | 1 | 3 | 3 | 25.00% |
| 不可回答 | 8 | 2 | 4 | 0 | 2 | 75.00% |

## 判别力（G 分带）

- example：1.0000
- corrupted：0.6125
- gap（example - corrupted）：+0.3875

## 降级计数

| kind | count |
|---|---|
| llm_failure | 0 |
| invalid_score | 0 |
| missing_criterion | 0 |

降级合计：0 / 32。（无降级）

## 不一致样本

其中 **5 条是 G 已达门槛却被判错** —— 问题在 F 的 claim 口径，不在 rubric 判分。

| task_id | type | kind | judge_G | F | n_unsup | judge | expected | reason |
|---|---|---|---|---|---|---|---|---|
| qa-0009 | 证据不足 | example | 1.0 | 0.333 | 2 | 错 | 对 | 逐条 rubric 判定完成 |
| qa-0010 | 证据不足 | example | 1.0 | 0.667 | 1 | 错 | 对 | 逐条 rubric 判定完成 |
| qa-0010 | 证据不足 | corrupted | 0.85 | 1.000 | 0 | 对 | 错 | 逐条 rubric 判定完成 |
| qa-0011 | 证据不足 | corrupted | 0.85 | 1.000 | 0 | 对 | 错 | 逐条 rubric 判定完成 |
| qa-0012 | 证据不足 | example | 1.0 | 0.750 | 1 | 错 | 对 | 逐条 rubric 判定完成 |
| qa-0012 | 证据不足 | corrupted | 0.925 | 1.000 | 0 | 对 | 错 | 逐条 rubric 判定完成 |
| qa-0014 | 不可回答 | example | 1.0 | 0.667 | 1 | 错 | 对 | 逐条 rubric 判定完成 |
| qa-0016 | 不可回答 | example | 1.0 | 0.667 | 1 | 错 | 对 | 逐条 rubric 判定完成 |
