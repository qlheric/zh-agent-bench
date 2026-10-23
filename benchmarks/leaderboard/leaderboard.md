# zh-agent-bench 统一榜单（A/B/C 三场景）

> 口径：A 场景按 `exact` 逐字相等；B 场景按主榜门槛 `G ≥ 0.8`；C 场景按 `delta_ok ∧ reply_ok`（回复点 `G ≥ 0.8`）。
> 「未跑」= 该场景没有跑分产物；**不等于 0 分**。「未发现产物」= 有产物但是旧布局（路径里没有 agent 目录），猜不出是谁跑的，不进榜。judge 降级与 agent 失败单列，不计入正确率。

| agent | A 场景正确率 (exact 逐字相等) | B 场景正确率 (G≥0.8) | B G | B F | B T | C 场景正确率 (delta_ok∧reply_ok, 回复点 G≥0.8) | C delta_ok | C reply_ok | 样本数 | 降级/失败 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llm-agent-deepseek-chat | 24/24 = 100.00% | 24/120 = 20.00% | 0.5108 | 0.7299 | 1.0000 | 12/16 = 75.00% | 16/16 | 12/16 | 160 | 2｜B 2 |
| llm-agent-gpt-6-astra | 24/24 = 100.00% | 73/120 = 60.83% | 0.7744 | 0.6296 | 1.0000 | 12/16 = 75.00% | 15/16 | 12/16 | 160 | 10｜B 9，C 1 |

## 样本来源

| agent | 场景 | 产物 | 样本数 |
|---|---|---|---|
| llm-agent-deepseek-chat | A | `benchmarks/office/heldout/llm-agent-deepseek-chat/records.jsonl` | 24 |
| llm-agent-deepseek-chat | B | `benchmarks/qa/dev/llm-agent-deepseek-chat/records.jsonl` | 120 |
| llm-agent-deepseek-chat | C | `benchmarks/orchestration/dev/llm-agent-deepseek-chat/records.jsonl` | 16 |
| llm-agent-gpt-6-astra | A | `benchmarks/office/heldout/llm-agent-gpt-6-astra/records.jsonl` | 24 |
| llm-agent-gpt-6-astra | B | `benchmarks/qa/dev/llm-agent-gpt-6-astra/records.jsonl` | 120 |
| llm-agent-gpt-6-astra | C | `benchmarks/orchestration/dev/llm-agent-gpt-6-astra/records.jsonl` | 16 |

## 旧布局产物（未纳入榜单）

无。
