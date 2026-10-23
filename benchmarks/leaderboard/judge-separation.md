# judge 与被测 agent 分离 —— 双判对比（#3）

> 目的：B/C 场景原本是 **deepseek-chat 判 deepseek-chat 的答案**（同模型自评）。
> 本报告把 judge 换成 `gpt-6-astra`（走 LLMAPI 通道），量一次差。
> **两套数都报，不挑好看的。**

- 被测 agent：`llm-agent:deepseek-chat`（两遍都是它，`--model` 未变）
- judge A：`deepseek-chat`（`benchmarks/qa/dev-results`、`benchmarks/orchestration/dev-results`）
- judge B：`gpt-6-astra`（`benchmarks/qa/dev-results-gpt-judge`、`benchmarks/orchestration/dev-results-gpt-judge`）
- 规模：B dev 120 条、C dev 16 条

## 双判分数表

| 场景 | judge | 正确率 | G 均值 | F | T | capped | 失败/降级 |
|---|---|---|---|---|---|---|---|
| B dev | deepseek-chat | **24/120 = 20.00%** | 0.511 | 0.730 | 1.000 | 31 | 2 / 0 |
| B dev | gpt-6-astra | **11/120 = 9.17%** | 0.531 | 1.000 | 1.000 | 34 | 1 / 1 |
| C dev | deepseek-chat | **12/16 = 75.00%** | 0.844 | — | — | — | 0 / 0 |
| C dev | gpt-6-astra | **12/16 = 75.00%** | 0.750 | — | — | — | 0 / 0 |

## task 级不一致（共同 task_id 对齐）

| 场景 | 共同 | deepseek 对/gpt 错 | gpt 对/deepseek 错 | 净差 |
|---|---|---|---|---|
| B | 120 | **20** | **7** | gpt 严格 13 条 |
| C | 16 | 0 | 0 | 0 |

B 的不一致集中在 `kb-d3`（单跳）与 `kb-d4`（多跳）两个知识库后缀（前 5 个例子：`qa-dev-kb-d3-11`、`qa-dev-kb-d4-01/04/05/11`）。

## 结论（如实，含 caveat）

1. **B 场景：gpt-judge 明显更严**（11 vs 24，方向一致——deepseek 自评偏高）。这与「同模型自评有偏」的假设同向。
2. **C 场景：两个 judge 结论完全一致**（12/16，0 翻转）。C 的主榜是 `delta_ok ∧ reply_ok`，而 **delta 三段是机器判、与 judge 模型无关**；两个 judge 只在 `reply_ok` 上可能分歧，本批 16 条没出现。
3. **⚠️ 本次双判是「同批重跑」不是「同批答案重判」**：两遍各自跑了一遍 agent，LLM 有采样随机性，所以 27 条 B 不一致里**混了 agent 方差**，不能全算到 judge 头上。
   - 同答案的纯 judge 差，以 TASK-017 的双标为准：**judge vs 共识 86.4%**（即纯 judge 差约 13.6%），本次「重跑」的 22.5% 不一致 = 13.6% judge + 约 9% agent 随机。
4. **未解决**：`records.jsonl` 既无 agent 字段也无 judge 字段——两套产物靠目录名/`report.md` 区分。这与「多 agent 横评」是同一个根问题，需在 #2 里连路径布局一起改。

## 复现

```bash
export DEEPSEEK_API_KEY=<deepseek key>
export LLMAPI_API_KEY=<LLMAPI key>
export LLMAPI_BASE_URL=https://llmapi.pro/v1

# judge A（deepseek）
uv run --project . python -m zh_agent_bench.eval_run --split dev \
    --out benchmarks/qa/dev-results --save-answers
uv run --project . python -m zh_agent_bench.eval_run --split orchestration \
    --preset dev --out benchmarks/orchestration/dev-results

# judge B（gpt-6-astra）
uv run --project . python -m zh_agent_bench.eval_run --split dev \
    --out benchmarks/qa/dev-results-gpt-judge --save-answers \
    --judge-model gpt-6-astra --judge-base-url "$LLMAPI_BASE_URL" --judge-key-env LLMAPI_API_KEY
uv run --project . python -m zh_agent_bench.eval_run --split orchestration \
    --preset dev --out benchmarks/orchestration/dev-results-gpt-judge \
    --judge-model gpt-6-astra --judge-base-url "$LLMAPI_BASE_URL" --judge-key-env LLMAPI_API_KEY
```
