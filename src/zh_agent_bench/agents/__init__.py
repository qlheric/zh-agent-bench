"""agent 集合 —— benchmark 真正评的对象。

M1 只有 :mod:`reference`（确定性求解器）一个，它证明「任务可解」。
M2 起，:mod:`llm` 接真 LLM，证明「**一个 agent** 能不能读指令、调对工具、给出可判分的答案」——
这才是这个 benchmark 的存在理由。
"""

from .llm import LLM_AGENT_NAME, LlmAgent, make_llm_agent

__all__ = ["LLM_AGENT_NAME", "LlmAgent", "make_llm_agent"]
