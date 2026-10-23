"""``python -m zh_agent_bench`` 的入口。只做转发，实现在 :mod:`zh_agent_bench.cli`。"""

from __future__ import annotations

from .cli import main

raise SystemExit(main())
