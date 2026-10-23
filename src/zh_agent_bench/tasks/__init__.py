"""任务 schema 与构造器的公开入口。"""

from .build import (
    ANSWER_SPECS,
    build_category,
    build_dataset,
    heldout_count,
    split_pools,
    write_dataset,
)
from .schema import (
    A_SCENE_CATEGORIES,
    A_SCENE_SCHEMA_VERSION,
    B_SCENE_COMPARISON,
    LEGAL_COMPARISONS,
    SCHEMA_VERSION,
    AnswerKind,
    AnswerSpec,
    Category,
    Comparison,
    Normalization,
    Split,
    Task,
)

__all__ = [
    "SCHEMA_VERSION",
    "A_SCENE_CATEGORIES",
    "A_SCENE_SCHEMA_VERSION",
    "B_SCENE_COMPARISON",
    "LEGAL_COMPARISONS",
    "AnswerKind",
    "AnswerSpec",
    "Category",
    "Comparison",
    "Normalization",
    "Split",
    "Task",
    "ANSWER_SPECS",
    "build_category",
    "build_dataset",
    "heldout_count",
    "split_pools",
    "write_dataset",
]