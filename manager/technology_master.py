from __future__ import annotations

from .base_master import BaseLLMMaster


class TechnologyMaster(BaseLLMMaster):
    """LLM 기반 기술 도메인 마스터."""

    def __init__(self, llm) -> None:
        super().__init__(master_name="technology_master", llm=llm)


__all__ = ["TechnologyMaster"]
