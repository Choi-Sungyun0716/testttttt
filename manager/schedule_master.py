from __future__ import annotations

from .base_master import BaseLLMMaster


class ScheduleMaster(BaseLLMMaster):
    """LLM 기반 스케줄 도메인 마스터."""

    def __init__(self, llm) -> None:
        super().__init__(master_name="schedule_master", llm=llm)


__all__ = ["ScheduleMaster"]

