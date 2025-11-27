from __future__ import annotations

from .base_master import BaseLLMMaster


class QAMaster(BaseLLMMaster):
    """LLM 기반 Q&A 도메인 마스터."""

    def __init__(self, llm) -> None:
        super().__init__(master_name="qa_master", llm=llm)


__all__ = ["QAMaster"]
