from __future__ import annotations

from .base_master import BaseLLMMaster


class EmailMaster(BaseLLMMaster):
    """LLM 기반 이메일 도메인 마스터."""

    def __init__(self, llm) -> None:
        super().__init__(master_name="email_master", llm=llm)


__all__ = ["EmailMaster"]
