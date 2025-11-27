from __future__ import annotations

from .base_master import BaseLLMMaster


class DocumentMaster(BaseLLMMaster):
    """LLM 기반 문서 도메인 마스터."""

    def __init__(self, llm) -> None:
        super().__init__(master_name="document_master", llm=llm)


__all__ = ["DocumentMaster"]
