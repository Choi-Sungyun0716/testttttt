from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class ShortTermMemoryManager:
    """간단한 단기 메모리 저장소. 최근 N개의 대화만 유지한다."""

    def __init__(
        self,
        storage_path: str | Path = "data/short_term_memory.json",
        max_conversations: int = 10,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.max_conversations = max_conversations
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def _load(self) -> Dict[str, Any]:
        if not self.storage_path.exists():
            return {"conversations": []}
        try:
            return json.loads(self.storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"conversations": []}

    def _save(self, data: Dict[str, Any]) -> None:
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    def record_interaction(
        self,
        session_id: str,
        user_query: str,
        intent: str,
        hitl_required: bool,
        hitl_reason: str | None,
        tools: List[Dict[str, Any]],
    ) -> None:
        """단일 대화를 저장하고 최대 개수를 유지한다."""

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "user_query": user_query,
            "intent": intent,
            "tools": tools,
            "hitl_required": hitl_required,
            "hitl_reason": hitl_reason,
        }

        data = self._load()
        conversations = data.setdefault("conversations", [])
        conversations.append(entry)
        # 최근 max_conversations 개만 유지
        if len(conversations) > self.max_conversations:
            conversations[:] = conversations[-self.max_conversations :]

        self._save(data)


__all__ = ["ShortTermMemoryManager"]

