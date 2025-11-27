from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .catalog import MASTER_SPECS, MasterSpec


class PlannedMasterTask(BaseModel):
    """Single master call that the supervisor expects to run."""

    master: str
    intent: str
    reason: str
    priority: int = Field(
        default=0, description="Lower number means the task should run earlier."
    )
    expected_tools: List[str] = Field(
        default_factory=list,
        description="Optional suggestion of tools the master will likely need.",
    )


class SupervisorPlan(BaseModel):
    """Full scenario output produced by the supervisor."""

    domain: str = "general"
    intent: str = ""
    master_tasks: List[PlannedMasterTask] = Field(default_factory=list)
    hitl_required: bool = False
    hitl_reason: Optional[str] = None


class SupervisorRouter:
    """LLM-driven supervisor that plans which masters should run."""

    def __init__(
        self,
        llm,
        master_specs: Dict[str, MasterSpec] | None = None,
    ) -> None:
        if llm is None:
            raise ValueError("llm must be provided to initialize the supervisor")

        self.llm = llm
        self.master_specs = master_specs or MASTER_SPECS

        self._parser = JsonOutputParser(pydantic_object=SupervisorPlan)
        catalog_text = self._format_master_catalog()

        system_prompt = (
            "You are the supervisor for an enterprise assistant. "
            "Determine which masters must run (one or multiple) and in what order. "
            "If any mandatory information is missing, set hitl_required=true and explain."
        )

        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "system",
                    (
                        "Known masters and their intents/tools:\n"
                        f"{catalog_text}\n"
                        "Always return valid JSON matching the provided format instructions."
                    ),
                ),
                (
                    "user",
                    (
                        "User query: {query}\n"
                        "Current state snapshot:\n{state_summary}\n"
                        "Format instructions:\n{format_instructions}"
                    ),
                ),
            ]
        )
        self._chain = self._prompt | self.llm | self._parser

    def plan(
        self,
        query: str,
        state: Dict[str, Any] | None = None,
        domain_hint: Optional[str] = None,
    ) -> SupervisorPlan:
        """Plan the overall scenario for the provided query."""

        state_summary = json.dumps(state or {}, ensure_ascii=False, default=str)
        payload = {
            "query": query,
            "state_summary": state_summary,
            "format_instructions": self._parser.get_format_instructions(),
        }

        raw_plan = self._chain.invoke(payload)
        if isinstance(raw_plan, SupervisorPlan):
            plan = raw_plan
        else:
            plan = SupervisorPlan(**raw_plan)

        if domain_hint:
            plan.domain = domain_hint
        elif not plan.domain:
            plan.domain = self._infer_domain_from_tasks(plan.master_tasks)

        if not plan.intent and plan.master_tasks:
            plan.intent = plan.master_tasks[0].intent

        return plan

    def _format_master_catalog(self) -> str:
        lines: List[str] = []
        for spec in self.master_specs.values():
            lines.append(
                f"- {spec.name} (domain: {spec.domain})\n"
                f"  Description: {spec.description}\n"
                f"  Intents: {', '.join(spec.intents)}\n"
                f"  Tools: {', '.join(spec.tool_chain)}\n"
                f"  HitL triggers: {', '.join(spec.hitl_triggers)}"
            )
        return "\n".join(lines)

    def _infer_domain_from_tasks(self, tasks: List[PlannedMasterTask]) -> str:
        for task in tasks:
            spec = self.master_specs.get(task.master)
            if spec:
                return spec.domain
        return "general"


__all__ = ["SupervisorRouter", "SupervisorPlan", "PlannedMasterTask"]
