from __future__ import annotations

import json
import logging
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
    sub_query: str = Field(
        default="",
        description="Subset of the user query relevant to this master. Should be short and specific.",
    )
    priority: int = Field(
        default=0, description="Lower number means the task should run earlier."
    )
    expected_tools: List[str] = Field(
        default_factory=list,
        description="Optional suggestion of tools the master will likely need.",
    )


class SubQueryItem(BaseModel):
    master: str
    sub_query: str


class SubQuerySplit(BaseModel):
    items: List[SubQueryItem] = Field(default_factory=list)


class SupervisorPlan(BaseModel):
    """Full scenario output produced by the supervisor."""

    domain: str = "general"
    intent: str = ""
    master_tasks: List[PlannedMasterTask] = Field(default_factory=list)
    hitl_required: bool = False
    hitl_reason: Optional[str] = None


class SupervisorRouter:
    """LLM-driven supervisor that plans which masters should run."""

    _REWRITE_HINTS: Dict[str, str] = {
        "email_master": (
            "Include recipient names/emails and the purpose/time of the meeting in the message. "
            "Do not mention booking rooms."
        ),
        "schedule_master": (
            "Include meeting time, date, location requirements, and participant count for the reservation. "
            "Do not mention sending emails or messages."
        ),
        "document_master": "Include document type, purpose, and any recipients mentioned.",
        "qa_master": "Include the exact question or topic the user wants to search.",
        "technology_master": "Include keywords or IDs the user wants to research.",
    }

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
                        "Always return valid JSON matching the provided format instructions. "
                        "For each planned task include a 'sub_query' field that contains ONLY the portion "
                        "of the user request that this master should handle. "
                        "The sub_query must include all relevant entities (people, time, quantities, etc.) "
                        "needed for that master to act without referring back to the full query, "
                        "and must exclude instructions that belong to other masters. "
                        "Example:\n"
                        'User query: "팀장에게 3시 회의 공유 메일 보내고 5명 회의실 잡아줘"\n'
                        'email_master sub_query: "팀장에게 3시 회의 공유 메일을 보내 줘"\n'
                        'schedule_master sub_query: "내일 3시에 5명이 사용할 회의실을 예약해 줘"\n'
                        "Follow this pattern."
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

        self._subquery_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You rewrite task-specific instructions. "
                    "For the given master, produce a standalone command that includes "
                    "all entities (people, time, counts, locations, etc.) from the original query "
                    "that are relevant to that master, even if those entities were mentioned elsewhere. "
                    "Do not mention tasks intended for other masters. "
                    "The rewritten instruction must stand alone and never omit critical context. "
                    "Respond with plain text only.",
                ),
                (
                    "user",
                    (
                        "Full user query: {full_query}\n"
                        "Master name: {master}\n"
                        "Master intent: {intent}\n"
                        "Hint: {hint}\n"
                        "Existing sub_query (may be empty): {existing_sub_query}\n"
                        "Rewrite the instruction so that this master can execute it without "
                        "needing context from other masters."
                    ),
                ),
            ]
        )
        self._subquery_chain = self._subquery_prompt | self.llm

        self._split_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a planner that splits a user query into master-specific instructions. "
                    "For each master, return a short instruction that includes all necessary entities "
                    "(people, times, counts, etc.) relevant to that master only. "
                    "Each instruction must repeat any critical context (dates, times, participants) even if it was stated elsewhere in the full query.",
                ),
                (
                    "user",
                    "User query: {full_query}\n"
                    "Tasks:\n{tasks}\n"
                    "Only include masters that appear in the tasks list.\n"
                    "{format_instructions}",
                ),
            ]
        )
        self._split_parser = JsonOutputParser(pydantic_object=SubQuerySplit)
        self._split_chain = self._split_prompt | self.llm | self._split_parser

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

        self._ensure_sub_queries(query, plan)
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

    def _ensure_sub_queries(self, full_query: str, plan: SupervisorPlan) -> None:
        """Ensure each task has a well-formed sub_query."""

        for task in plan.master_tasks:
            task.sub_query = task.sub_query or full_query


__all__ = ["SupervisorRouter", "SupervisorPlan", "PlannedMasterTask"]
