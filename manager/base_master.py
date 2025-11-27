from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .catalog import MASTER_SPECS, TOOL_SPECS, MasterSpec, ToolSpec


class ToolPlan(BaseModel):
    """Desired tool execution plan for a single master step."""

    tool: str = Field(..., description="Tool name to execute")
    action: str = Field(..., description="Short natural language description")
    reason: str = Field(..., description="LLM rationale for choosing this tool")
    inputs_to_collect: List[str] = Field(
        default_factory=list,
        description="State fields or info that must exist before running the tool",
    )
    expected_outputs: List[str] = Field(
        default_factory=list, description="Fields that will be updated by the tool"
    )
    fallback: Optional[str] = Field(
        default=None,
        description="Optional fallback recommendation if the tool fails",
    )
    extracted_inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Automatically extracted input values for this tool",
    )


class MasterDecision(BaseModel):
    """Structured instructions for master execution."""

    master: str
    intent: str = ""
    plan: List[ToolPlan] = Field(default_factory=list)
    hitl_required: bool = False
    hitl_reason: Optional[str] = None
    final_goal: Optional[str] = Field(
        default=None,
        description="One sentence summary of what success looks like for this master",
    )


class BaseLLMMaster:
    """Reusable LLM-driven planner for deciding which tools to call."""

    def __init__(
        self,
        master_name: str,
        llm,
        master_specs: Dict[str, MasterSpec] | None = None,
        tool_specs: Dict[str, ToolSpec] | None = None,
    ) -> None:
        if llm is None:
            raise ValueError("llm must be provided to initialize a master")

        specs = master_specs or MASTER_SPECS
        if master_name not in specs:
            raise ValueError(f"Unknown master: {master_name}")

        self.spec = specs[master_name]
        self.llm = llm
        self.tool_specs = tool_specs or TOOL_SPECS
        self._parser = JsonOutputParser(pydantic_object=MasterDecision)

        available_tools = self._format_tool_catalog(self.spec.tool_chain)
        system_prompt = (
            "You are a domain master orchestrator. "
            "Select the minimal set of tools to satisfy the user's request. "
            "If required information is missing, mark hitl_required=true and explain."
        )

        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "system",
                    (
                        "Master spec:\n"
                        f"Name: {self.spec.name}\n"
                        f"Domain: {self.spec.domain}\n"
                        f"Description: {self.spec.description}\n"
                        f"Supported intents: {', '.join(self.spec.intents)}\n"
                        f"Known tools:\n{available_tools}\n"
                        f"HitL triggers: {', '.join(self.spec.hitl_triggers)}\n"
                        "Output JSON MUST match the schema provided."
                    ),
                ),
                (
                    "user",
                    (
                        "User query: {query}\n"
                        "Active supervisor task (optional): {task_context}\n"
                        "Snapshot of current state:\n{state_summary}\n"
                        "Follow these JSON instructions:\n{format_instructions}"
                    ),
                ),
            ]
        )
        self._chain = self._prompt | self.llm | self._parser

    def plan_tools(
        self,
        query: str,
        state: Dict[str, Any] | None = None,
        task_context: Dict[str, Any] | None = None,
        intent: Optional[str] = None,
    ) -> MasterDecision:
        """Return the LLM-authored plan for this master."""

        state_summary = json.dumps(state or {}, ensure_ascii=False, default=str)
        task_summary = json.dumps(task_context or {}, ensure_ascii=False, default=str)
        payload = {
            "query": query,
            "task_context": task_summary,
            "state_summary": state_summary,
            "format_instructions": self._parser.get_format_instructions(),
        }

        raw_result = self._chain.invoke(payload)
        if isinstance(raw_result, MasterDecision):
            result = raw_result
        else:
            result = MasterDecision(**raw_result)

        # Fill intent if caller constrained it
        if intent:
            result.intent = intent
        elif not result.intent:
            result.intent = self.spec.intents[0]

        result.master = self.spec.name
        self._extract_tool_inputs(query, result)
        return result

    def _format_tool_catalog(self, tool_names: List[str]) -> str:
        lines: List[str] = []
        for name in tool_names:
            if name not in self.tool_specs:
                continue
            spec = self.tool_specs[name]
            lines.append(
                f"- {spec.name}: {spec.description}\n"
                f"  Inputs: {', '.join(spec.inputs) or 'n/a'}\n"
                f"  Outputs: {', '.join(spec.outputs) or 'n/a'}"
            )
        return "\n".join(lines) if lines else "No tools registered."

    def _extract_tool_inputs(self, query: str, decision: MasterDecision) -> None:
        """Use the LLM to infer tool inputs directly from the user query."""

        for plan in decision.plan:
            spec = self.tool_specs.get(plan.tool)
            required_fields: List[str] = []
            if spec and spec.inputs:
                required_fields.extend(spec.inputs)
            if plan.inputs_to_collect:
                required_fields.extend(plan.inputs_to_collect)

            unique_fields: List[str] = []
            for field in required_fields:
                if field and field not in unique_fields:
                    unique_fields.append(field)

            if not unique_fields:
                plan.extracted_inputs = {}
                continue

            prompt = (
                "Extract structured values for the following tool inputs. "
                "Return a JSON object with each field as a key. "
                "If a value cannot be determined from the query, set it to null.\n"
                f"Tool: {plan.tool}\n"
                f"Action: {plan.action}\n"
                f"Fields: {unique_fields}\n"
                f"User query: {query}\n"
                "JSON:"
            )

            try:
                response = self.llm.invoke(prompt)
                content = getattr(response, "content", "")
                plan.extracted_inputs = json.loads(content) if content else {}
            except Exception:
                plan.extracted_inputs = {field: None for field in unique_fields}


__all__ = ["BaseLLMMaster", "MasterDecision", "ToolPlan"]

