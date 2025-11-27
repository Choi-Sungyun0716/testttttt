# Orchestration Runtime Usage

This project now exposes LLM-driven supervisor and master planners. The snippet
below demonstrates the expected wiring using a LangChain chat model.

```python
from langchain_openai import ChatOpenAI

from manager.supervisor import SupervisorRouter
from manager.schedule_master import ScheduleMaster

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

supervisor = SupervisorRouter(llm=llm)
plan = supervisor.plan(
    query="내일 1시에 6명이 사용할 회의실을 예약해줘. 스크린도 필요해.",
    state={},
)

print(plan.model_dump())

schedule_master = ScheduleMaster(llm=llm)
decision = schedule_master.plan_tools(
    query=plan.master_tasks[0].reason,
    state={},
    task_context=plan.master_tasks[0].model_dump(),
    intent=plan.master_tasks[0].intent,
)

print(decision.model_dump())
```

## Implementation Notes

- Both supervisor and masters rely on the catalog defined in `manager.catalog`
  to describe available domains and tools. Updating that file automatically
  changes the planning context.
- Every planner returns a structured Pydantic model, so downstream LangGraph or
  workflow nodes can consume the plan without ad-hoc parsing.
- If the LLM determines that essential information is missing, it will mark
  `hitl_required=true` and include `hitl_reason`, allowing the runtime to pause
  and collect data from the user before resuming.


