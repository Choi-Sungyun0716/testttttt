import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from manager.supervisor import SupervisorRouter
from manager.schedule_master import ScheduleMaster
from manager.document_master import DocumentMaster
from manager.qa_master import QAMaster
from manager.email_master import EmailMaster
from manager.technology_master import TechnologyMaster
from manager.catalog import TOOL_SPECS

LOG_FILE = Path("logs/agent.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")],
)

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

query = "김철수한테 내일 3시 미팅 어떠냐고 물어봐주고 4명이서 쓸 회의실도 좀 잡아줘"

# 1) Supervisor가 전체 시나리오 작성
supervisor = SupervisorRouter(llm=llm)
plan = supervisor.plan(query=query, state={})
logging.info("Supervisor plan: %s", plan.model_dump())

# 2) 계획된 모든 마스터를 순차 실행
MASTER_CLASS_MAP = {
    "schedule_master": ScheduleMaster,
    "document_master": DocumentMaster,
    "qa_master": QAMaster,
    "email_master": EmailMaster,
    "technology_master": TechnologyMaster,
}

for task in plan.master_tasks:
    master_cls = MASTER_CLASS_MAP.get(task.master)
    if not master_cls:
        logging.warning("Unknown master '%s' skipped.", task.master)
        continue
    task_query = getattr(task, "sub_query", "") or query
    logging.info("Dispatching to %s with query: %s", task.master, task_query)

    master = master_cls(llm=llm)
    decision = master.plan_tools(
        query=task_query,
        state={},
        task_context=task.model_dump(),
        intent=task.intent,
    )
    logging.info(
        "%s decision summary: intent=%s hitl_required=%s hitl_reason=%s",
        task.master,
        decision.intent,
        decision.hitl_required,
        decision.hitl_reason,
    )
    print(f"[Master] {task.master} (intent={decision.intent})")
    for tool_plan in decision.plan:
        spec = TOOL_SPECS.get(tool_plan.tool)
        required_inputs = spec.inputs if spec else []
        output_fields = tool_plan.expected_outputs or []
        print(f"  - Tool: {tool_plan.tool}")
        print(
            f"    필요한 입력값: {', '.join(required_inputs) if required_inputs else '-'}"
        )
        print(
            "    예상 반환값: "
            f"{', '.join(tool_plan.expected_outputs) if tool_plan.expected_outputs else '-'}"
        )
        logging.info(
            "%s tool: %s | action=%s | required_inputs=%s | inputs_to_collect=%s | expected_outputs=%s",
            task.master,
            tool_plan.tool,
            tool_plan.action,
            ", ".join(required_inputs) if required_inputs else "-",
            ", ".join(tool_plan.inputs_to_collect) if tool_plan.inputs_to_collect else "-",
            ", ".join(tool_plan.expected_outputs) if tool_plan.expected_outputs else "-",
        )
        extracted_inputs = getattr(tool_plan, "extracted_inputs", None) or {}
        if extracted_inputs:
            logging.info(
                "%s tool extracted inputs: %s",
                tool_plan.tool,
                extracted_inputs,
            )
        else:
            logging.info("%s tool extracted inputs: {}", tool_plan.tool)
logging.info("-" * 60)
print("로그 파일 기록 완료")