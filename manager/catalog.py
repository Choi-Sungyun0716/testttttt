from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ToolSpec:
    """Metadata describing what a tool does and which fields it touches."""

    name: str
    module: str
    description: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MasterSpec:
    """High level description of a domain master and the tools it can use."""

    name: str
    domain: str
    description: str
    intents: List[str]
    tool_chain: List[str]
    hitl_triggers: List[str] = field(default_factory=list)


TOOL_SPECS: Dict[str, ToolSpec] = {
    # Schedule domain tools
    "annual_leave_inquiry": ToolSpec(
        name="annual_leave_inquiry",
        module="tool.스케줄.annual_leave_inquiry",
        description="연차/반차 잔여 및 사용 이력을 조회합니다.",
        inputs=["input.employee_id", "schedule_domain.leave.leave_type"],
        outputs=["schedule_domain.leave.leave_balance", "output.messages"],
    ),
    "annual_leave_application": ToolSpec(
        name="annual_leave_application",
        module="tool.스케줄.annual_leave_application",
        description="연차/반차 신청서를 생성하거나 상태를 갱신합니다.",
        inputs=[
            "input.employee_id",
            "schedule_domain.leave.leave_type",
            "schedule_domain.leave.start_date",
            "schedule_domain.leave.end_date",
        ],
        outputs=["schedule_domain.leave.leave_request_id", "schedule_domain.leave.leave_status"],
    ),
    "meeting_room_inquiry": ToolSpec(
        name="meeting_room_inquiry",
        module="tool.스케줄.meeting_room_inquiry",
        description="지정된 시간대에 사용 가능한 회의실 목록을 조회합니다.",
        inputs=[
            "schedule_domain.meeting_room.start_time",
            "schedule_domain.meeting_room.end_time",
            "schedule_domain.meeting_room.participants",
            "schedule_domain.meeting_room.require_video",
        ],
        outputs=["schedule_domain.meeting_room.available_rooms"],
    ),
    "meeting_room_recommendation": ToolSpec(
        name="meeting_room_recommendation",
        module="tool.스케줄.meeting_room_recommendation",
        description="가용 회의실을 점수화하고 추천 순위를 제공합니다.",
        inputs=["schedule_domain.meeting_room.available_rooms"],
        outputs=["schedule_domain.meeting_room.recommended_rooms"],
    ),
    "meeting_room_reservation_cancel": ToolSpec(
        name="meeting_room_reservation_cancel",
        module="tool.스케줄.meeting_room_reservation_cancel",
        description="회의실 예약을 생성하거나 취소하고 reservation_id를 관리합니다.",
        inputs=[
            "schedule_domain.meeting_room.selected_room_id",
            "schedule_domain.meeting_room.start_time",
            "schedule_domain.meeting_room.end_time",
        ],
        outputs=[
            "schedule_domain.meeting_room.reservation_id",
            "schedule_domain.meeting_room.reservation_status",
        ],
    ),
    "schedule_register_cancel": ToolSpec(
        name="schedule_register_cancel",
        module="tool.스케줄.schedule_register_cancel",
        description="사내 캘린더 일정을 등록하거나 취소합니다.",
        inputs=[
            "schedule_domain.schedule.title",
            "schedule_domain.schedule.start_time",
            "schedule_domain.schedule.end_time",
            "schedule_domain.meeting_room.reservation_id",
        ],
        outputs=["schedule_domain.schedule.schedule_id", "schedule_domain.schedule.schedule_status"],
    ),
    "schedule_inquiry": ToolSpec(
        name="schedule_inquiry",
        module="tool.스케줄.schedule_inquiry",
        description="직원의 특정 기간 일정 목록을 조회합니다.",
        inputs=["input.employee_id", "schedule_domain.schedule.start_time", "schedule_domain.schedule.end_time"],
        outputs=["schedule_domain.schedule"],
    ),
    "schedule_notification": ToolSpec(
        name="schedule_notification",
        module="tool.스케줄.schedule_notification",
        description="일정/휴가 처리 결과를 알림 메시지로 발송합니다.",
        inputs=["schedule_domain.schedule.schedule_id", "schedule_domain.leave.leave_request_id"],
        outputs=["output.messages"],
    ),
    # Document domain tools
    "document_auto_creation": ToolSpec(
        name="document_auto_creation",
        module="tool.문서.document_auto_creation",
        description="템플릿 기반으로 문서를 자동 작성합니다.",
        inputs=["document_domain.template_name", "document_domain.document_content"],
        outputs=["document_domain.document_content", "document_domain.document_id", "document_domain.document_path"],
    ),
    "auto_approval_request": ToolSpec(
        name="auto_approval_request",
        module="tool.문서.auto_approval_request",
        description="결재선 정보를 이용해 결재 요청을 전송합니다.",
        inputs=["document_domain.document_id", "document_domain.approval_line"],
        outputs=["document_domain.approval_status"],
    ),
    "pdf_conversion": ToolSpec(
        name="pdf_conversion",
        module="tool.문서.pdf_conversion",
        description="생성된 문서를 PDF로 변환합니다.",
        inputs=["document_domain.document_content"],
        outputs=["document_domain.document_path"],
    ),
    "knowledge_upload_rag": ToolSpec(
        name="knowledge_upload_rag",
        module="tool.문서.knowledge_upload_rag",
        description="문서를 RAG 지식베이스에 업로드하고 컬렉션을 관리합니다.",
        inputs=["document_domain.document_path", "document_domain.vector_db_collection"],
        outputs=["document_domain.upload_status", "document_domain.vector_db_collection"],
    ),
    # QA domain tools
    "manual_search_rag": ToolSpec(
        name="manual_search_rag",
        module="tool.Q&A.manual_search_rag",
        description="사내 매뉴얼을 RAG 방식으로 검색합니다.",
        inputs=["qa_domain.search_query", "qa_domain.search_type"],
        outputs=["qa_domain.rag_results"],
    ),
    "welfare_inquiry": ToolSpec(
        name="welfare_inquiry",
        module="tool.Q&A.welfare_inquiry",
        description="복리후생 정보를 조회합니다.",
        inputs=["qa_domain.benefit_category"],
        outputs=["qa_domain.benefit_info"],
    ),
    "menu_inquiry": ToolSpec(
        name="menu_inquiry",
        module="tool.Q&A.menu_inquiry",
        description="사내 식단표를 조회합니다.",
        inputs=["qa_domain.menu_date", "qa_domain.menu_corner"],
        outputs=["qa_domain.menu"],
    ),
    "menu_recommendation": ToolSpec(
        name="menu_recommendation",
        module="tool.Q&A.menu_recommendation",
        description="개인 맞춤 식단을 추천합니다.",
        inputs=["qa_domain.menu_preferences"],
        outputs=["qa_domain.menu_recommendation"],
    ),
    # Email domain tools
    "email_search": ToolSpec(
        name="email_search",
        module="tool.이메일.email_search",
        description="메일함에서 조건에 맞는 이메일을 검색합니다.",
        inputs=["email_domain.search_query", "email_domain.email_importance"],
        outputs=["email_domain.email_search_results"],
    ),
    "reply_draft_generation": ToolSpec(
        name="reply_draft_generation",
        module="tool.이메일.reply_draft_generation",
        description="이메일 회신 초안을 생성합니다.",
        inputs=["input.query"],
        outputs=["email_domain.email_draft", "email_domain.email_subject"],
    ),
    "auto_sending": ToolSpec(
        name="auto_sending",
        module="tool.이메일.auto_sending",
        description="작성된 이메일을 자동으로 발송합니다.",
        inputs=["email_domain.email_draft", "email_domain.email_to", "email_domain.email_cc"],
        outputs=["email_domain.email_sent"],
    ),
    "email_receipt_detection": ToolSpec(
        name="email_receipt_detection",
        module="tool.이메일.email_receipt_detection",
        description="발송된 이메일의 수신 여부를 추적합니다.",
        inputs=["email_domain.email_id"],
        outputs=["email_domain.email_receipt_status"],
    ),
    # Tech domain tools
    "patent_search": ToolSpec(
        name="patent_search",
        module="tool.기술.patent_search",
        description="특허 검색을 수행합니다.",
        inputs=["tech_domain.search_keywords"],
        outputs=["tech_domain.search_results"],
    ),
    "patent_vectorization": ToolSpec(
        name="patent_vectorization",
        module="tool.기술.patent_vectorization",
        description="선택한 특허를 벡터화하여 DB에 적재합니다.",
        inputs=["tech_domain.selected_patent_id", "tech_domain.vector_db_collection"],
        outputs=["tech_domain.vectorization_status", "tech_domain.vector_db_collection"],
    ),
    "paper_search": ToolSpec(
        name="paper_search",
        module="tool.기술.paper_search",
        description="논문/기사 검색을 수행합니다.",
        inputs=["tech_domain.search_keywords"],
        outputs=["tech_domain.search_results"],
    ),
    "paper_vectorization": ToolSpec(
        name="paper_vectorization",
        module="tool.기술.paper_vectorization",
        description="선택한 논문을 벡터화합니다.",
        inputs=["tech_domain.selected_article_id"],
        outputs=["tech_domain.vectorization_status"],
    ),
}


MASTER_SPECS: Dict[str, MasterSpec] = {
    "schedule_master": MasterSpec(
        name="schedule_master",
        domain="schedule",
        description="스케줄/휴가/회의실 관련 복합 작업을 조율합니다.",
        intents=["leave", "meeting_room", "meeting", "inquiry"],
        tool_chain=[
            "meeting_room_inquiry",
            "meeting_room_recommendation",
            "meeting_room_reservation_cancel",
            "schedule_register_cancel",
            "schedule_notification",
            "annual_leave_inquiry",
            "annual_leave_application",
            "schedule_inquiry",
        ],
        hitl_triggers=[
            "participants 정보 부족",
            "회의실 조건 미확정",
            "휴가 승인자 미지정",
        ],
    ),
    "document_master": MasterSpec(
        name="document_master",
        domain="document",
        description="문서 생성, 결재선 관리, 지식 업로드를 담당합니다.",
        intents=["generate", "approval", "upload"],
        tool_chain=[
            "document_auto_creation",
            "pdf_conversion",
            "auto_approval_request",
            "knowledge_upload_rag",
        ],
        hitl_triggers=["approval_line 누락", "템플릿 변수 부족"],
    ),
    "qa_master": MasterSpec(
        name="qa_master",
        domain="qa",
        description="사내 매뉴얼, 복지, 식단 질의에 대응합니다.",
        intents=["manual", "benefit", "menu", "recommend"],
        tool_chain=[
            "manual_search_rag",
            "welfare_inquiry",
            "menu_inquiry",
            "menu_recommendation",
        ],
        hitl_triggers=["검색 범위 모호", "메뉴 날짜 미지정"],
    ),
    "email_master": MasterSpec(
        name="email_master",
        domain="email",
        description="이메일 검색, 초안 작성, 발송 및 연동을 담당합니다.",
        intents=["search", "compose", "notify"],
        tool_chain=[
            "email_search",
            "reply_draft_generation",
            "auto_sending",
            "email_receipt_detection",
        ],
        hitl_triggers=["수신자 정보 누락", "민감도 확인 필요"],
    ),
    "technology_master": MasterSpec(
        name="technology_master",
        domain="tech",
        description="특허/논문 검색과 벡터화를 조율합니다.",
        intents=["patent", "article", "vectorize"],
        tool_chain=[
            "patent_search",
            "patent_vectorization",
            "paper_search",
            "paper_vectorization",
        ],
        hitl_triggers=["검색 키워드 부족", "유료 문서 접근 확인"],
    ),
}

__all__ = ["ToolSpec", "MasterSpec", "TOOL_SPECS", "MASTER_SPECS"]

