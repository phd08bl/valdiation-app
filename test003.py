from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
try:
    from langchain_ollama import ChatOllama
except Exception:
    from langchain_community.chat_models import ChatOllama

# ============================================================
# ✅ CONFIG — 你只需要改这一段
# ============================================================

CONFIG = {
    # (1) Ollama 连接地址（默认本地）
    "OLLAMA_BASE_URL": "http://localhost:11434",

    # (2) Ollama 模型名（你机器上 `ollama list` 能看到的名字）
    "OLLAMA_MODEL": "qwen3:4b-instruct-2507-q4_K_M",

    # (3) 你的 PDF 路径
    "PDF_PATH": "./data/ss123.pdf",

    # (4) Chunk size：每次给模型多少段落（越大越慢、越容易超上下文）
    "CHUNK_SIZE": 20,

    # (5) 上下文长度（取决于模型支持）
    "NUM_CTX": 8192,

    # (6) 温度（越低越稳定）
    "TEMPERATURE": 0.2,

    # (7) 是否打印模型原始输出（调试用）
    "DEBUG_PRINT_RAW_MODEL_OUTPUT": False,

    # (8) Docling 最大页数（None=不限制；Demo 可设 5）
    "DOCLING_MAX_PAGES": None,

    # (9) HITL 模式：
    #   - "langgraph": 使用 HumanInTheLoopMiddleware + InMemorySaver + interrupt/resume
    #   - "manual": 强制纯人工确认更新（不依赖 langgraph/agent 组件）
    "HITL_MODE": "langgraph",

    # (10) CLI 审阅者 ID（写入 resolved_by）
    "REVIEWER_ID": "ivt_user_001",
}

# ============================================================
# 0) Optional HITL dependencies (LangGraph agent middleware)
# ============================================================

HITL_AVAILABLE = False
try:
    # These imports match your HITL tutorial code
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    HITL_AVAILABLE = True
except Exception:
    HITL_AVAILABLE = False


# ============================================================
# 1) PDF Parsing (Docling 2.75.0)
# ============================================================

def normalize_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_into_paragraphs(text: str) -> List[str]:
    txt = normalize_text(text)
    if not txt:
        return []

    parts = [p.strip() for p in txt.split("\n\n") if p.strip()]
    if len(parts) >= 2:
        return parts

    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
    if not lines:
        return []
    return [" ".join(lines).strip()]


def _try_export_dict(doc) -> Optional[dict]:
    for fn_name in ("export_to_dict", "to_dict", "model_dump"):
        if hasattr(doc, fn_name):
            try:
                fn = getattr(doc, fn_name)
                data = fn() if callable(fn) else fn
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return None


def _try_export_markdown(doc) -> Optional[str]:
    for fn_name in ("export_to_markdown", "to_markdown"):
        if hasattr(doc, fn_name):
            try:
                fn = getattr(doc, fn_name)
                md = fn() if callable(fn) else fn
                if isinstance(md, str) and md.strip():
                    return md
            except Exception:
                pass
    return None


def extract_text_from_pdf_docling(pdf_path: str, max_pages: Optional[int] = None) -> list[dict]:
    """
    Returns list of pages:
      [{"page_num": 1, "content": "..."}, ...]
    """
    pdf_path = str(Path(pdf_path))
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = getattr(result, "document", None)
    if doc is None:
        raise RuntimeError("Docling conversion failed: result.document is None")

    doc_dict = _try_export_dict(doc)
    if isinstance(doc_dict, dict):
        pages = doc_dict.get("pages") or doc_dict.get("document", {}).get("pages") or []
        out_pages: list[dict] = []
        for idx, page in enumerate(pages, start=1):
            if isinstance(page, dict):
                page_num = page.get("page_num") or page.get("page_number") or page.get("page") or idx
            else:
                page_num = idx
            if max_pages is not None and int(page_num) > max_pages:
                break

            page_text = None
            if isinstance(page, dict):
                page_text = page.get("text") or page.get("content_text")
                if not page_text:
                    blocks = page.get("paragraphs") or page.get("blocks") or []
                    if isinstance(blocks, list):
                        page_text = "\n".join(
                            (b.get("text") or b.get("content") or "").strip()
                            for b in blocks
                            if isinstance(b, dict) and (b.get("text") or b.get("content"))
                        )
            elif isinstance(page, str):
                page_text = page

            out_pages.append({"page_num": int(page_num), "content": normalize_text(page_text or "")})
        if out_pages:
            return out_pages

    md = _try_export_markdown(doc)
    if md:
        return [{"page_num": 1, "content": normalize_text(md)}]

    return [{"page_num": 1, "content": normalize_text(str(doc))}]


# ============================================================
# 2) Ollama LLM init
# ============================================================

def init_llm_ollama() -> ChatOllama:
    return ChatOllama(
        model=CONFIG["OLLAMA_MODEL"],
        base_url=CONFIG["OLLAMA_BASE_URL"],
        temperature=CONFIG["TEMPERATURE"],
        model_kwargs={"num_ctx": CONFIG["NUM_CTX"]},
    )


def check_ollama_connection(llm: ChatOllama) -> None:
    try:
        resp = llm.invoke([HumanMessage(content="请只回复OK")])
        if not (resp and getattr(resp, "content", "").strip()):
            raise RuntimeError("Ollama responded but content is empty.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to Ollama or run model.\n"
            f"- base_url: {CONFIG['OLLAMA_BASE_URL']}\n"
            f"- model: {CONFIG['OLLAMA_MODEL']}\n"
            f"Troubleshooting:\n"
            f"  1) Ensure Ollama is running: `ollama serve`\n"
            f"  2) Ensure model is pulled: `ollama pull {CONFIG['OLLAMA_MODEL']}`\n"
            f"Original error: {e}"
        )


# ============================================================
# 3) Review output schemas (unchanged)
# ============================================================

IssueType = Literal["语法错误", "用词不当", "逻辑问题", "敏感表述"]


class ReviewIssue(BaseModel):
    type: str = Field(description="问题类型，如：语法错误、用词不当、逻辑问题、敏感表述")
    text: str = Field(description="问题所在的原文片段")
    explanation: str = Field(description="问题的详细说明")
    suggested_fix: str = Field(description="修改建议")
    para_index: int = Field(description="问题所在段落的索引（从0开始）")


class ReviewOutput(BaseModel):
    issues: List[ReviewIssue] = Field(description="发现的问题列表")


# ============================================================
# 4) Custom rules models (unchanged)
# ============================================================

class RiskLevel(str, Enum):
    HIGH = "高"
    MEDIUM = "中"
    LOW = "低"


class RuleExample(BaseModel):
    text: str = Field(description="示例文本")
    explanation: str = Field(description="说明")


class ReviewRule(BaseModel):
    name: str = Field(description="规则名称")
    description: str = Field(description="规则描述，说明什么情况下触发此规则")
    risk_level: RiskLevel = Field(description="风险等级")
    examples: list[RuleExample] = Field(default=[], description="示例列表")


# ============================================================
# 5) Prompts (unchanged)
# ============================================================

def build_system_prompt(custom_rules: list[ReviewRule] = None) -> str:
    issue_types = [
        "- 语法错误：错别字、标点符号错误、语病等",
        "- 用词不当：使用了不恰当的词语或表达",
        "- 敏感表述：使用了'必须'、'保证'、'一定'等过度承诺的措辞",
    ]

    if custom_rules:
        for rule in custom_rules:
            rule_desc = f"- {rule.name}：{rule.description}"
            if rule.examples:
                examples_str = "；".join([f'"{ex.text}"' for ex in rule.examples[:2]])
                rule_desc += f"（示例：{examples_str}）"
            issue_types.append(rule_desc)

    return f"""你是一位专业的文档审核专家。
请仔细审查提供的文本，识别其中的问题。

需要检查的问题类型：
{chr(10).join(issue_types)}

注意事项：
1. 文档可能是中文或英文，请根据语言选择合适的审核标准
2. 使用输入中提供的段落索引（如 [0], [1], ...）来标识问题位置
3. 每个问题都需要提供具体的修改建议
4. 如果没有发现问题，返回空的问题列表
5. 按照要求的 JSON 格式输出结果
"""


def build_user_prompt(paragraphs: list[dict], parser: PydanticOutputParser) -> str:
    formatted_text = "\n".join([f"[{i}] {p['content']}" for i, p in enumerate(paragraphs)])
    return f"""请审核以下文本内容：

{formatted_text}

如果发现问题，请按以下格式输出；如果没有问题，返回空的 issues 列表。

{parser.get_format_instructions()}
"""


# ============================================================
# 6) Robust JSON parsing helper (Ollama)
# ============================================================

def parse_llm_output_to_reviewoutput(parser: PydanticOutputParser, raw: str) -> ReviewOutput:
    raw = (raw or "").strip()
    try:
        return parser.parse(raw)
    except Exception:
        pass

    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        candidate = m.group(0).strip()
        try:
            json.loads(candidate)
            return parser.parse(candidate)
        except Exception:
            pass

    return ReviewOutput(issues=[])


# ============================================================
# 7) Chunking + streaming review (unchanged)
# ============================================================

def chunk_paragraphs(paragraphs: list[dict], chunk_size: int = 20) -> list[list[dict]]:
    return [paragraphs[i:i + chunk_size] for i in range(0, len(paragraphs), chunk_size)]


def stream_review_document(
    paragraphs: list[dict],
    llm: ChatOllama,
    custom_rules: list[ReviewRule] = None,
    chunk_size: int = 20,
) -> Generator[ReviewOutput, None, None]:
    chunks = chunk_paragraphs(paragraphs, chunk_size)
    total_chunks = len(chunks)

    print(f"文档共 {len(paragraphs)} 个段落，分成 {total_chunks} 块处理")
    print("-" * 50)

    parser = PydanticOutputParser(pydantic_object=ReviewOutput)
    system_prompt = build_system_prompt(custom_rules)

    for i, chunk in enumerate(chunks):
        print(f"\n正在处理第 {i+1}/{total_chunks} 块...")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=build_user_prompt(chunk, parser)),
        ]

        response = llm.invoke(messages)
        raw = getattr(response, "content", "") or ""

        if CONFIG["DEBUG_PRINT_RAW_MODEL_OUTPUT"]:
            print("\n--- RAW MODEL OUTPUT START ---")
            print(raw[:2000])
            print("--- RAW MODEL OUTPUT END ---\n")

        result = parse_llm_output_to_reviewoutput(parser, raw)
        print(f"   发现 {len(result.issues)} 个问题")
        yield result

    print("\n" + "-" * 50)
    print("全部处理完成！")


# ============================================================
# 8) HITL data model + in-memory DB
# ============================================================

class IssueStatus(str, Enum):
    not_reviewed = "not_reviewed"
    accepted = "accepted"
    dismissed = "dismissed"


class Issue(BaseModel):
    """
    HITL-protected Issue object stored in DB.
    This extends ReviewIssue with review status + audit fields.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    page_num: int
    para_index: int

    type: str
    text: str
    explanation: str = ""
    suggested_fix: str = ""

    status: IssueStatus = IssueStatus.not_reviewed
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None


@dataclass
class DecisionLog:
    ts: str
    user: str
    action: str
    issue_id: str
    before: Dict[str, Any]
    after: Dict[str, Any]


class IssuesDatabase:
    """
    In-memory DB. Replace with SQLite/Postgres in production.
    """
    def __init__(self):
        self._issues: Dict[str, Issue] = {}
        self._logs: List[DecisionLog] = []

    def add_issue(self, issue: Issue) -> None:
        self._issues[issue.id] = issue

    def get_issue(self, issue_id: str) -> Optional[Issue]:
        return self._issues.get(issue_id)

    def list_issues(self) -> List[Issue]:
        return list(self._issues.values())

    def update_issue(self, issue_id: str, update_fields: Dict[str, Any], actor: str) -> Issue:
        issue = self._issues.get(issue_id)
        if not issue:
            raise ValueError(f"问题不存在: {issue_id}")

        before = issue.model_dump()
        new_data = dict(before)
        new_data.update(update_fields)

        updated = Issue(**new_data)
        self._issues[issue_id] = updated

        self._logs.append(
            DecisionLog(
                ts=datetime.now(timezone.utc).isoformat(),
                user=actor,
                action="update_issue",
                issue_id=issue_id,
                before=before,
                after=updated.model_dump(),
            )
        )
        return updated

    def list_logs(self) -> List[DecisionLog]:
        return self._logs


# ============================================================
# 9) HITL agent wrapper (LangGraph middleware) + manual fallback
# ============================================================

def make_update_fields(status: IssueStatus, reviewer_id: str, edited_fix: Optional[str] = None) -> Dict[str, Any]:
    update_fields = {
        "status": status.value,
        "resolved_by": reviewer_id,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }
    if edited_fix is not None and edited_fix.strip():
        update_fields["suggested_fix"] = edited_fix.strip()
    return update_fields


def build_hitl_agent(llm: ChatOllama, tool_update_issue):
    """
    Build HITL agent exactly like tutorial (create_agent + HumanInTheLoopMiddleware + InMemorySaver).
    """
    if not HITL_AVAILABLE:
        raise RuntimeError("HITL dependencies not available in environment.")

    SYSTEM_PROMPT = """你是一个审阅工作流执行器。
你会收到 issue_id 和 update_fields。
你必须且只能调用一次 `update_issue` 工具，并严格使用提供的参数。
不要自行猜测、不要新增字段、不要修改字段含义。
"""

    agent = create_agent(
        model=llm,
        tools=[tool_update_issue],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"update_issue": True},
                description_prefix="需要人工确认的操作",
            ),
        ],
        checkpointer=InMemorySaver(),
    )
    return agent


async def start_hitl(agent, thread_id: str, issue_id: str, update_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start HITL and return interrupt info.
    """
    config = {"configurable": {"thread_id": thread_id}}

    prompt = (
        "请按照提供的参数更新 issue。\n"
        f"issue_id: {issue_id}\n"
        f"update_fields(JSON): {json.dumps(update_fields, ensure_ascii=False)}\n"
        "你必须调用 update_issue。\n"
    )

    async for step in agent.astream(
        {"messages": [HumanMessage(content=prompt)]},
        config,
        stream_mode="values"
    ):
        if "__interrupt__" in step:
            interrupt = step["__interrupt__"][0]
            # robust return shape
            if hasattr(interrupt, "id") and hasattr(interrupt, "value"):
                return {"id": interrupt.id, "value": interrupt.value, "thread_id": thread_id}
            return {"value": interrupt, "thread_id": thread_id}

    return {}


async def resume_hitl(agent, thread_id: str, decision: Dict[str, Any], interrupt_id: str = None) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    cmd = Command(resume={"decisions": [decision]})
    async for step in agent.astream(cmd, config, stream_mode="values"):
        if "__interrupt__" in step:
            raise RuntimeError("HITL 恢复后产生了新的中断（异常情况）")


def run_hitl_update(
    *,
    mode: str,
    llm: ChatOllama,
    db: IssuesDatabase,
    issue_id: str,
    update_fields: Dict[str, Any],
    reviewer_id: str,
) -> Tuple[bool, str]:
    """
    Apply update under HITL control.

    Returns: (success, message)
    - success=True means DB updated.
    - success=False means rejected / cancelled.
    """
    if mode == "manual" or not HITL_AVAILABLE:
        # Manual: user confirms; no agent/middleware
        db.update_issue(issue_id, update_fields, actor=reviewer_id)
        return True, "manual_update_ok"

    # LangGraph HITL agent mode
    # tool function that the agent will call (interrupted by middleware)
    def update_issue(issue_id: str, update_fields: Dict[str, Any]) -> str:
        """Update an issue in the DB with reviewer action fields."""
        db.update_issue(issue_id, update_fields, actor=reviewer_id)
        return "ok"

    agent = build_hitl_agent(llm, update_issue)

    thread_id = f"issue:{issue_id}:{uuid.uuid4()}"
    interrupt_info = asyncio.run(start_hitl(agent, thread_id, issue_id, update_fields))
    if not interrupt_info:
        # should not happen
        return False, "no_interrupt_returned"

    # Now HUMAN decision happens outside (we do it in CLI below).
    # This function expects the caller already decided; here we just approve by default.
    # We'll override in CLI flow (approve/edit/reject).
    return True, f"interrupt_ready:{thread_id}"


# ============================================================
# 10) Full PDF review -> Issues DB
# ============================================================

def review_pdf_to_db(
    pdf_path: str,
    llm: ChatOllama,
    custom_rules: list[ReviewRule] = None,
    chunk_size: int = 20,
    max_pages: Optional[int] = None,
) -> Tuple[str, IssuesDatabase]:
    """
    Parse PDF -> paragraphs -> stream review -> store Issues in IssuesDatabase.
    Returns (doc_id, db)
    """
    doc_id = f"doc:{uuid.uuid4()}"

    print("开始文档审核流程")
    print("=" * 60)

    print("\n步骤1：解析 PDF 文档（Docling，本地）")
    pages = extract_text_from_pdf_docling(pdf_path, max_pages=max_pages)
    print(f"提取了 {len(pages)} 页内容")

    all_paragraphs: list[dict] = []
    # Keep mapping from global para index to (page_num, content)
    for page in pages:
        for para in split_into_paragraphs(page["content"]):
            if para.strip():
                all_paragraphs.append({"content": para.strip(), "page_num": page["page_num"]})

    print(f"  共 {len(all_paragraphs)} 个段落")

    print("\n步骤2：执行文档审核（Ollama + 分块）")
    db = IssuesDatabase()

    # We review chunk-by-chunk; para_index in model output is chunk-local.
    # We must map it back to global index.
    global_offset = 0
    for chunk in chunk_paragraphs(all_paragraphs, chunk_size):
        # Run single chunk review using stream function mechanics (reuse internals)
        parser = PydanticOutputParser(pydantic_object=ReviewOutput)
        system_prompt = build_system_prompt(custom_rules)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=build_user_prompt(chunk, parser)),
        ]
        response = llm.invoke(messages)
        raw = getattr(response, "content", "") or ""

        if CONFIG["DEBUG_PRINT_RAW_MODEL_OUTPUT"]:
            print("\n--- RAW MODEL OUTPUT START ---")
            print(raw[:2000])
            print("--- RAW MODEL OUTPUT END ---\n")

        result = parse_llm_output_to_reviewoutput(parser, raw)

        # Store each issue in DB with page_num/global para_index
        for it in result.issues:
            # map chunk-local para_index -> global
            local_idx = it.para_index
            if 0 <= local_idx < len(chunk):
                page_num = int(chunk[local_idx]["page_num"])
                global_para_index = global_offset + local_idx
            else:
                # fallback: unknown mapping
                page_num = 0
                global_para_index = global_offset + max(0, min(local_idx, len(chunk)-1))

            issue = Issue(
                doc_id=doc_id,
                page_num=page_num,
                para_index=global_para_index,
                type=it.type,
                text=it.text,
                explanation=it.explanation,
                suggested_fix=it.suggested_fix,
            )
            db.add_issue(issue)

        global_offset += len(chunk)

    print("\n" + "=" * 60)
    print(f"审核完成！共发现 {len(db.list_issues())} 个问题（待人工 HITL 处理）")
    return doc_id, db


# ============================================================
# 11) CLI HITL Review Loop (approve / edit / reject)
# ============================================================

def print_issue_card(issue: Issue) -> None:
    print("\n" + "-" * 70)
    print(f"Issue ID: {issue.id}")
    print(f"Doc: {issue.doc_id}")
    print(f"Page: {issue.page_num} | ParaIndex(Global): [{issue.para_index}]")
    print(f"Type: {issue.type}")
    print(f"Status: {issue.status.value}")
    print(f"Text: {issue.text}")
    print(f"Explanation: {issue.explanation}")
    print(f"Suggested fix: {issue.suggested_fix}")
    print("-" * 70)


def hitl_cli_process_issues(
    llm: ChatOllama,
    db: IssuesDatabase,
    reviewer_id: str,
    hitl_mode: str,
) -> None:
    """
    For each issue:
      - approve -> accepted
      - edit -> edit suggested_fix and accept
      - reject -> no update (keep not_reviewed)
      - dismiss -> dismissed
    """
    if hitl_mode == "langgraph" and not HITL_AVAILABLE:
        print("⚠️ HITL_MODE=langgraph but HITL deps not available. Falling back to manual.")
        hitl_mode = "manual"

    print("\n进入 HITL 人工复核阶段（CLI）")
    print("输入指令：")
    print("  a = approve(accepted)")
    print("  d = dismiss(dismissed)")
    print("  e = edit fix then accept")
    print("  r = reject (no change)")
    print("  q = quit")
    print("-" * 70)

    # If langgraph mode, we build a single agent + tool once (better than per issue)
    agent = None
    if hitl_mode == "langgraph":
        def update_issue(issue_id: str, update_fields: Dict[str, Any]) -> str:
            """Update an issue in the DB with reviewer action fields."""
            db.update_issue(issue_id, update_fields, actor=reviewer_id)
            return "ok"
        agent = build_hitl_agent(llm, update_issue)

    for issue in db.list_issues():
        # skip already processed
        if issue.status != IssueStatus.not_reviewed:
            continue

        print_issue_card(issue)
        cmd = input("Decision [a/d/e/r/q]: ").strip().lower()

        if cmd == "q":
            print("退出 HITL 复核。")
            break

        if cmd == "r":
            print("✅ Reject: 不更新（保持 not_reviewed）。")
            continue

        if cmd == "a":
            update_fields = make_update_fields(IssueStatus.accepted, reviewer_id)
            decision = {"type": "approve"}
        elif cmd == "d":
            update_fields = make_update_fields(IssueStatus.dismissed, reviewer_id)
            decision = {"type": "approve"}  # approve the proposed action
        elif cmd == "e":
            new_fix = input("Enter edited suggested_fix: ").strip()
            update_fields = make_update_fields(IssueStatus.accepted, reviewer_id, edited_fix=new_fix)
            # In tutorial, edit_decision can change tool args; we implement that.
            decision = {
                "type": "edit",
                "edited_action": {
                    "name": "update_issue",
                    "args": {
                        "issue_id": issue.id,
                        "update_fields": update_fields
                    }
                }
            }
        else:
            print("无效输入，跳过该 issue。")
            continue

        if hitl_mode == "manual":
            # Manual apply
            db.update_issue(issue.id, update_fields, actor=reviewer_id)
            updated = db.get_issue(issue.id)
            print(f"✅ Updated (manual): status={updated.status.value} by={updated.resolved_by}")
            continue

        # langgraph HITL flow: start -> interrupt -> resume with decision
        thread_id = f"issue:{issue.id}:{uuid.uuid4()}"
        interrupt_info = asyncio.run(start_hitl(agent, thread_id, issue.id, update_fields))
        if not interrupt_info:
            print("❌ 未触发中断（异常情况），跳过。")
            continue

        print("\n⏸ HITL interrupt triggered. Proposed action: update_issue(...)")
        print(f"   thread_id: {thread_id}")
        print(f"   interrupt_id: {interrupt_info.get('id', '(none)')}")

        if cmd in ("a", "d"):
            # approve decision
            asyncio.run(resume_hitl(agent, thread_id, decision={"type": "approve"}, interrupt_id=interrupt_info.get("id")))
        elif cmd == "e":
            asyncio.run(resume_hitl(agent, thread_id, decision=decision, interrupt_id=interrupt_info.get("id")))

        updated = db.get_issue(issue.id)
        print(f"✅ Updated (HITL): status={updated.status.value} by={updated.resolved_by}")

    print("\nHITL 复核阶段结束。")


# ============================================================
# 12) Report generator (same idea but uses HITL Issue objects)
# ============================================================

def generate_report_from_db(db: IssuesDatabase, output_path: str = None) -> str:
    issues = db.list_issues()

    type_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    for it in issues:
        type_counts[it.type] = type_counts.get(it.type, 0) + 1
        status_counts[it.status.value] = status_counts.get(it.status.value, 0) + 1

    report = []
    report.append("# MRO 文档审核报告（含 HITL 状态）")
    report.append("")
    report.append("## 概要")
    report.append(f"- 问题总数：**{len(issues)}**")
    report.append(f"- 状态分布：{json.dumps(status_counts, ensure_ascii=False)}")
    report.append("")
    report.append("## 问题类型分布")
    for issue_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        report.append(f"- {issue_type}：{count} 个")
    report.append("")
    report.append("## 问题详情")
    for i, it in enumerate(issues, 1):
        report.append(f"\n### 问题 {i}")
        report.append(f"- **ID**：{it.id}")
        report.append(f"- **页面**：{it.page_num}")
        report.append(f"- **段落索引**：[{it.para_index}]")
        report.append(f"- **类型**：{it.type}")
        report.append(f"- **状态**：{it.status.value}")
        report.append(f"- **原文**：{it.text}")
        report.append(f"- **说明**：{it.explanation}")
        report.append(f"- **建议**：{it.suggested_fix}")
        report.append(f"- **处理人**：{it.resolved_by or ''}")
        report.append(f"- **处理时间**：{it.resolved_at or ''}")

    report_text = "\n".join(report)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"报告已保存到：{output_path}")

    return report_text


# ============================================================
# 13) Main
# ============================================================

if __name__ == "__main__":
    llm = init_llm_ollama()
    check_ollama_connection(llm)
    print(f"✅ Ollama ready: base_url={CONFIG['OLLAMA_BASE_URL']} model={CONFIG['OLLAMA_MODEL']}")

    # Example custom rules (replace with MRO pack rules)
    sample_rules = [
        ReviewRule(
            name="夸大宣传",
            description="检查是否有夸大效果或功能的表述，如'最好'、'第一'、'独家'等",
            risk_level=RiskLevel.HIGH,
            examples=[
                RuleExample(text="我们的产品是市场上最好的", explanation="使用了绝对化用语'最好'"),
                RuleExample(text="独家技术，行业第一", explanation="使用了'独家'、'第一'等夸大词汇"),
            ],
        ),
        ReviewRule(
            name="数据引用",
            description="检查引用的数据是否标注了来源",
            risk_level=RiskLevel.MEDIUM,
            examples=[
                RuleExample(text="据统计，90%的用户表示满意", explanation="未标注统计数据的来源"),
            ],
        ),
    ]

    # 1) Review -> DB (issues are not_reviewed)
    doc_id, db = review_pdf_to_db(
        pdf_path=CONFIG["PDF_PATH"],
        llm=llm,
        custom_rules=sample_rules,
        chunk_size=CONFIG["CHUNK_SIZE"],
        max_pages=CONFIG["DOCLING_MAX_PAGES"],
    )

    # 2) HITL process (approve/edit/reject/dismiss) in CLI
    hitl_mode = CONFIG["HITL_MODE"].strip().lower()
    if hitl_mode == "langgraph" and not HITL_AVAILABLE:
        print("⚠️ LangGraph HITL components not available; switching to manual.")
        hitl_mode = "manual"

    hitl_cli_process_issues(
        llm=llm,
        db=db,
        reviewer_id=CONFIG["REVIEWER_ID"],
        hitl_mode=hitl_mode,
    )

    # 3) Export report
    report = generate_report_from_db(db, output_path="./output/mro_review_report_with_hitl.md")
    print("\n[报告预览]\n")
    print(report[:1200] + ("\n...\n" if len(report) > 1200 else ""))

    # 4) Optional: show decision logs
    logs = db.list_logs()
    print(f"\nDecision logs: {len(logs)} entries")
    for lg in logs[:5]:
        print(f"- {lg.ts} | {lg.user} | {lg.action} | issue={lg.issue_id}")
