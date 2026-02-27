from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import gradio as gr
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
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_MODEL": "qwen3:4b-instruct-2507-q4_K_M",
    "PDF_PATH": "./data/ss123.pdf",
    "CHUNK_SIZE": 20,
    "NUM_CTX": 8192,
    "TEMPERATURE": 0.2,
    "DEBUG_PRINT_RAW_MODEL_OUTPUT": False,
    "DOCLING_MAX_PAGES": None,
    "HITL_MODE": "langgraph",  # "langgraph" or "manual"
    "REVIEWER_ID": "ivt_user_001",
}

# ============================================================
# 0) Optional HITL dependencies
# ============================================================

HITL_AVAILABLE = False
try:
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    HITL_AVAILABLE = True
except Exception:
    HITL_AVAILABLE = False


# ============================================================
# 1) PDF Parsing (Docling)
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
    pdf_path = str(Path(pdf_path))
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = getattr(result, "document", None)
    if doc is None:
        raise RuntimeError("Docling conversion failed: result.document is None")

    # Preferred path: markdown export
    md = _try_export_markdown(doc)
    if md:
        md_text = normalize_text(md)
        if not md_text.strip():
            print("[docling] WARNING: markdown export returned empty text.")
            return [{"page_num": 1, "content": ""}]
        return [{"page_num": 1, "content": md_text}]

    doc_dict = _try_export_dict(doc)
    if isinstance(doc_dict, dict):
        pages = doc_dict.get("pages") or doc_dict.get("document", {}).get("pages") or []
        print(f"[docling] pages type={type(pages).__name__} len={len(pages) if isinstance(pages, list) else 'n/a'}")
        if isinstance(pages, list) and pages:
            first = pages[0]
            print(f"[docling] first page item type={type(first).__name__}")
            if isinstance(first, dict):
                print(f"[docling] first page keys={sorted(first.keys())}")

        out_pages: list[dict] = []
        if isinstance(pages, list):
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
                    if not page_text:
                        alt = page.get("content")
                        if isinstance(alt, str):
                            page_text = alt
                elif isinstance(page, str):
                    page_text = page
                else:
                    for attr in ("text", "content"):
                        val = getattr(page, attr, None)
                        if isinstance(val, str):
                            page_text = val
                            break

                normalized = normalize_text(page_text or "")
                # Avoid numeric placeholders like "1", "2"
                if normalized.strip().isdigit() or normalized.strip() == str(page_num):
                    print(f"[docling] WARNING: page {page_num} has numeric-only content; setting empty.")
                    normalized = ""

                print(f"[docling] page {page_num} text_len={len(normalized)}")
                out_pages.append({"page_num": int(page_num), "content": normalized})

        if out_pages:
            return out_pages

    print("[docling] WARNING: no text extracted; returning empty page content.")
    return [{"page_num": 1, "content": ""}]


# ============================================================
# 2) LLM init
# ============================================================

def init_llm_ollama() -> ChatOllama:
    return ChatOllama(
        model=CONFIG["OLLAMA_MODEL"],
        base_url=CONFIG["OLLAMA_BASE_URL"],
        temperature=CONFIG["TEMPERATURE"],
        model_kwargs={"num_ctx": CONFIG["NUM_CTX"]},
    )


def check_ollama_connection(llm: ChatOllama) -> None:
    resp = llm.invoke([HumanMessage(content="请只回复OK")])
    if not (resp and getattr(resp, "content", "").strip()):
        raise RuntimeError("Ollama responded but content is empty.")


# ============================================================
# 3) Review output schemas
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
# 4) Custom rules models
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
# 5) Prompts
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
# 6) Robust JSON parsing helper
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


def chunk_paragraphs(paragraphs: list[dict], chunk_size: int = 20) -> list[list[dict]]:
    return [paragraphs[i:i + chunk_size] for i in range(0, len(paragraphs), chunk_size)]


# ============================================================
# 8) HITL data model + in-memory DB
# ============================================================

class IssueStatus(str, Enum):
    not_reviewed = "not_reviewed"
    accepted = "accepted"
    dismissed = "dismissed"


class Issue(BaseModel):
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
# 9) HITL agent wrapper (LangGraph)
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


def start_hitl(agent, thread_id: str, issue_id: str, update_fields: Dict[str, Any]) -> Dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    prompt = (
        "请按照提供的参数更新 issue。\n"
        f"issue_id: {issue_id}\n"
        f"update_fields(JSON): {json.dumps(update_fields, ensure_ascii=False)}\n"
        "你必须调用 update_issue。\n"
    )

    for step in agent.stream(
        {"messages": [HumanMessage(content=prompt)]},
        config,
        stream_mode="values",
    ):
        if "__interrupt__" in step:
            interrupt = step["__interrupt__"][0]
            if hasattr(interrupt, "id") and hasattr(interrupt, "value"):
                return {"id": interrupt.id, "value": interrupt.value, "thread_id": thread_id}
            return {"value": interrupt, "thread_id": thread_id}
    return {}


def resume_hitl(agent, thread_id: str, decision: Dict[str, Any], interrupt_id: str = None) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    cmd = Command(resume={"decisions": [decision]})
    for step in agent.stream(cmd, config, stream_mode="values"):
        if "__interrupt__" in step:
            raise RuntimeError("HITL 恢复后产生了新的中断（异常情况）")


# ============================================================
# 10) Review PDF -> DB
# ============================================================

def review_pdf_to_db(
    pdf_path: str,
    llm: ChatOllama,
    custom_rules: list[ReviewRule] = None,
    chunk_size: int = 20,
    max_pages: Optional[int] = None,
) -> Tuple[str, IssuesDatabase, List[Dict[str, Any]]]:
    """
    Returns: (doc_id, db, all_paragraphs)
      all_paragraphs = [{"content": ..., "page_num": ...}, ...]  (global para_index == list index)
    """
    doc_id = f"doc:{uuid.uuid4()}"
    pages = extract_text_from_pdf_docling(pdf_path, max_pages=max_pages)

    all_paragraphs: list[dict] = []
    for page in pages:
        for para in split_into_paragraphs(page["content"]):
            if para.strip():
                all_paragraphs.append({"content": para.strip(), "page_num": page["page_num"]})

    db = IssuesDatabase()
    global_offset = 0

    for chunk in chunk_paragraphs(all_paragraphs, chunk_size):
        parser = PydanticOutputParser(pydantic_object=ReviewOutput)
        system_prompt = build_system_prompt(custom_rules)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=build_user_prompt(chunk, parser)),
        ]
        response = llm.invoke(messages)
        raw = getattr(response, "content", "") or ""

        if CONFIG["DEBUG_PRINT_RAW_MODEL_OUTPUT"]:
            print(raw[:2000])

        result = parse_llm_output_to_reviewoutput(parser, raw)

        for it in result.issues:
            local_idx = it.para_index
            if 0 <= local_idx < len(chunk):
                page_num = int(chunk[local_idx]["page_num"])
                global_para_index = global_offset + local_idx
            else:
                page_num = 0
                global_para_index = global_offset + max(0, min(local_idx, len(chunk) - 1))

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

    return doc_id, db, all_paragraphs


# ============================================================
# 11) UI helpers
# ============================================================

def rules_to_options(rules: List[ReviewRule]) -> List[str]:
    return [f"{r.name}（{r.risk_level.value}）" for r in rules]


def issues_to_table(issues: List[Issue]) -> List[List[Any]]:
    rows = []
    for it in issues:
        rows.append([
            it.id,
            it.page_num,
            it.para_index,
            it.type,
            it.status.value,
            (it.text[:80] + "…") if len(it.text) > 80 else it.text,
        ])
    return rows


def format_issue_detail(issue: Optional[Issue]) -> str:
    if not issue:
        return "未选择 Issue。"
    return (
        f"**Issue ID:** `{issue.id}`  \n"
        f"**Doc:** `{issue.doc_id}`  \n"
        f"**Page:** {issue.page_num}  \n"
        f"**ParaIndex(Global):** [{issue.para_index}]  \n"
        f"**Type:** {issue.type}  \n"
        f"**Status:** {issue.status.value}  \n\n"
        f"**Text:** {issue.text}  \n\n"
        f"**Explanation:** {issue.explanation}  \n\n"
        f"**Suggested fix:** {issue.suggested_fix}  \n\n"
        f"**Resolved by:** {issue.resolved_by or ''}  \n"
        f"**Resolved at:** {issue.resolved_at or ''}"
    )


def _escape_html(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def _jump_link(selected_para_index: Optional[int]) -> str:
    if selected_para_index is None:
        return ""
    return f"**Jump:** [Open location](#p-{selected_para_index})"


def _location_line(paragraphs: List[Dict[str, Any]], selected_para_index: Optional[int]) -> str:
    if selected_para_index is None or not paragraphs:
        return "**Location:** —"
    if 0 <= selected_para_index < len(paragraphs):
        page = paragraphs[selected_para_index].get("page_num", 0)
        return f"**Location:** Page {page} / ParaIndex {selected_para_index}"
    return "**Location:** —"


def render_document_markdown_full(paragraphs: List[Dict[str, Any]], selected_para_index: Optional[int]) -> str:
    """
    Full document markdown with anchors and optional highlight.
    """
    if not paragraphs:
        return "（暂无文档内容）"

    out = []
    current_page = None
    for i, p in enumerate(paragraphs):
        page = p.get("page_num", 0)
        if page != current_page:
            current_page = page
            out.append(f"\n\n### Page {current_page}\n")

        text = p.get("content", "")
        safe_text = _escape_html(text)
        anchor = f"<a id=\"p-{i}\"></a>"

        if selected_para_index is not None and i == selected_para_index:
            out.append(
                f"- {anchor} **[{i}]** "
                f"<span style='background:#fff3b0; padding:2px 4px;'>{safe_text}</span>"
            )
        else:
            out.append(f"- {anchor} **[{i}]** {safe_text}")

    return "\n".join(out)


def render_document_markdown_focused(
    paragraphs: List[Dict[str, Any]],
    selected_para_index: Optional[int],
    window: int = 5,
) -> str:
    """
    Focused view: render only a window of paragraphs around selected index.
    """
    if not paragraphs:
        return "（暂无文档内容）"

    if selected_para_index is None or selected_para_index < 0:
        center = 0
    else:
        center = min(selected_para_index, len(paragraphs) - 1)

    start = max(0, center - window)
    end = min(len(paragraphs) - 1, center + window)
    out = [f"### Focused View ({start}–{end})\n"]

    for i in range(start, end + 1):
        p = paragraphs[i]
        page = p.get("page_num", 0)
        text = p.get("content", "")
        safe_text = _escape_html(text)
        anchor = f"<a id=\"p-{i}\"></a>"
        prefix = f"**Page {page} | ParaIndex [{i}]**"

        if selected_para_index is not None and i == selected_para_index:
            out.append(
                f"- {anchor} {prefix} "
                f"<span style='background:#fff3b0; padding:2px 4px;'>{safe_text}</span>"
            )
        else:
            out.append(f"- {anchor} {prefix} {safe_text}")

    return "\n".join(out)


def render_document_markdown(
    paragraphs: List[Dict[str, Any]],
    selected_para_index: Optional[int],
    view_mode: str,
    window: int = 5,
    cached_full: Optional[str] = None,
) -> str:
    if view_mode == "Focused View":
        return render_document_markdown_focused(paragraphs, selected_para_index, window=window)
    if selected_para_index is None and isinstance(cached_full, str) and cached_full:
        return cached_full
    return render_document_markdown_full(paragraphs, selected_para_index)


def logs_to_markdown(logs: List[DecisionLog], max_items: int = 30) -> str:
    if not logs:
        return "（暂无决策日志）"
    lines = ["### Decision Logs（最新在前）\n"]
    for lg in list(reversed(logs))[:max_items]:
        lines.append(
            f"- `{lg.ts}` | **{lg.user}** | {lg.action} | issue=`{lg.issue_id}` | "
            f"status: `{lg.before.get('status')}` → `{lg.after.get('status')}`"
        )
    return "\n".join(lines)


# ============================================================
# 12) Gradio callbacks
# ============================================================

def _get_hitl_mode() -> str:
    mode = (CONFIG.get("HITL_MODE") or "manual").strip().lower()
    if mode == "langgraph" and not HITL_AVAILABLE:
        return "manual"
    return mode


def ui_load_and_review(
    pdf_file,
    selected_rules: List[str],
    llm: ChatOllama,
    rules: List[ReviewRule],
    view_mode: str,
):
    """
    Click 'Run Review' -> parse pdf -> run review -> populate db/paragraphs/issues
    """
    if pdf_file is None:
        pdf_path = CONFIG["PDF_PATH"]
    else:
        pdf_path = pdf_file.name  # gr.File gives a temp file path

    # Filter rules (demo: if user selects none -> use all rules)
    use_rules = []
    if selected_rules:
        selected_names = set([s.split("（")[0] for s in selected_rules])
        use_rules = [r for r in rules if r.name in selected_names]
    else:
        use_rules = rules

    doc_id, db, paragraphs = review_pdf_to_db(
        pdf_path=pdf_path,
        llm=llm,
        custom_rules=use_rules,
        chunk_size=CONFIG["CHUNK_SIZE"],
        max_pages=CONFIG["DOCLING_MAX_PAGES"],
    )

    issues = db.list_issues()
    table = issues_to_table(issues)
    full_doc_md = render_document_markdown_full(paragraphs, selected_para_index=None)
    doc_md = render_document_markdown(
        paragraphs,
        selected_para_index=None,
        view_mode=view_mode,
        cached_full=full_doc_md,
    )
    logs_md = logs_to_markdown(db.list_logs())
    summary = f"✅ Review done. doc_id={doc_id} | issues={len(issues)} | HITL_MODE={_get_hitl_mode()}"

    # state objects
    state = {
        "doc_id": doc_id,
        "db": db,
        "paragraphs": paragraphs,
        "selected_issue_id": None,
        "selected_para_index": None,
        "hitl_agent": None,  # lazily build
        "full_doc_md": full_doc_md,
    }
    location_md = _location_line(paragraphs, None)
    jump_md = _jump_link(None)
    return state, summary, doc_md, table, "未选择 Issue。", logs_md, location_md, jump_md


def ui_select_issue(evt: gr.SelectData, state, view_mode: str):
    """
    Select a row from issues table -> show issue detail + highlight paragraph
    """
    if not state or not state.get("db"):
        return state, "（请先 Run Review）", "未选择 Issue。", "**Location:** —", ""

    db: IssuesDatabase = state["db"]
    paragraphs = state.get("paragraphs") or []

    # evt.index could be (row, col)
    row = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
    table_rows = issues_to_table(db.list_issues())
    if row < 0 or row >= len(table_rows):
        doc_md = render_document_markdown(
            paragraphs,
            None,
            view_mode=view_mode,
            cached_full=state.get("full_doc_md"),
        )
        return state, doc_md, "未选择 Issue。", _location_line(paragraphs, None), _jump_link(None)

    issue_id = table_rows[row][0]
    issue = db.get_issue(issue_id)
    state["selected_issue_id"] = issue_id
    state["selected_para_index"] = issue.para_index if issue else None

    doc_md = render_document_markdown(
        paragraphs,
        issue.para_index if issue else None,
        view_mode=view_mode,
        cached_full=state.get("full_doc_md"),
    )
    detail_md = format_issue_detail(issue)
    location_md = _location_line(paragraphs, issue.para_index if issue else None)
    jump_md = _jump_link(issue.para_index if issue else None)
    return state, doc_md, detail_md, location_md, jump_md


def ui_change_view(view_mode: str, state):
    if not state or not state.get("paragraphs"):
        return "（暂无文档内容）", "**Location:** —", ""
    paragraphs = state.get("paragraphs") or []
    selected_idx = state.get("selected_para_index")
    doc_md = render_document_markdown(
        paragraphs,
        selected_idx,
        view_mode=view_mode,
        cached_full=state.get("full_doc_md"),
    )
    location_md = _location_line(paragraphs, selected_idx)
    jump_md = _jump_link(selected_idx)
    return doc_md, location_md, jump_md


def _find_para_index(paragraphs: List[Dict[str, Any]], query: str) -> Optional[int]:
    q = (query or "").strip()
    if not q:
        return None
    if q.isdigit():
        idx = int(q)
        if 0 <= idx < len(paragraphs):
            return idx
        return None
    q_lower = q.lower()
    for i, p in enumerate(paragraphs):
        content = (p.get("content") or "").lower()
        if q_lower in content:
            return i
    return None


def ui_go_to_paragraph(query: str, view_mode: str, state):
    if not state or not state.get("paragraphs"):
        return state, "（暂无文档内容）", "**Location:** —", ""
    paragraphs = state.get("paragraphs") or []
    idx = _find_para_index(paragraphs, query)
    if idx is None:
        doc_md = render_document_markdown(
            paragraphs,
            None,
            view_mode=view_mode,
            cached_full=state.get("full_doc_md"),
        )
        return state, doc_md, "**Location:** 未找到匹配段落", ""

    state["selected_issue_id"] = None
    state["selected_para_index"] = idx
    doc_md = render_document_markdown(
        paragraphs,
        idx,
        view_mode=view_mode,
        cached_full=state.get("full_doc_md"),
    )
    location_md = _location_line(paragraphs, idx)
    jump_md = _jump_link(idx)
    return state, doc_md, location_md, jump_md


def _ensure_agent(llm: ChatOllama, state, reviewer_id: str):
    """
    Build one HITL agent per session/state if mode=langgraph.
    The tool closure captures db + reviewer_id.
    """
    mode = _get_hitl_mode()
    if mode != "langgraph":
        return None

    if state.get("hitl_agent") is not None:
        return state["hitl_agent"]

    db: IssuesDatabase = state["db"]

    def update_issue(issue_id: str, update_fields: Dict[str, Any]) -> str:
        """Update an issue with provided fields (HITL tool)."""
        target_id = issue_id
        if target_id not in db._issues:
            fallback = state.get("selected_issue_id") or state.get("hitl_issue_id")
            if fallback in db._issues:
                target_id = fallback
        if target_id not in db._issues:
            raise ValueError(f"问题不存在: {issue_id}")
        db.update_issue(target_id, update_fields, actor=reviewer_id)
        return "ok"

    agent = build_hitl_agent(llm, update_issue)
    state["hitl_agent"] = agent
    return agent


def _apply_decision_with_hitl(
    llm: ChatOllama,
    state,
    decision_type: str,
    view_mode: str,
    edited_fix: str = "",
):
    """
    decision_type: "approve_accept" | "dismiss" | "edit_accept" | "reject"
    Returns updated outputs for UI.
    """
    if not state or not state.get("db"):
        return state, "（请先 Run Review）", [], "未选择 Issue。", "（暂无决策日志）", render_document_markdown([], None, view_mode=view_mode), "**Location:** —", ""

    db: IssuesDatabase = state["db"]
    paragraphs = state.get("paragraphs") or []
    issue_id = state.get("selected_issue_id")
    if not issue_id:
        return state, "（请先在右侧表格选中一个 Issue）", issues_to_table(db.list_issues()), "未选择 Issue。", logs_to_markdown(db.list_logs()), render_document_markdown(paragraphs, None, view_mode=view_mode, cached_full=state.get("full_doc_md")), _location_line(paragraphs, None), _jump_link(None)

    issue = db.get_issue(issue_id)
    if not issue:
        return state, "（Issue 不存在）", issues_to_table(db.list_issues()), "未选择 Issue。", logs_to_markdown(db.list_logs()), render_document_markdown(paragraphs, None, view_mode=view_mode, cached_full=state.get("full_doc_md")), _location_line(paragraphs, None), _jump_link(None)

    reviewer_id = CONFIG["REVIEWER_ID"]
    mode = _get_hitl_mode()

    # Map decision to update_fields + langgraph decision payload
    if decision_type == "approve_accept":
        update_fields = make_update_fields(IssueStatus.accepted, reviewer_id)
        resume_decision = {"type": "approve"}
    elif decision_type == "dismiss":
        update_fields = make_update_fields(IssueStatus.dismissed, reviewer_id)
        resume_decision = {"type": "approve"}
    elif decision_type == "edit_accept":
        update_fields = make_update_fields(IssueStatus.accepted, reviewer_id, edited_fix=edited_fix)
        resume_decision = {
            "type": "edit",
            "edited_action": {
                "name": "update_issue",
                "args": {"issue_id": issue_id, "update_fields": update_fields},
            },
        }
    elif decision_type == "reject":
        update_fields = make_update_fields(IssueStatus.accepted, reviewer_id)  # proposed action (won't execute)
        resume_decision = {"type": "reject", "message": "Reviewer rejected this action."}
    else:
        return state, "（未知操作）", issues_to_table(db.list_issues()), format_issue_detail(issue), logs_to_markdown(db.list_logs()), render_document_markdown(paragraphs, issue.para_index, view_mode=view_mode, cached_full=state.get("full_doc_md")), _location_line(paragraphs, issue.para_index), _jump_link(issue.para_index)

    # Apply
    if mode == "manual":
        if decision_type == "reject":
            # no change
            msg = "✅ Rejected (manual): no DB update."
        else:
            db.update_issue(issue_id, update_fields, actor=reviewer_id)
            msg = f"✅ Updated (manual): status={db.get_issue(issue_id).status.value}"
    else:
        # langgraph HITL: start -> interrupt -> resume(decision)
        state["hitl_issue_id"] = issue_id
        agent = _ensure_agent(llm, state, reviewer_id=reviewer_id)

        thread_id = f"issue:{issue_id}:{uuid.uuid4()}"
        interrupt_info = start_hitl(agent, thread_id, issue_id, update_fields)
        if not interrupt_info:
            msg = "❌ HITL: No interrupt returned (unexpected)."
        else:
            resume_hitl(agent, thread_id, decision=resume_decision, interrupt_id=interrupt_info.get("id"))
            if decision_type == "reject":
                msg = "✅ Rejected (HITL): tool not executed, DB unchanged."
            else:
                msg = f"✅ Updated (HITL): status={db.get_issue(issue_id).status.value}"

    # Refresh UI
    new_issue = db.get_issue(issue_id)
    table = issues_to_table(db.list_issues())
    detail = format_issue_detail(new_issue)
    logs_md = logs_to_markdown(db.list_logs())
    state["selected_para_index"] = new_issue.para_index if new_issue else None
    doc_md = render_document_markdown(
        paragraphs,
        new_issue.para_index if new_issue else None,
        view_mode=view_mode,
        cached_full=state.get("full_doc_md"),
    )
    location_md = _location_line(paragraphs, new_issue.para_index if new_issue else None)
    jump_md = _jump_link(new_issue.para_index if new_issue else None)
    return state, msg, table, detail, logs_md, doc_md, location_md, jump_md


# ============================================================
# 13) Build Demo rules
# ============================================================

def default_rules() -> List[ReviewRule]:
    return [
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


# ============================================================
# 14) Gradio UI (三栏)
# ============================================================

def build_app():
    llm = init_llm_ollama()
    check_ollama_connection(llm)

    rules = default_rules()
    rules_opts = rules_to_options(rules)

    with gr.Blocks(title="MRO Document Review (Gradio + HITL)") as demo:
        gr.Markdown(
            "# MRO 文档审核 Demo（左规则 / 中文档 / 右概览 + HITL）\n"
            f"- Ollama model: `{CONFIG['OLLAMA_MODEL']}`  \n"
            f"- HITL_MODE: `{_get_hitl_mode()}` (HITL_AVAILABLE={HITL_AVAILABLE})"
        )

        state = gr.State({})

        with gr.Row():
            # Left: rules
            with gr.Column(scale=1):
                gr.Markdown("## 审核规则（左栏）")
                rule_select = gr.CheckboxGroup(
                    choices=rules_opts,
                    value=[],
                    label="选择规则（不选=全部）",
                )
                pdf_file = gr.File(label="上传 PDF（不传则用 CONFIG['PDF_PATH']）", file_types=[".pdf"])
                run_btn = gr.Button("Run Review", variant="primary")
                status = gr.Markdown("（等待运行）")

            # Middle: doc
            with gr.Column(scale=2):
                gr.Markdown("## 文档显示（中栏）")
                view_mode = gr.Radio(
                    choices=["Focused View", "Full Document"],
                    value="Focused View",
                    label="View Mode",
                )
                view_mode_state = gr.State("Focused View")
                with gr.Row():
                    go_query = gr.Textbox(
                        label="Search / Go to paragraph index",
                        placeholder="输入段落索引或关键词…",
                    )
                    go_btn = gr.Button("Go", variant="secondary")
                location_md = gr.Markdown("**Location:** —")
                jump_md = gr.Markdown("")
                doc_md = gr.Markdown(value="（暂无文档内容）", elem_id="doc_view")

            # Right: overview + detail + HITL buttons
            with gr.Column(scale=2):
                gr.Markdown("## 审阅概览（右栏）")

                issues_df = gr.Dataframe(
                    headers=["issue_id", "page", "para_index", "type", "status", "text_snippet"],
                    value=[],
                    interactive=False,
                    wrap=True,
                    max_height=260,
                    label="Issues 概览（点击行选择）",
                )

                issue_detail = gr.Markdown("未选择 Issue。")

                with gr.Row():
                    approve_btn = gr.Button("✅ Approve (Accept)", variant="primary")
                    dismiss_btn = gr.Button("🗑 Dismiss", variant="secondary")
                    reject_btn = gr.Button("❌ Reject", variant="stop")

                edit_fix = gr.Textbox(label="Edit suggested_fix（可选）", placeholder="输入修改后的 suggested_fix …")
                edit_accept_btn = gr.Button("✏️ Edit + Accept")

                hitl_msg = gr.Markdown("（等待操作）")
                logs_md = gr.Markdown("（暂无决策日志）")

        # --- Events ---
        run_btn.click(
            fn=lambda pdf, selected, vm: ui_load_and_review(pdf, selected, llm, rules, vm),
            inputs=[pdf_file, rule_select, view_mode_state],
            outputs=[state, status, doc_md, issues_df, issue_detail, logs_md, location_md, jump_md],
        )

        issues_df.select(
            fn=ui_select_issue,
            inputs=[state, view_mode_state],
            outputs=[state, doc_md, issue_detail, location_md, jump_md],
        )

        view_mode.change(
            fn=ui_change_view,
            inputs=[view_mode, state],
            outputs=[doc_md, location_md, jump_md],
        )

        view_mode.change(
            fn=lambda vm: vm,
            inputs=[view_mode],
            outputs=[view_mode_state],
        )

        go_btn.click(
            fn=ui_go_to_paragraph,
            inputs=[go_query, view_mode_state, state],
            outputs=[state, doc_md, location_md, jump_md],
        )

        approve_btn.click(
            fn=lambda st, vm: _apply_decision_with_hitl(llm, st, "approve_accept", vm),
            inputs=[state, view_mode_state],
            outputs=[state, hitl_msg, issues_df, issue_detail, logs_md, doc_md, location_md, jump_md],
        )

        dismiss_btn.click(
            fn=lambda st, vm: _apply_decision_with_hitl(llm, st, "dismiss", vm),
            inputs=[state, view_mode_state],
            outputs=[state, hitl_msg, issues_df, issue_detail, logs_md, doc_md, location_md, jump_md],
        )

        reject_btn.click(
            fn=lambda st, vm: _apply_decision_with_hitl(llm, st, "reject", vm),
            inputs=[state, view_mode_state],
            outputs=[state, hitl_msg, issues_df, issue_detail, logs_md, doc_md, location_md, jump_md],
        )

        edit_accept_btn.click(
            fn=lambda st, fx, vm: _apply_decision_with_hitl(llm, st, "edit_accept", vm, edited_fix=fx),
            inputs=[state, edit_fix, view_mode_state],
            outputs=[state, hitl_msg, issues_df, issue_detail, logs_md, doc_md, location_md, jump_md],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
