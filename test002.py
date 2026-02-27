from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Generator, List, Literal, Optional

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
    # 推荐：qwen2.5:7b-instruct / llama3.1:8b / mistral:7b-instruct
    "OLLAMA_MODEL": "qwen3:4b-instruct-2507-q4_K_M",

    # (3) 你的 PDF 路径
    "PDF_PATH": "./data/ss123.pdf",

    # (4) Chunk size：每次给模型多少段落（越大越慢、越容易超上下文）
    "CHUNK_SIZE": 20,

    # (5) 上下文长度（取决于模型支持；越大占用越高）
    "NUM_CTX": 8192,

    # (6) 温度（越低越稳定）
    "TEMPERATURE": 0.2,

    # (7) 是否打印模型原始输出（调试用）
    "DEBUG_PRINT_RAW_MODEL_OUTPUT": False,

    # (8) Docling 最大页数（None=不限制，Demo 时可以先设 5）
    "DOCLING_MAX_PAGES": None,
}

# ============================================================
# 1) PDF Parsing (Docling 2.75.0)  —— replace MinerU
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
    Docling 2.75.0 local PDF parse -> list of page dict:
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
                if max_pages is not None and int(page_num) > max_pages:
                    break

                page_text = page.get("text") or page.get("content_text")
                if not page_text:
                    blocks = page.get("paragraphs") or page.get("blocks") or []
                    if isinstance(blocks, list):
                        page_text = "\n".join(
                            (b.get("text") or b.get("content") or "").strip()
                            for b in blocks
                            if isinstance(b, dict) and (b.get("text") or b.get("content"))
                        )

                out_pages.append({"page_num": int(page_num), "content": normalize_text(page_text or "")})
            elif isinstance(page, str):
                if max_pages is not None and idx > max_pages:
                    break
                out_pages.append({"page_num": idx, "content": normalize_text(page)})
            else:
                if max_pages is not None and idx > max_pages:
                    break
                out_pages.append({"page_num": idx, "content": normalize_text(str(page))})
        if out_pages:
            return out_pages

    md = _try_export_markdown(doc)
    if md:
        return [{"page_num": 1, "content": normalize_text(md)}]

    return [{"page_num": 1, "content": normalize_text(str(doc))}]


# ============================================================
# 2) LLM init (Ollama)
# ============================================================

def init_llm_ollama() -> ChatOllama:
    llm = ChatOllama(
        model=CONFIG["OLLAMA_MODEL"],
        base_url=CONFIG["OLLAMA_BASE_URL"],
        temperature=CONFIG["TEMPERATURE"],
        model_kwargs={"num_ctx": CONFIG["NUM_CTX"]},
    )
    return llm


def check_ollama_connection(llm: ChatOllama) -> None:
    """
    Simple connectivity test; raises if Ollama isn't reachable or model missing.
    """
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
# 3) Output schemas
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

BASE_SYSTEM_PROMPT = """你是一位专业的文档审核专家。
请仔细审查提供的文本，识别其中的问题。

需要检查的问题类型：
- 语法错误：错别字、标点符号错误、语病等
- 用词不当：使用了不恰当的词语或表达
- 敏感表述：使用了"必须"、"保证"、"一定"、"绝对"等过度承诺的措辞

注意事项：
1. 文档可能是中文或英文，请根据语言选择合适的审核标准
2. 使用输入中提供的段落索引（如 [0], [1], ...）来标识问题位置
3. 每个问题都需要提供具体的修改建议
4. 如果没有发现问题，返回空的问题列表
5. 按照要求的 JSON 格式输出结果
"""


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

    prompt = f"""你是一位专业的文档审核专家。
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
    return prompt


def build_user_prompt(paragraphs: list[dict], parser: PydanticOutputParser) -> str:
    formatted_text = "\n".join([f"[{i}] {p['content']}" for i, p in enumerate(paragraphs)])
    user_prompt = f"""请审核以下文本内容：

{formatted_text}

如果发现问题，请按以下格式输出；如果没有问题，返回空的 issues 列表。

{parser.get_format_instructions()}
"""
    return user_prompt


# ============================================================
# 6) Robust JSON parsing helper (for Ollama)
# ============================================================

def parse_llm_output_to_reviewoutput(parser: PydanticOutputParser, raw: str) -> ReviewOutput:
    """
    Ollama models sometimes wrap JSON with extra text.
    We try:
      1) direct parser.parse()
      2) extract first {...} block and parse
      3) return empty
    """
    raw = (raw or "").strip()
    try:
        return parser.parse(raw)
    except Exception:
        pass

    # Try to extract a JSON object from the text
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        candidate = m.group(0).strip()
        try:
            # validate JSON
            json.loads(candidate)
            return parser.parse(candidate)
        except Exception:
            pass

    return ReviewOutput(issues=[])


# ============================================================
# 7) Review functions (Ollama)
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


def review_pdf_document(
    pdf_path: str,
    llm: ChatOllama,
    custom_rules: list[ReviewRule] = None,
    chunk_size: int = 20,
    max_pages: Optional[int] = None,
) -> list[ReviewIssue]:
    print("开始文档审核流程")
    print("=" * 60)

    print("\n步骤1：解析 PDF 文档（Docling，本地）")
    pages = extract_text_from_pdf_docling(pdf_path, max_pages=max_pages)
    print(f"提取了 {len(pages)} 页内容")

    all_paragraphs: list[dict] = []
    for page in pages:
        for para in split_into_paragraphs(page["content"]):
            if para.strip():
                all_paragraphs.append({"content": para.strip(), "page_num": page["page_num"]})
    print(f"  共 {len(all_paragraphs)} 个段落")

    print("\n步骤2：执行文档审核（Ollama + 分块）")
    all_issues: list[ReviewIssue] = []
    for result in stream_review_document(all_paragraphs, llm, custom_rules, chunk_size):
        all_issues.extend(result.issues)

    print("\n" + "=" * 60)
    print(f"审核完成！共发现 {len(all_issues)} 个问题")
    return all_issues


# ============================================================
# 8) Report generator
# ============================================================

def generate_report(issues: list[ReviewIssue], output_path: str = None) -> str:
    type_counts = {}
    for issue in issues:
        type_counts[issue.type] = type_counts.get(issue.type, 0) + 1

    report = []
    report.append("# 文档审核报告")
    report.append("")
    report.append("## 概要")
    report.append(f"- 发现问题总数：**{len(issues)}**")
    report.append("")
    report.append("## 问题类型分布")
    for issue_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        report.append(f"- {issue_type}：{count} 个")
    report.append("")
    report.append("## 问题详情")
    for i, issue in enumerate(issues, 1):
        report.append(f"\n### 问题 {i}")
        report.append(f"- **类型**：{issue.type}")
        report.append(f"- **位置**：段落 [{issue.para_index}]")
        report.append(f"- **原文**：{issue.text}")
        report.append(f"- **说明**：{issue.explanation}")
        report.append(f"- **建议**：{issue.suggested_fix}")

    report_text = "\n".join(report)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"报告已保存到：{output_path}")

    return report_text


# ============================================================
# 9) Demo main
# ============================================================

if __name__ == "__main__":
    # Init Ollama model
    llm = init_llm_ollama()

    # Check connectivity (fail fast)
    check_ollama_connection(llm)
    print(f"✅ Ollama ready: base_url={CONFIG['OLLAMA_BASE_URL']} model={CONFIG['OLLAMA_MODEL']}")

    # Quick LLM test
    resp = llm.invoke([HumanMessage(content="你好，请用一句话介绍你自己。请不要输出多余内容。")])
    print(f"\n[LLM 测试] 模型回复：{resp.content}\n")

    # Sample custom rules (你可替换为 MRO 规则库)
    sample_rules = [
        ReviewRule(
            name="夸大宣传",
            description="检查是否有夸大产品效果或功能的表述，如'最好'、'第一'、'独家'等",
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

    issues = review_pdf_document(
        pdf_path=CONFIG["PDF_PATH"],
        llm=llm,
        custom_rules=sample_rules,
        chunk_size=CONFIG["CHUNK_SIZE"],
        max_pages=CONFIG["DOCLING_MAX_PAGES"],
    )

    print("\n" + "=" * 60)
    print("审核结果：")
    print("=" * 60)

    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"\n问题 {i}：")
            print(f"   类型：{issue.type}")
            print(f"   位置：段落 [{issue.para_index}]")
            print(f"   原文：{issue.text}")
            print(f"   说明：{issue.explanation}")
            print(f"   建议：{issue.suggested_fix}")
    else:
        print("未发现问题")

    report = generate_report(issues, output_path="./output/review_report.md")
    print("\n[报告预览]\n")
    print(report[:1200] + ("\n...\n" if len(report) > 1200 else ""))
