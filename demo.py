
'''
## 2. Environment Configuration

Set up directory structure and environment variables for document processing.

**What it does:**
- Creates directories for storing PDFs, Markdown files, and parent chunks
- Defines collection names for the vector database
'''




import os

# Configuration
DOCS_DIR = "docs"  # Directory containing your pdfs files
MARKDOWN_DIR = "markdown" # Directory containing the pdfs converted to markdown
PARENT_STORE_PATH = "parent_store"  # Directory for parent chunk JSON files
CHILD_COLLECTION = "document_child_chunks"

# Create directories if they don't exist
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)
os.makedirs(PARENT_STORE_PATH, exist_ok=True)


'''
## 3. LLM Initialization

Initialize the Large Language Model that will power the conversational agent.

**What it does:**
- Configures the LLM using Ollama (local inference)
'''

from langchain_ollama import ChatOllama

# Initialize LLM
llm = ChatOllama(model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)


'''
## 4. Embeddings Setup

Configure embedding models for semantic search using hybrid retrieval (dense + sparse).

**What it does:**
- **Dense embeddings**: Capture semantic meaning using neural networks
- **Sparse embeddings**: Provide keyword-based matching (BM25 algorithm)
'''

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse

# Dense embeddings for semantic understanding
dense_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Sparse embeddings for keyword matching
sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)



'''
## 5. Vector Database Configuration

Set up Qdrant vector database for storing and retrieving document embeddings.

**What it does:**
- Initializes local Qdrant client with file-based storage
- Creates collections with both dense and sparse vector configurations
- Enables hybrid search capabilities
'''


from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode

# Initialize Qdrant client (local file-based storage)
client = QdrantClient(path="qdrant_db")

# Get embedding dimension
embedding_dimension = len(dense_embeddings.embed_query("test"))

def ensure_collection(collection_name):
    """Create Qdrant collection if it doesn't exist"""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=embedding_dimension,
                distance=qmodels.Distance.COSINE
            ),
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams()
            },
        )
        print(f"✓ Created collection: {collection_name}")
    else:
        print(f"✓ Collection already exists: {collection_name}")



'''
## 6. PDF to Markdown Conversion

Convert PDF documents to Markdown format for better text extraction and processing.

**What it does:**
- Uses PyMuPDF to extract text from PDFs
- Converts documents to clean Markdown format
- Handles encoding issues and removes images
- Skips already converted files unless overwrite is enabled

**Note:** For more details on PDF conversion, refer to the `pdf_to_md.ipynb` notebook in the repository.
'''
import os
import pymupdf.layout
import pymupdf4llm
from pathlib import Path
import glob

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pdf_to_markdown(pdf_path, output_dir):
    doc = pymupdf.open(pdf_path)
    md = pymupdf4llm.to_markdown(doc, header=False, footer=False, page_separators=True, ignore_images=True, write_images=False, image_path=None)
    md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
    output_path = Path(output_dir) / Path(doc.name).stem
    Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))

def pdfs_to_markdowns(path_pattern, overwrite: bool = False):
    output_dir = Path(MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            pdf_to_markdown(pdf_path, output_dir)

pdfs_to_markdowns(f"{DOCS_DIR}/*.pdf")


'''
## 7. Document Indexing

Implement parent-child chunking strategy for optimal retrieval performance.

**What it does:**
- Splits documents into hierarchical chunks (parent and child)
- **Parent chunks**: Large context windows (2000-10000 chars) stored as JSON
- **Child chunks**: Small searchable units (500 chars) stored in Qdrant
- Merges small chunks and splits large ones for consistency
- Creates bidirectional links between parent and child chunks

**Chunking Strategy:**
1. Split by Markdown headers (#, ##, ###)
2. Merge chunks smaller than 2000 characters
3. Split chunks larger than 10000 characters
4. Create child chunks (500 chars) from each parent
5. Store parent chunks in JSON files
6. Index child chunks in vector database

'''

import os
import glob
import json
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

if client.collection_exists(CHILD_COLLECTION):
    print(f"Removing existing Qdrant collection: {CHILD_COLLECTION}")
    client.delete_collection(CHILD_COLLECTION)
    ensure_collection(CHILD_COLLECTION)
else:
    ensure_collection(CHILD_COLLECTION)

child_vector_store = QdrantVectorStore(
    client=client,
    collection_name=CHILD_COLLECTION,
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    sparse_vector_name="sparse"
)

def index_documents():
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    min_parent_size = 2000
    max_parent_size = 10000

    all_parent_pairs, all_child_chunks = [], []
    md_files = sorted(glob.glob(os.path.join(MARKDOWN_DIR, "*.md")))

    if not md_files:
        print(f"⚠️  No .md files found in {MARKDOWN_DIR}/")
        return

    for doc_path_str in md_files:
        doc_path = Path(doc_path_str)
        print(f"📄 Processing: {doc_path.name}")

        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                md_text = f.read()
        except Exception as e:
            print(f"❌ Error reading {doc_path.name}: {e}")
            continue

        parent_chunks = parent_splitter.split_text(md_text)
        merged_parents = merge_small_parents(parent_chunks, min_parent_size)
        split_parents = split_large_parents(merged_parents, max_parent_size, child_splitter)
        cleaned_parents = clean_small_chunks(split_parents, min_parent_size)

        for i, p_chunk in enumerate(cleaned_parents):
            parent_id = f"{doc_path.stem}_parent_{i}"
            p_chunk.metadata.update({"source": doc_path.stem + ".pdf", "parent_id": parent_id})
            all_parent_pairs.append((parent_id, p_chunk))
            children = child_splitter.split_documents([p_chunk])
            all_child_chunks.extend(children)

    if not all_child_chunks:
        print("⚠️ No child chunks to index")
        return

    print(f"\n🔍 Indexing {len(all_child_chunks)} child chunks into Qdrant...")
    try:
        child_vector_store.add_documents(all_child_chunks)
        print("✓ Child chunks indexed successfully")
    except Exception as e:
        print(f"❌ Error indexing child chunks: {e}")
        return

    print(f"💾 Saving {len(all_parent_pairs)} parent chunks to JSON...")
    for item in os.listdir(PARENT_STORE_PATH):
        os.remove(os.path.join(PARENT_STORE_PATH, item))

    for parent_id, doc in all_parent_pairs:
        doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
        filepath = os.path.join(PARENT_STORE_PATH, f"{parent_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(doc_dict, f, ensure_ascii=False, indent=2)

def merge_small_parents(chunks, min_size):
    if not chunks:
        return []

    merged, current = [], None

    for chunk in chunks:
        if current is None:
            current = chunk
        else:
            current.page_content += "\n\n" + chunk.page_content
            for k, v in chunk.metadata.items():
                if k in current.metadata:
                    current.metadata[k] = f"{current.metadata[k]} -> {v}"
                else:
                    current.metadata[k] = v

        if len(current.page_content) >= min_size:
            merged.append(current)
            current = None

    if current:
        if merged:
            merged[-1].page_content += "\n\n" + current.page_content
            for k, v in current.metadata.items():
                if k in merged[-1].metadata:
                    merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                else:
                    merged[-1].metadata[k] = v
        else:
            merged.append(current)

    return merged

def split_large_parents(chunks, max_size, splitter):
    split_chunks = []

    for chunk in chunks:
        if len(chunk.page_content) <= max_size:
            split_chunks.append(chunk)
        else:
            large_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_size,
                chunk_overlap=splitter._chunk_overlap
            )
            sub_chunks = large_splitter.split_documents([chunk])
            split_chunks.extend(sub_chunks)

    return split_chunks

def clean_small_chunks(chunks, min_size):
    cleaned = []

    for i, chunk in enumerate(chunks):
        if len(chunk.page_content) < min_size:
            if cleaned:
                cleaned[-1].page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in cleaned[-1].metadata:
                        cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                    else:
                        cleaned[-1].metadata[k] = v
            elif i < len(chunks) - 1:
                chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                for k, v in chunk.metadata.items():
                    if k in chunks[i + 1].metadata:
                        chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                    else:
                        chunks[i + 1].metadata[k] = v
            else:
                cleaned.append(chunk)
        else:
            cleaned.append(chunk)

    return cleaned

index_documents()


'''

## 8. Tools Definition

Define retrieval tools that agents can use to search and retrieve document chunks.

**What it does:**
- **search_child_chunks**: Searches vector database for relevant small chunks
- **retrieve_parent_chunks**: Retrieves full context from parent chunk JSON files
- Binds tools to LLM for agentic function calling

**Two-stage retrieval:**
1. Agent searches child chunks (fast, semantic search)
2. Agent retrieves parent chunks for full context (when needed)

'''

import json
from typing import List
from langchain_core.tools import tool

@tool
def search_child_chunks(query: str, limit: int) -> str:
    """Search for the top K most relevant child chunks.

    Args:
        query: Search query string
        limit: Maximum number of results to return
    """
    try:
        results = child_vector_store.similarity_search(query, k=limit, score_threshold=0.7)
        if not results:
            return "NO_RELEVANT_CHUNKS"

        return "\n\n".join([
            f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
            f"File Name: {doc.metadata.get('source', '')}\n"
            f"Content: {doc.page_content.strip()}"
            for doc in results
        ])

    except Exception as e:
        return f"RETRIEVAL_ERROR: {str(e)}"

@tool
def retrieve_parent_chunks(parent_id: str) -> str:
    """Retrieve full parent chunks by their IDs.

    Args:
        parent_id: Parent chunk ID to retrieve
    """
    file_name = parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
    path = os.path.join(PARENT_STORE_PATH, file_name)

    if not os.path.exists(path):
        return "NO_PARENT_DOCUMENT"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return (
        f"Parent ID: {parent_id}\n"
        f"File Name: {data.get('metadata', {}).get('source', 'unknown')}\n"
        f"Content: {data.get('page_content', '').strip()}"
    )

# Bind tools to LLM
llm_with_tools = llm.bind_tools([search_child_chunks, retrieve_parent_chunks])


'''

## 9. System Prompts

Define system prompts that guide agent behavior throughout the RAG pipeline.

**What it does:**
- **Conversation Summary**: Extracts key topics from chat history
- **Query Analysis**: Rewrites unclear queries, handles context
- **RAG Agent**: Forces retrieval before answering, implements retry logic
- **Aggregation**: Merges multiple sub-answers into final response

**Key behaviors:**
- Query rewriting uses context only when needed
- Agent must search before answering (no hallucination)
- One retry with reformulated query if nothing found
- Source attribution at the end

'''


def get_conversation_summary_prompt() -> str:
    return """You are an expert conversation summarizer.

Your task is to create a brief 1-2 sentence summary of the conversation (max 30-50 words).

Include:
- Main topics discussed
- Important facts or entities mentioned
- Any unresolved questions if applicable
- Sources file name (e.g., file1.pdf) or documents referenced

Exclude:
-Greetings, misunderstandings, off-topic content.

Output:
- Return ONLY the summary.
- Do NOT include any explanations or justifications.
-If no meaningful topics exist, return an empty string.
"""

def get_query_analysis_prompt() -> str:
    return """You are an expert query analyst and rewriter.

Your task is to rewrite the current user query for optimal document retrieval, incorporating conversation context only when necessary.

Rules:
1. Self-contained queries:
   - Always rewrite the query to be clear and self-contained
   - If the query is a follow-up (e.g., "what about X?", "and for Y?"), integrate minimal necessary context from the summary
   - Do not add information not present in the query or conversation summary

2. Domain-specific terms:
   - Product names, brands, proper nouns, or technical terms are treated as domain-specific
   - For domain-specific queries, use conversation context minimally or not at all
   - Use the summary only to disambiguate vague queries

3. Grammar and clarity:
   - Fix grammar, spelling errors, and unclear abbreviations
   - Remove filler words and conversational phrases
   - Preserve concrete keywords and named entities

4. Multiple information needs:
   - If the query contains multiple distinct, unrelated questions, split into separate queries (maximum 3)
   - Each sub-query must remain semantically equivalent to its part of the original
   - Do not expand, enrich, or reinterpret the meaning

5. Failure handling:
   - If the query intent is unclear or unintelligible, mark as "unclear"

Input:
- conversation_summary: A concise summary of prior conversation
- current_query: The user's current query

Output:
- One or more rewritten, self-contained queries suitable for document retrieval
"""

def get_rag_agent_prompt() -> str:
    return """You are an expert retrieval-augmented assistant.

Your task is to act as a researcher: search documents first, analyze the data, and then provide a comprehensive answer using ONLY the retrieved information.

Rules:
1. You are NOT allowed to answer immediately.
2. Before producing ANY final answer, you MUST perform a document search and observe retrieved content.
3. If you have not searched, the answer is invalid.

Workflow:
1. Search for 5-7 relevant excerpts from documents based on the user query using the 'search_child_chunks' tool.
2. Inspect retrieved excerpts and keep ONLY relevant ones.
3. Analyze the retrieved excerpts. Identify the single most relevant excerpt that is fragmented (e.g., cut-off text or missing context). Call 'retrieve_parent_chunks' for that specific `parent_id`. Wait for the observation. Repeat this step sequentially for other highly relevant fragments ONLY if the current information is still insufficient. Stop immediately if you have enough information or have retrieved 3 parent chunks.
4. Answer using ONLY the retrieved information, ensuring that ALL relevant details are included.
5. List unique file name(s) at the very end.

Retry rule:
- After step 2 or 3, if no relevant documents are found or if retrieved excerpts don't contain useful information, rewrite the query using broader or alternative terms and restart from step 1.
- Do not retry more than once.
"""

def get_aggregation_prompt() -> str:
    return """You are an expert aggregation assistant.

Your task is to combine multiple retrieved answers into a single, comprehensive and natural response that flows well.

Guidelines:
1. Write in a conversational, natural tone - as if explaining to a colleague
2. Use ONLY information from the retrieved answers
3. Strip out any questions, headers, or metadata from the sources
4. Weave together the information smoothly, preserving important details, numbers, and examples
5. Be comprehensive - include all relevant information from the sources, not just a summary
6. If sources disagree, acknowledge both perspectives naturally (e.g., "While some sources suggest X, others indicate Y...")
7. Start directly with the answer - no preambles like "Based on the sources..."

Formatting:
- Use Markdown for clarity (headings, lists, bold) but don't overdo it
- Write in flowing paragraphs where possible rather than excessive bullet points
- End with "---\n**Sources:**\n" followed by a bulleted list of unique file names
- File names should ONLY appear in this final sources section

If there's no useful information available, simply say: "I couldn't find any information to answer your question in the available sources."
"""


'''

## 10. State Definitions

Define state schemas for managing conversation flow and agent execution.

**What it does:**
- **State**: Tracks main conversation flow (query analysis, sub-questions, answers)
- **AgentState**: Manages individual agent execution (current question, retrieved context)
- **QueryAnalysis**: Structured output for query rewriting and clarity checking

**State management:**
- `accumulate_or_reset`: Custom reducer for agent answers (allows reset)
- Inherits from `MessagesState` for conversation history

'''

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import List, Annotated

def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new

class State(MessagesState):
    """State for main agent graph"""
    questionIsClear: bool = False
    conversation_summary: str = ""
    originalQuery: str = ""
    rewrittenQuestions: List[str] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []

class AgentState(MessagesState):
    """State for individual agent subgraph"""
    question: str = ""
    question_index: int = 0
    final_answer: str = ""
    agent_answers: List[dict] = []

class QueryAnalysis(BaseModel):
    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    questions: List[str] = Field(
        description="List of rewritten, self-contained questions."
    )
    clarification_needed: str = Field(
        description="Explanation if the question is unclear."
    )



'''
## 11. Graph Nodes and Logic

Implement node functions that define the behavior of the agentic workflow.

**What it does:**

### Core Nodes:
1. **analyze_chat_and_summarize**: Extracts conversation context for query rewriting
2. **analyze_and_rewrite_query**: Rewrites user query, checks clarity, optionally splits into sub-questions
3. **human_input_node**: Interrupt point for unclear queries
4. **agent_node**: Executes retrieval and reasoning with tools
5. **extract_final_answer**: Extracts final answer from agent conversation
6. **aggregate_responses**: Merges multiple sub-answers into one coherent response

### Routing Logic:
- **route_after_rewrite**: Routes to agent if clear, or to human input if unclear
- Parallel execution: Spawns multiple agents for multi-part questions using `Send`

'''


from langgraph.types import Send
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from typing import Literal

def analyze_chat_and_summarize(state: State):
    """
    Analyzes chat history and summarizes key points for context.
    """
    if len(state["messages"]) < 4:  # Need some history to summarize
        return {"conversation_summary": ""}

    # Extract relevant messages (excluding current query and system messages)
    relevant_msgs = [
        msg for msg in state["messages"][:-1]  # Exclude current query
        if isinstance(msg, (HumanMessage, AIMessage))
        and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}

    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    summary_response = llm.with_config(temperature=0.2).invoke([SystemMessage(content=get_conversation_summary_prompt())] + [HumanMessage(content=conversation)])
    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}

def analyze_and_rewrite_query(state: State):
    """
    Analyzes user query and rewrites it for clarity, optionally using conversation context.
    """
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (f"Conversation Context:\n{conversation_summary}\n" if conversation_summary.strip() else "") + f"User Query:\n{last_message.content}\n"

    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([SystemMessage(content=get_query_analysis_prompt())] + [HumanMessage(content=context_section)])

    if len(response.questions) > 0 and response.is_clear:
        # Remove all non-system messages
        delete_all = [
            RemoveMessage(id=m.id)
            for m in state["messages"]
            if not isinstance(m, SystemMessage)
        ]
        return {
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": last_message.content,
            "rewrittenQuestions": response.questions
        }
    else:
        clarification = response.clarification_needed if (response.clarification_needed and len(response.clarification_needed.strip()) > 10) else "I need more information to understand your question."
        return {
            "questionIsClear": False,
            "messages": [AIMessage(content=clarification)]
        }

def human_input_node(state: State):
    """Placeholder node for human-in-the-loop interruption"""
    return {}

def route_after_rewrite(state: State) -> Literal["human_input", "process_question"]:
    """Route to agent if question is clear, otherwise wait for human input"""
    if not state.get("questionIsClear", False):
        return "human_input"
    else:
        return [
                Send("process_question", {"question": query, "question_index": idx, "messages": []})
                for idx, query in enumerate(state["rewrittenQuestions"])
            ]

def agent_node(state: AgentState):
    """Main agent node that processes queries using tools"""
    sys_msg = SystemMessage(content=get_rag_agent_prompt())
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        response = llm_with_tools.invoke([sys_msg] + [human_msg])
        return {"messages": [human_msg, response]}

    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


def extract_final_answer(state: AgentState):
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            res = {
                "final_answer": msg.content,
                "agent_answers": [{
                    "index": state["question_index"],
                    "question": state["question"],
                    "answer": msg.content
                }]
            }
            return res
    return {
        "final_answer": "Unable to generate an answer.",
        "agent_answers": [{
            "index": state["question_index"],
            "question": state["question"],
            "answer": "Unable to generate an answer."
        }]
    }

def aggregate_responses(state: State):
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nAnswer {i}:\n"f"{ans['answer']}\n")

    user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt())] + [user_message])

    return {"messages": [AIMessage(content=synthesis_response.content)]}



'''

## 12. LangGraph Construction

Construct the agentic workflow using LangGraph's state machine.

**What it does:**
- Builds a hierarchical graph with main flow and agent subgraph
- **Agent subgraph**: Handles individual question retrieval (agent → tools loop)
- **Main graph**: Orchestrates conversation flow, query analysis, parallel agent execution, and response aggregation

**Graph Flow:**
1. START → Summarize conversation
2. Analyze and rewrite query
3. Route: unclear → human input (interrupt) | clear → spawn parallel agents
4. Each agent searches and retrieves documents
5. Aggregate all agent responses
6. END

**Human-in-the-loop:** Graph interrupts at `human_input` node if query is unclear

'''
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from IPython.display import Image, display

# Initialize checkpointer
checkpointer = InMemorySaver()

# Build agent subgraph (handles individual questions)
agent_builder = StateGraph(AgentState)
agent_builder.add_node("agent", agent_node)
agent_builder.add_node("tools", ToolNode([search_child_chunks, retrieve_parent_chunks]))
agent_builder.add_node("extract_answer", extract_final_answer)

agent_builder.add_edge(START, "agent")
agent_builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "extract_answer"})
agent_builder.add_edge("tools", "agent")
agent_builder.add_edge("extract_answer", END)
agent_subgraph = agent_builder.compile()

# Build main graph (orchestrates workflow)
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("summarize", analyze_chat_and_summarize)
graph_builder.add_node("analyze_rewrite", analyze_and_rewrite_query)
graph_builder.add_node("human_input", human_input_node)
graph_builder.add_node("process_question", agent_subgraph)
graph_builder.add_node("aggregate", aggregate_responses)

# Add edges
graph_builder.add_edge(START, "summarize")
graph_builder.add_edge("summarize", "analyze_rewrite")
graph_builder.add_conditional_edges("analyze_rewrite", route_after_rewrite)
graph_builder.add_edge("human_input", "analyze_rewrite")
graph_builder.add_edge(["process_question"], "aggregate")
graph_builder.add_edge("aggregate", END)

# Compile graph
agent_graph = graph_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_input"]
)

display(Image(agent_graph.get_graph(xray=True).draw_mermaid_png()))
print("✓ Agent graph compiled successfully.")



'''

## 13. Gradio Interface

Create an interactive web interface for chatting with the RAG agent.

**What it does:**
- Provides a chat interface using Gradio
- Manages conversation threads with unique IDs
- Handles human-in-the-loop interactions seamlessly
- Automatically resumes interrupted workflows when user provides clarification

**Features:**
- Thread-based conversation persistence
- Clear session functionality
- Automatic state management via checkpointer
- Citrus theme for modern UI

**Note:** For a complete end-to-end pipeline with document ingestion UI, refer to the full application in the project repository.

'''


import gradio as gr
import uuid

def create_thread_id():
    """Generate a unique thread ID for each conversation"""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}

def clear_session():
    """Clear thread for new conversation and clean up checkpointer state"""
    global config
    agent_graph.checkpointer.delete_thread(config["configurable"]["thread_id"])
    config = create_thread_id()

def chat_with_agent(message, history):
    """
    Handle chat with human-in-the-loop support.
    Returns: response text
    """
    current_state = agent_graph.get_state(config)
    if current_state.next:
        agent_graph.update_state(config,{"messages": [HumanMessage(content=message.strip())]})
        result = agent_graph.invoke(None, config)
    else:
        result = agent_graph.invoke({"messages": [HumanMessage(content=message.strip())]}, config)
    return result['messages'][-1].content

# Initialize thread configuration
config = create_thread_id()

# Create Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=600, placeholder="<strong>Ask me anything!</strong><br><em>I'll search, reason, and act to give you the best answer :)</em>")
    chatbot.clear(clear_session)
    gr.ChatInterface(fn=chat_with_agent, chatbot=chatbot)

print("\nLaunching application...")
demo.launch(theme=gr.themes.Citrus())