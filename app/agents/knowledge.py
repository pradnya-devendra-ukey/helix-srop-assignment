"""
KnowledgeAgent — answers Helix product questions via RAG.

Called by the root orchestrator via AgentTool when the user asks a HOW/WHAT
question about the product. Always cites chunk IDs in its response.
"""
from google.adk.agents import LlmAgent

from app.agents.tools.search_docs import search_docs
from app.settings import settings

KNOWLEDGE_INSTRUCTION = """
You are the Helix Knowledge Agent — a product documentation specialist.

Your job: answer questions about the Helix platform using ONLY the context
chunks returned by the search_docs tool.

Rules:
1. Always call search_docs first before answering.
2. Cite every chunk you use: write [chunk_id] inline, e.g.
   "According to [chunk_abc12345678901234], you can rotate a key by..."
3. If search_docs returns no relevant results, say:
   "I don't have documentation on that topic."
4. Never guess or use knowledge outside the provided context.
5. Be concise and direct. Use markdown formatting where helpful.
""".strip()

knowledge_agent = LlmAgent(
    name="knowledge_agent",
    model=settings.adk_model,
    instruction=KNOWLEDGE_INSTRUCTION,
    tools=[search_docs],
)
