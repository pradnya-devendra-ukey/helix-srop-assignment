"""
AccountAgent — answers account-specific questions using mock DB tools.

Called by the root orchestrator via AgentTool when the user asks about their
builds, account status, or usage.
"""
from google.adk.agents import LlmAgent

from app.agents.tools.account_tools import get_account_status, get_recent_builds
from app.settings import settings

ACCOUNT_INSTRUCTION = """
You are the Helix Account Agent — a specialist for account and build information.

Your job: answer questions about the user's builds, account status, and usage
limits by calling the appropriate tool.

Rules:
1. Use get_recent_builds when the user asks about their builds, pipelines, or
   CI/CD history.
2. Use get_account_status when the user asks about their plan, limits, or usage.
3. The user_id will be provided in the conversation context — pass it to the tool.
4. Present data clearly. Use markdown tables or lists where appropriate.
5. Do not fabricate data — only report what the tools return.
""".strip()

account_agent = LlmAgent(
    name="account_agent",
    model=settings.adk_model,
    instruction=ACCOUNT_INSTRUCTION,
    tools=[get_recent_builds, get_account_status],
)
