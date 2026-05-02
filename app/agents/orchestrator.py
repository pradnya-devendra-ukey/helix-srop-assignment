"""
SROP Root Orchestrator — Google ADK agent.

Routes every user turn to KnowledgeAgent or AccountAgent via ADK's AgentTool.
The LLM decides which tool to call — no string parsing.

Intent → sub-agent:
  knowledge:  HOW/WHAT questions, docs/feature questions → knowledge_agent
  account:    builds, account status, usage → account_agent
  smalltalk:  greetings, thanks, off-topic → inline reply (no tool call)

Guardrail: out-of-scope queries (poems, personal advice, unrelated tasks)
are refused inline.

Session context (user_id, plan_tier, turn_count, last_agent) is injected
dynamically into the instruction at pipeline.run() time (Pattern C).
"""
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from app.agents.account import account_agent
from app.agents.knowledge import knowledge_agent
from app.settings import settings

ROOT_INSTRUCTION_TEMPLATE = """
You are the Helix Support Concierge — a routing agent for the Helix B2B dev-tools platform.

## Current user context
- user_id: {user_id}
- plan_tier: {plan_tier}
- turn_count: {turn_count}
- last_agent_used: {last_agent}

## Routing rules
Call the correct specialist tool based on the user's intent:
- HOW to do something, WHAT something is, docs/feature/configuration questions
  → call knowledge_agent
- Their account info, builds, CI/CD history, plan status, usage limits
  → call account_agent
- Greetings or brief acknowledgements → respond directly, no tool needed

## Guardrails — Refusal policy
If the user asks for something completely unrelated to Helix
(e.g. write a poem, personal advice, cooking recipes, general coding help
unrelated to Helix), respond ONLY with:
"I can only assist with Helix product questions and your account information."
Do not call any tool for out-of-scope requests.

## Important
- Always call a tool when the intent matches knowledge or account.
- Never answer knowledge questions from your own training — always use knowledge_agent.
- The user_id above is already known — do not ask the user to provide it again.
""".strip()


def build_root_agent(
    user_id: str,
    plan_tier: str,
    turn_count: int,
    last_agent: str | None,
) -> LlmAgent:
    """
    Build the root orchestrator with session context injected into the instruction.

    A new LlmAgent is created per turn so the instruction reflects current state.
    Sub-agents are module-level singletons (no overhead there).
    """
    instruction = ROOT_INSTRUCTION_TEMPLATE.format(
        user_id=user_id,
        plan_tier=plan_tier,
        turn_count=turn_count,
        last_agent=last_agent or "none",
    )
    return LlmAgent(
        name="srop_root",
        model=settings.adk_model,
        instruction=instruction,
        tools=[
            AgentTool(agent=knowledge_agent),
            AgentTool(agent=account_agent),
        ],
    )
