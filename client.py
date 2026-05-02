"""
Helix SROP — Interactive Demo Client
=====================================
Run this to walk through all API features in one shot.

Usage:
    python client.py                  # full demo flow
    python client.py --base http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import textwrap
from typing import Any

# Force UTF-8 on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

import httpx

# ── Colours (graceful fallback on Windows without ANSI support) ──────────────
try:
    import colorama; colorama.init()
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
except ImportError:
    CYAN = GREEN = YELLOW = RED = BOLD = RESET = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def header(msg: str) -> None:
    print(f"\n{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}{msg}{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}")


def ok(label: str, value: Any = "") -> None:
    print(f"  {GREEN}[OK] {label}{RESET}", value if value else "")


def fail(label: str, value: Any = "") -> None:
    print(f"  {RED}[FAIL] {label}{RESET}", value if value else "")
    sys.exit(1)


def info(label: str, value: Any = "") -> None:
    print(f"  {YELLOW}>> {label}{RESET}", value if value else "")


def pretty_json(data: dict | list) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def wrap_reply(text: str, width: int = 80) -> str:
    return textwrap.fill(text, width=width, subsequent_indent="    ")


def post(client: httpx.Client, path: str, **kwargs) -> httpx.Response:
    r = client.post(path, **kwargs)
    return r


def get(client: httpx.Client, path: str) -> httpx.Response:
    return client.get(path)


# ── Test steps ────────────────────────────────────────────────────────────────

def test_healthz(client: httpx.Client) -> None:
    header("TEST 1 — GET /healthz")
    r = get(client, "/healthz")
    if r.status_code == 200 and r.json().get("status") == "ok":
        ok("Server is healthy", r.json())
    else:
        fail("Healthz failed", r.text)


def test_create_session(client: httpx.Client) -> str:
    header("TEST 2 — POST /v1/sessions")
    payload = {"user_id": "alice_demo", "plan_tier": "pro"}
    info("Request body", pretty_json(payload))

    r = post(client, "/v1/sessions", json=payload)
    data = r.json()
    info("Response", pretty_json(data))

    if r.status_code == 200 and "session_id" in data:
        ok("Session created")
        ok("user_id", data["user_id"])
        ok("session_id", data["session_id"])
        return data["session_id"]
    else:
        fail("Session creation failed", r.text)
        return ""


def test_session_not_found(client: httpx.Client) -> None:
    header("TEST 3 — 404 on missing session")
    r = post(client, "/v1/chat/no-such-session-id", json={"content": "hello"})
    data = r.json()
    info("Response", pretty_json(data))

    if r.status_code == 404 and data.get("title") == "SESSION_NOT_FOUND":
        ok("Correct 404 SESSION_NOT_FOUND")
    else:
        fail("Expected 404", r.text)


def test_knowledge_turn(client: httpx.Client, session_id: str) -> str:
    header("TEST 4 — Turn 1: Knowledge query (RAG)")
    payload = {"content": "How do I rotate a deploy key?"}
    info("User says", f"\"{payload['content']}\"")
    info("Waiting for LLM (~15s)...")

    t0 = time.monotonic()
    r = post(client, f"/v1/chat/{session_id}", json=payload)
    elapsed = time.monotonic() - t0

    if r.status_code != 200:
        fail(f"HTTP {r.status_code}", r.text)
        return ""

    data = r.json()
    ok(f"Response in {elapsed:.1f}s")
    ok("routed_to", data["routed_to"])
    ok("trace_id", data["trace_id"])
    print(f"\n  {YELLOW}Agent reply:{RESET}")
    print("  " + wrap_reply(data["reply"]))
    return data["trace_id"]


def test_fetch_trace(client: httpx.Client, trace_id: str) -> None:
    header("TEST 5 — GET /v1/traces/{trace_id}")
    r = get(client, f"/v1/traces/{trace_id}")

    if r.status_code != 200:
        fail(f"HTTP {r.status_code}", r.text)
        return

    data = r.json()
    ok("Trace fetched")
    ok("routed_to", data["routed_to"])
    ok("latency_ms", f"{data['latency_ms']} ms")
    ok("retrieved_chunk_ids", data["retrieved_chunk_ids"])
    ok("tool_calls count", len(data["tool_calls"]))
    for tc in data["tool_calls"]:
        info(f"  tool: {tc['tool_name']}", f"args={tc['args']}")


def test_account_turn(client: httpx.Client, session_id: str) -> None:
    header("TEST 6 — Turn 2: Account query (state persistence)")
    payload = {"content": "Show me my last 3 builds"}
    info("User says", f"\"{payload['content']}\"")
    info("Agent should know plan_tier=pro and user_id from session state (not re-asked)...")

    t0 = time.monotonic()
    r = post(client, f"/v1/chat/{session_id}", json=payload)
    elapsed = time.monotonic() - t0

    if r.status_code != 200:
        fail(f"HTTP {r.status_code}", r.text)
        return

    data = r.json()
    ok(f"Response in {elapsed:.1f}s")
    ok("routed_to", data["routed_to"])
    print(f"\n  {YELLOW}Agent reply:{RESET}")
    print("  " + wrap_reply(data["reply"]))


def test_guardrail(client: httpx.Client, session_id: str) -> None:
    header("TEST 7 — Guardrail: out-of-scope query")
    payload = {"content": "Write me a poem about the ocean"}
    info("User says", f"\"{payload['content']}\"")
    info("Agent should refuse and NOT call any tool...")

    r = post(client, f"/v1/chat/{session_id}", json=payload)
    data = r.json()

    if r.status_code == 200:
        ok("routed_to", data["routed_to"])
        print(f"\n  {YELLOW}Agent reply:{RESET}")
        print("  " + wrap_reply(data["reply"]))
        if "only assist" in data["reply"].lower() or "helix" in data["reply"].lower():
            ok("Guardrail fired correctly ✓")
        else:
            info("Guardrail note: agent replied but may not have refused explicitly")
    else:
        fail(f"HTTP {r.status_code}", r.text)


def test_trace_not_found(client: httpx.Client) -> None:
    header("TEST 8 — 404 on unknown trace")
    r = get(client, "/v1/traces/nonexistent-trace-xyz")
    data = r.json()
    info("Response", pretty_json(data))

    if r.status_code == 404 and data.get("title") == "TRACE_NOT_FOUND":
        ok("Correct 404 TRACE_NOT_FOUND")
    else:
        fail("Expected 404", r.text)


def test_idempotency(client: httpx.Client, session_id: str) -> None:
    header("TEST 9 — E1: Idempotency-Key deduplication")
    payload = {"content": "How do I set up secret scanning?"}
    key = "demo-idem-key-001"
    headers = {"Idempotency-Key": key}
    info("Sending same request TWICE with Idempotency-Key:", key)

    r1 = post(client, f"/v1/chat/{session_id}", json=payload, headers=headers)
    t1 = r1.json().get("trace_id", "")

    r2 = post(client, f"/v1/chat/{session_id}", json=payload, headers=headers)
    t2 = r2.json().get("trace_id", "")

    ok("Request 1 trace_id", t1)
    ok("Request 2 trace_id", t2)

    if t1 and t1 == t2:
        ok("IDEMPOTENCY PASS — same trace_id returned, pipeline ran once")
    else:
        fail("IDEMPOTENCY FAIL — different trace_ids!")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Helix SROP demo client")
    parser.add_argument("--base", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    args = parser.parse_args()

    print(f"\n{BOLD}Helix SROP — Demo Client{RESET}")
    print(f"  API base : {args.base}")
    print(f"  Timeout  : {args.timeout}s")

    with httpx.Client(base_url=args.base, timeout=args.timeout) as client:
        # ── Core tests ──────────────────────────────────────────────────────
        test_healthz(client)
        session_id = test_create_session(client)
        test_session_not_found(client)

        # ── Multi-turn conversation ──────────────────────────────────────────
        trace_id = test_knowledge_turn(client, session_id)
        test_fetch_trace(client, trace_id)
        test_account_turn(client, session_id)

        # ── Extensions ──────────────────────────────────────────────────────
        test_guardrail(client, session_id)
        test_trace_not_found(client)
        test_idempotency(client, session_id)

    print(f"\n{GREEN}{BOLD}{'=' * 60}")
    print("  All tests complete ✔")
    print(f"{'=' * 60}{RESET}\n")


if __name__ == "__main__":
    main()
