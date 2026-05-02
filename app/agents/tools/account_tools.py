"""
Account tools — used by AccountAgent.
Mock data is used for the take-home; the integration (wiring as ADK tools) is
what's evaluated.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta


@dataclass
class BuildSummary:
    build_id: str
    pipeline: str
    status: str  # passed | failed | cancelled
    branch: str
    started_at: str  # ISO string (easier for LLM to read)
    duration_seconds: int


@dataclass
class AccountStatus:
    user_id: str
    plan_tier: str
    concurrent_builds_used: int
    concurrent_builds_limit: int
    storage_used_gb: float
    storage_limit_gb: float


# ── Mock data helpers ──────────────────────────────────────────────────────────

_PIPELINES = ["build-and-test", "deploy-staging", "release", "lint", "e2e-tests"]
_BRANCHES = ["main", "dev", "feature/auth-v2", "fix/cache-miss", "release/1.4"]
_STATUSES = ["passed", "failed", "cancelled"]
_STATUS_WEIGHTS = [0.6, 0.3, 0.1]

_PLAN_LIMITS: dict[str, dict] = {
    "free": {"concurrent": 1, "storage_gb": 5.0},
    "pro": {"concurrent": 4, "storage_gb": 50.0},
    "enterprise": {"concurrent": 20, "storage_gb": 500.0},
}


def _seed(user_id: str) -> int:
    """Deterministic seed from user_id so the same user always gets the same mock data."""
    return int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)  # noqa: S324


import hashlib  # noqa: E402 (placed after dataclasses for readability)


async def get_recent_builds(user_id: str, limit: int = 5) -> str:
    """
    Return the most recent builds for a user, newest first.

    Args:
        user_id: the user's ID
        limit: number of builds to return (default 5, max 10)

    Returns:
        Formatted string describing recent builds.
    """
    rng = random.Random(_seed(user_id))
    limit = min(limit, 10)
    builds: list[BuildSummary] = []
    for i in range(limit):
        offset_hours = i * rng.randint(1, 12)
        started = datetime.utcnow() - timedelta(hours=offset_hours)
        status = rng.choices(_STATUSES, weights=_STATUS_WEIGHTS, k=1)[0]
        builds.append(
            BuildSummary(
                build_id=f"bld_{user_id[-4:]}_{i:04d}",
                pipeline=rng.choice(_PIPELINES),
                status=status,
                branch=rng.choice(_BRANCHES),
                started_at=started.strftime("%Y-%m-%d %H:%M UTC"),
                duration_seconds=rng.randint(30, 600),
            )
        )

    lines = [f"Recent {limit} builds for user '{user_id}':"]
    for b in builds:
        lines.append(
            f"  [{b.build_id}] {b.pipeline} | {b.status.upper()} | "
            f"branch: {b.branch} | started: {b.started_at} | duration: {b.duration_seconds}s"
        )
    return "\n".join(lines)


async def get_account_status(user_id: str) -> str:
    """
    Return current account plan and usage limits for a user.

    Args:
        user_id: the user's ID

    Returns:
        Formatted string with plan tier and resource usage.
    """
    rng = random.Random(_seed(user_id))
    # Derive plan tier from hash (for demo, most users are 'pro')
    tier = rng.choices(["free", "pro", "enterprise"], weights=[0.2, 0.6, 0.2], k=1)[0]
    limits = _PLAN_LIMITS[tier]

    concurrent_used = rng.randint(0, limits["concurrent"])
    storage_used = round(rng.uniform(0.1, limits["storage_gb"] * 0.8), 2)

    status = AccountStatus(
        user_id=user_id,
        plan_tier=tier,
        concurrent_builds_used=concurrent_used,
        concurrent_builds_limit=limits["concurrent"],
        storage_used_gb=storage_used,
        storage_limit_gb=limits["storage_gb"],
    )

    return (
        f"Account status for '{user_id}':\n"
        f"  Plan tier: {status.plan_tier}\n"
        f"  Concurrent builds: {status.concurrent_builds_used}/{status.concurrent_builds_limit}\n"
        f"  Storage used: {status.storage_used_gb} GB / {status.storage_limit_gb} GB"
    )
