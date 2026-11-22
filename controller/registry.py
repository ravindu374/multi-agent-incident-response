"""Block 3 - Agent registry: which agents exist, and which are allowed to take a task right now.

Cards (static) come from A2A discovery or from agents/cards/*.json. The registry adds live state:
  status      up / down (agent reported down, or its process is gone)
  heartbeat   an agent that has not checked in for HEARTBEAT_TTL_S is treated as gone
  load        in-flight tasks vs the card's max_concurrency
  breaker     after BREAKER_THRESHOLD consecutive failures the agent is skipped for BREAKER_COOLDOWN_S,
              then gets one trial task (half-open); a success closes the breaker again

eligible() is the *hard filter*: every rule here is a yes/no gate that a ranking score must never
override. Ranking among the eligible agents (cost, success history, ...) is Block 4's job.
A2A itself does not standardise a registry, so this one is ours.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .cards import AgentCard, RoutingProfile

HEARTBEAT_TTL_S = 30.0
BREAKER_THRESHOLD = 3
BREAKER_COOLDOWN_S = 60.0
LATENCY_EWMA_ALPHA = 0.3


class AgentState(BaseModel):
    agent_id: str
    status: Literal["up", "down"] = "up"
    last_heartbeat: float | None = None
    in_flight: int = 0
    consecutive_failures: int = 0
    breaker_open_until: float | None = None
    successes: int = 0
    failures: int = 0
    latency_ewma_s: float | None = None


class Candidate(BaseModel):
    agent_id: str
    card: AgentCard
    profile: RoutingProfile
    state: AgentState


class Exclusion(BaseModel):
    agent_id: str
    reason: str


class AgentRegistry:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._cards: dict[str, AgentCard] = {}
        self._profiles: dict[str, RoutingProfile] = {}
        self._states: dict[str, AgentState] = {}

    # -- membership -----------------------------------------------------------------------------
    def register(self, card: AgentCard, now: float | None = None) -> str:
        profile = card.routing_profile()  # validates the extension; refuses cards without it
        agent_id = card.name
        self._cards[agent_id], self._profiles[agent_id] = card, profile
        self._states[agent_id] = AgentState(agent_id=agent_id, last_heartbeat=self._now(now))
        return agent_id

    def unregister(self, agent_id: str) -> None:
        for table in (self._cards, self._profiles, self._states):
            table.pop(agent_id, None)

    @classmethod
    def from_directory(cls, directory: Path, clock: Callable[[], float] = time.monotonic) -> AgentRegistry:
        registry = cls(clock)
        for path in sorted(Path(directory).glob("*.json")):
            registry.register(AgentCard.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        return registry

    def agents(self) -> list[str]:
        return list(self._cards)

    def card(self, agent_id: str) -> AgentCard:
        return self._cards[agent_id]

    def profile(self, agent_id: str) -> RoutingProfile:
        return self._profiles[agent_id]

    def state(self, agent_id: str) -> AgentState:
        return self._states[agent_id]

    # -- live state ------------------------------------------------------------------------------
    def heartbeat(self, agent_id: str, now: float | None = None) -> None:
        state = self._states[agent_id]
        state.status, state.last_heartbeat = "up", self._now(now)

    def mark_down(self, agent_id: str) -> None:
        self._states[agent_id].status = "down"

    def begin_task(self, agent_id: str) -> None:
        self._states[agent_id].in_flight += 1

    def end_task(self, agent_id: str, ok: bool, latency_s: float | None = None, now: float | None = None) -> None:
        state = self._states[agent_id]
        state.in_flight = max(0, state.in_flight - 1)
        if latency_s is not None:
            state.latency_ewma_s = (latency_s if state.latency_ewma_s is None
                                    else LATENCY_EWMA_ALPHA * latency_s + (1 - LATENCY_EWMA_ALPHA) * state.latency_ewma_s)
        if ok:
            state.successes += 1
            state.consecutive_failures, state.breaker_open_until = 0, None
            return
        state.failures += 1
        state.consecutive_failures += 1
        if state.consecutive_failures >= BREAKER_THRESHOLD:
            state.breaker_open_until = self._now(now) + BREAKER_COOLDOWN_S

    # -- hard filter -----------------------------------------------------------------------------
    def check(self, agent_id: str, domain: str | None = None, intent: str | None = None,
              now: float | None = None) -> str | None:
        """Return why the agent may NOT take this task, or None if it may."""
        now = self._now(now)
        profile, state = self._profiles[agent_id], self._states[agent_id]
        if profile.role == "policy":
            return "policy agents validate plans; they do not investigate incidents"
        if domain is not None and domain not in profile.domains:
            return f"does not cover the {domain} domain"
        if intent is not None and intent not in profile.allowed_intents:
            return f"may not propose {intent}"
        if state.status == "down":
            return "reported down"
        if state.last_heartbeat is None or now - state.last_heartbeat > HEARTBEAT_TTL_S:
            age = "never" if state.last_heartbeat is None else f"{now - state.last_heartbeat:.0f}s ago"
            return f"no heartbeat (last: {age})"
        if state.breaker_open_until is not None and now < state.breaker_open_until:
            return (f"circuit breaker open after {state.consecutive_failures} consecutive failures "
                    f"({state.breaker_open_until - now:.0f}s left)")
        if state.in_flight >= profile.max_concurrency:
            return f"at capacity ({state.in_flight}/{profile.max_concurrency} tasks)"
        return None

    def eligible(self, domain: str, intent: str | None = None, include_generalist: bool = True,
                 now: float | None = None) -> tuple[list[Candidate], list[Exclusion]]:
        """Agents that may investigate a `domain` task now (specialists first, generalist last)."""
        candidates, excluded = [], []
        for agent_id in self._order():
            profile = self._profiles[agent_id]
            if profile.role == "generalist" and not include_generalist:
                continue
            reason = self.check(agent_id, domain, intent, now)
            if reason is None:
                candidates.append(Candidate(agent_id=agent_id, card=self._cards[agent_id], profile=profile,
                                            state=self._states[agent_id]))
            elif profile.role != "policy" and domain in profile.domains:
                excluded.append(Exclusion(agent_id=agent_id, reason=reason))
        return candidates, excluded

    def policy_agents(self, now: float | None = None) -> list[str]:
        return [a for a in self._order() if self._profiles[a].role == "policy"
                and self._states[a].status == "up"
                and self._states[a].last_heartbeat is not None
                and self._now(now) - self._states[a].last_heartbeat <= HEARTBEAT_TTL_S]

    # -- helpers -----------------------------------------------------------------------------------
    def _order(self) -> list[str]:
        rank = {"specialist": 0, "generalist": 1, "policy": 2}
        tier = {"fast": 0, "strong": 1}
        return sorted(self._cards, key=lambda a: (rank[self._profiles[a].role], tier[self._profiles[a].tier], a))

    def _now(self, now: float | None) -> float:
        return self._clock() if now is None else now
