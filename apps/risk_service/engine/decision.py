"""
Deterministic policy engine for risk_service.

Produces the final Decision from:
  - fraud_probability (ML output from M4)
  - rule_signals (evidence from rules.py)
  - config (all thresholds sourced from environment)

Architecture:
  Phase 1 — ML baseline decision from fraud_probability + threshold config.
  Phase 2 — Rule signals can escalate the baseline (HIGH → BLOCK, MEDIUM → varies).
             Rules can NEVER de-escalate a decision.
             LOW severity rules are informational only — no escalation.
  Phase 3 — Return max(baseline, rule escalations).

relationship_risk_score and other graph-derived features are NOT used here as
arithmetic ensemble inputs — Member 4's model already consumed them. This avoids
double-counting graph information.
"""
from __future__ import annotations

from apps.risk_service.config import Settings
from apps.risk_service.schemas.risk import Decision, RuleSignal, RuleSeverity

# Escalation order — index increases with severity
_ORDER = [Decision.ALLOW, Decision.STEP_UP, Decision.REVIEW, Decision.BLOCK]


def _escalate(current: Decision, target: Decision) -> Decision:
    """Return target if it is higher in escalation order; otherwise keep current."""
    if _ORDER.index(target) > _ORDER.index(current):
        return target
    return current


def decide(
    fraud_probability: float,
    rule_signals: list[RuleSignal],
    config: Settings,
) -> Decision:
    """
    Deterministic policy function.

    All thresholds are read from config — never hardcoded.
    The current dev defaults (0.30 / 0.50 / 0.70) are NOT production-validated;
    final values will be set after M4 threshold sweep + M4/M5 joint calibration.
    """
    # ── Phase 1: ML baseline ──────────────────────────────────────────────────
    if fraud_probability < config.threshold_allow_max:
        baseline = Decision.ALLOW
    elif fraud_probability < config.threshold_step_up_max:
        baseline = Decision.STEP_UP
    elif fraud_probability < config.threshold_review_max:
        baseline = Decision.REVIEW
    else:
        baseline = Decision.BLOCK

    # ── Phase 2: Rule escalations (can only move decision up) ─────────────────
    current = baseline
    for signal in rule_signals:
        if not signal.triggered:
            continue
        if signal.severity == RuleSeverity.HIGH:
            current = _escalate(current, Decision.BLOCK)
        elif signal.severity == RuleSeverity.MEDIUM:
            if signal.rule_id == "MULTI_DEVICE":
                current = _escalate(current, Decision.REVIEW)
            else:
                # HIGH_AMOUNT_NEW_DEVICE, NEW_MERCHANT_DEVIATION → at least STEP_UP
                current = _escalate(current, Decision.STEP_UP)
        # LOW severity: informational only, no escalation

    # ── Phase 3: Return final decision ────────────────────────────────────────
    return current
