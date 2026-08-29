"""Unit tests for the deterministic policy engine (no I/O)."""
from __future__ import annotations

import pytest

from apps.risk_service.engine.decision import decide
from apps.risk_service.schemas.risk import Decision, RuleSeverity, RuleSignal
from apps.risk_service.tests.conftest import fast_settings


@pytest.fixture
def cfg():
    return fast_settings()


def _signal(rule_id: str, triggered: bool, severity: RuleSeverity) -> RuleSignal:
    return RuleSignal(
        rule_id=rule_id,
        triggered=triggered,
        severity=severity,
        detail="test",
    )


HIGH = lambda rule_id="VELOCITY_BURST": _signal(rule_id, True, RuleSeverity.HIGH)
MED_STEP = lambda rule_id="HIGH_AMOUNT_NEW_DEVICE": _signal(rule_id, True, RuleSeverity.MEDIUM)
MED_REVIEW = lambda: _signal("MULTI_DEVICE", True, RuleSeverity.MEDIUM)
LOW = lambda: _signal("LOCATION_SHIFT", True, RuleSeverity.LOW)
NOT_TRIGGERED = lambda: _signal("VELOCITY_BURST", False, RuleSeverity.HIGH)


# ── ML baseline (Phase 1) ─────────────────────────────────────────────────────

def test_allow_on_prob_below_0_30(cfg):
    assert decide(0.10, [], cfg) == Decision.ALLOW


def test_step_up_on_prob_0_30_to_0_50(cfg):
    assert decide(0.40, [], cfg) == Decision.STEP_UP


def test_review_on_prob_0_50_to_0_70(cfg):
    assert decide(0.60, [], cfg) == Decision.REVIEW


def test_block_on_prob_0_70_plus(cfg):
    assert decide(0.80, [], cfg) == Decision.BLOCK


def test_block_boundary_exact_0_70(cfg):
    """fraud_prob == 0.70 should be BLOCK (≥ threshold_review_max)."""
    assert decide(0.70, [], cfg) == Decision.BLOCK


def test_allow_boundary_exclusive(cfg):
    """fraud_prob == 0.30 should be STEP_UP (not ALLOW — < threshold_allow_max exclusive)."""
    assert decide(0.30, [], cfg) == Decision.STEP_UP


# ── Rule escalations (Phase 2) ────────────────────────────────────────────────

def test_high_severity_rule_escalates_allow_to_block(cfg):
    assert decide(0.10, [HIGH()], cfg) == Decision.BLOCK


def test_high_severity_rule_escalates_step_up_to_block(cfg):
    assert decide(0.40, [HIGH()], cfg) == Decision.BLOCK


def test_high_severity_rule_escalates_review_to_block(cfg):
    assert decide(0.60, [HIGH()], cfg) == Decision.BLOCK


def test_medium_step_up_rule_escalates_allow_to_step_up(cfg):
    assert decide(0.10, [MED_STEP()], cfg) == Decision.STEP_UP


def test_medium_step_up_rule_does_not_exceed_review(cfg):
    """STEP_UP rule on REVIEW baseline → stays REVIEW (can't de-escalate)."""
    assert decide(0.60, [MED_STEP()], cfg) == Decision.REVIEW


def test_medium_review_rule_escalates_allow_to_review(cfg):
    assert decide(0.10, [MED_REVIEW()], cfg) == Decision.REVIEW


def test_medium_review_rule_escalates_step_up_to_review(cfg):
    assert decide(0.40, [MED_REVIEW()], cfg) == Decision.REVIEW


def test_rules_cannot_de_escalate_block(cfg):
    """Any rule on BLOCK baseline must stay BLOCK."""
    assert decide(0.80, [MED_STEP(), LOW(), NOT_TRIGGERED()], cfg) == Decision.BLOCK


def test_location_shift_low_does_not_escalate(cfg):
    """LOW severity rule — informational only, no escalation."""
    assert decide(0.10, [LOW()], cfg) == Decision.ALLOW


def test_untriggered_rule_has_no_effect(cfg):
    assert decide(0.10, [NOT_TRIGGERED()], cfg) == Decision.ALLOW


# ── Custom thresholds from config ─────────────────────────────────────────────

def test_custom_threshold_allow_max(cfg):
    """With THRESHOLD_ALLOW_MAX=0.50, fraud_prob=0.45 should be ALLOW."""
    custom = fast_settings(threshold_allow_max=0.50, threshold_step_up_max=0.65, threshold_review_max=0.80)
    assert decide(0.45, [], custom) == Decision.ALLOW


def test_custom_threshold_block(cfg):
    """With THRESHOLD_REVIEW_MAX=0.60, fraud_prob=0.65 should be BLOCK."""
    custom = fast_settings(threshold_allow_max=0.20, threshold_step_up_max=0.40, threshold_review_max=0.60)
    assert decide(0.65, [], custom) == Decision.BLOCK


# ── Policy version ────────────────────────────────────────────────────────────

def test_multiple_rules_highest_escalation_wins(cfg):
    """HIGH + MED rules: highest escalation (BLOCK) wins."""
    assert decide(0.10, [MED_STEP(), HIGH()], cfg) == Decision.BLOCK


def test_multiple_medium_rules_review_wins_over_step_up(cfg):
    """MED_STEP + MED_REVIEW: REVIEW wins."""
    assert decide(0.10, [MED_STEP(), MED_REVIEW()], cfg) == Decision.REVIEW
