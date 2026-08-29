"""Unit tests for deterministic rule functions (no I/O)."""
from __future__ import annotations

import pytest

from apps.risk_service.engine.rules import (
    evaluate_all,
    evaluate_high_amount_new_device,
    evaluate_location_shift,
    evaluate_multi_device,
    evaluate_new_merchant_deviation,
    evaluate_velocity_burst,
)
from apps.risk_service.schemas.risk import Decision, RuleSignal, RuleSeverity
from apps.risk_service.tests.conftest import fast_settings


@pytest.fixture
def cfg():
    return fast_settings()


# ── VELOCITY_BURST ────────────────────────────────────────────────────────────

def test_velocity_burst_triggered(cfg):
    sig = evaluate_velocity_burst({"customer_txn_count_60m": 11}, cfg)
    assert sig.triggered is True
    assert sig.severity == RuleSeverity.HIGH
    assert sig.rule_id == "VELOCITY_BURST"


def test_velocity_burst_not_triggered(cfg):
    sig = evaluate_velocity_burst({"customer_txn_count_60m": 9}, cfg)
    assert sig.triggered is False


def test_velocity_burst_at_exact_threshold(cfg):
    """Strictly greater than threshold — equal does NOT trigger."""
    sig = evaluate_velocity_burst({"customer_txn_count_60m": 10}, cfg)
    assert sig.triggered is False


# ── HIGH_AMOUNT_NEW_DEVICE ────────────────────────────────────────────────────

def test_high_amount_new_device_triggered(cfg):
    sig = evaluate_high_amount_new_device(
        {"is_new_device": True}, amount=2500.0, config=cfg
    )
    assert sig.triggered is True
    assert sig.severity == RuleSeverity.MEDIUM


def test_high_amount_known_device_not_triggered(cfg):
    sig = evaluate_high_amount_new_device(
        {"is_new_device": False}, amount=9999.0, config=cfg
    )
    assert sig.triggered is False


def test_high_amount_new_device_but_low_amount_not_triggered(cfg):
    sig = evaluate_high_amount_new_device(
        {"is_new_device": True}, amount=100.0, config=cfg
    )
    assert sig.triggered is False


# ── NEW_MERCHANT_DEVIATION ────────────────────────────────────────────────────

def test_new_merchant_deviation_triggered(cfg):
    sig = evaluate_new_merchant_deviation(
        {"is_new_merchant": True, "amount_deviation_ratio": 3.5}, cfg
    )
    assert sig.triggered is True
    assert sig.severity == RuleSeverity.MEDIUM


def test_known_merchant_no_trigger(cfg):
    sig = evaluate_new_merchant_deviation(
        {"is_new_merchant": False, "amount_deviation_ratio": 5.0}, cfg
    )
    assert sig.triggered is False


def test_new_merchant_low_deviation_no_trigger(cfg):
    sig = evaluate_new_merchant_deviation(
        {"is_new_merchant": True, "amount_deviation_ratio": 1.0}, cfg
    )
    assert sig.triggered is False


# ── MULTI_DEVICE ──────────────────────────────────────────────────────────────

def test_multi_device_triggered(cfg):
    sig = evaluate_multi_device({"customer_device_degree": 6}, cfg)
    assert sig.triggered is True
    assert sig.severity == RuleSeverity.MEDIUM


def test_multi_device_at_threshold_not_triggered(cfg):
    sig = evaluate_multi_device({"customer_device_degree": 5}, cfg)
    assert sig.triggered is False


def test_multi_device_below_threshold(cfg):
    sig = evaluate_multi_device({"customer_device_degree": 2}, cfg)
    assert sig.triggered is False


# ── LOCATION_SHIFT ────────────────────────────────────────────────────────────

def test_location_shift_informational_only(cfg):
    """LOW severity — triggered flag True, but severity is LOW (no escalation)."""
    sig = evaluate_location_shift({"location_shift": True})
    assert sig.triggered is True
    assert sig.severity == RuleSeverity.LOW
    assert sig.rule_id == "LOCATION_SHIFT"


def test_location_shift_not_triggered(cfg):
    sig = evaluate_location_shift({"location_shift": False})
    assert sig.triggered is False


# ── evaluate_all ──────────────────────────────────────────────────────────────

def test_evaluate_all_returns_five_signals(cfg):
    signals = evaluate_all(
        {"customer_txn_count_60m": 2, "is_new_device": False, "is_new_merchant": False,
         "location_shift": False, "customer_device_degree": 1, "amount_deviation_ratio": 0.1},
        amount=100.0,
        config=cfg,
    )
    assert len(signals) == 5


def test_evaluate_all_returns_rule_signals_not_decisions(cfg):
    """Each element must be a RuleSignal, never a Decision."""
    signals = evaluate_all({}, amount=0.0, config=cfg)
    for sig in signals:
        assert isinstance(sig, RuleSignal)
        assert not isinstance(sig, Decision)


def test_all_rules_clear_on_low_risk_features(cfg):
    """Typical low-risk feature set — all rules should be not triggered except maybe location."""
    from apps.risk_service.tests.conftest import FEATURES_LOW_RISK
    signals = evaluate_all(FEATURES_LOW_RISK, amount=50.0, config=cfg)
    high_triggered = [s for s in signals if s.triggered and s.severity == RuleSeverity.HIGH]
    assert len(high_triggered) == 0
