"""
Deterministic rule engine for risk_service.

Each rule function returns a RuleSignal (evidence). Rules NEVER return a Decision.
The Decision is produced exclusively by engine/decision.py.

Graph-derived features (relationship_risk_score, shared_device_customer_count, etc.)
are NOT used as arithmetic ensemble weights here — Member 4's model already consumed
them. Rules use graph structural features (e.g. customer_device_degree) only as
threshold-crossing binary signals where explicitly justified.
"""
from __future__ import annotations

from apps.risk_service.config import Settings
from apps.risk_service.schemas.risk import RuleSignal, RuleSeverity


def evaluate_velocity_burst(features: dict, config: Settings) -> RuleSignal:
    """HIGH severity: excessive transaction velocity for this customer in 60 min."""
    count = int(features.get("customer_txn_count_60m", 0))
    triggered = count > config.rule_velocity_threshold
    return RuleSignal(
        rule_id="VELOCITY_BURST",
        triggered=triggered,
        severity=RuleSeverity.HIGH,
        detail=(
            f"{count} txns in 60 min exceeds threshold {config.rule_velocity_threshold}"
            if triggered
            else f"{count} txns in 60 min within threshold {config.rule_velocity_threshold}"
        ),
    )


def evaluate_high_amount_new_device(
    features: dict, amount: float, config: Settings
) -> RuleSignal:
    """MEDIUM severity: new device used for a high-value transaction."""
    is_new = bool(features.get("is_new_device", False))
    triggered = is_new and amount > config.rule_high_amount_threshold
    return RuleSignal(
        rule_id="HIGH_AMOUNT_NEW_DEVICE",
        triggered=triggered,
        severity=RuleSeverity.MEDIUM,
        detail=(
            f"New device + amount {amount:.2f} exceeds threshold {config.rule_high_amount_threshold:.2f}"
            if triggered
            else "No high-amount new-device signal"
        ),
    )


def evaluate_new_merchant_deviation(features: dict, config: Settings) -> RuleSignal:
    """MEDIUM severity: new merchant combined with large amount deviation."""
    is_new = bool(features.get("is_new_merchant", False))
    deviation = float(features.get("amount_deviation_ratio", 0.0))
    triggered = is_new and deviation > config.rule_deviation_threshold
    return RuleSignal(
        rule_id="NEW_MERCHANT_DEVIATION",
        triggered=triggered,
        severity=RuleSeverity.MEDIUM,
        detail=(
            f"New merchant + deviation {deviation:.3f} exceeds threshold {config.rule_deviation_threshold:.3f}"
            if triggered
            else "No new-merchant deviation signal"
        ),
    )


def evaluate_multi_device(features: dict, config: Settings) -> RuleSignal:
    """
    MEDIUM severity: customer linked to an unusual number of devices.
    Uses customer_device_degree as a count (graph structural feature),
    not as a re-weighted ensemble term. M4 already scored this feature.
    """
    degree = int(features.get("customer_device_degree", 0))
    triggered = degree > config.rule_multi_device_threshold
    return RuleSignal(
        rule_id="MULTI_DEVICE",
        triggered=triggered,
        severity=RuleSeverity.MEDIUM,
        detail=(
            f"customer_device_degree {degree} exceeds threshold {config.rule_multi_device_threshold}"
            if triggered
            else f"customer_device_degree {degree} within threshold {config.rule_multi_device_threshold}"
        ),
    )


def evaluate_location_shift(features: dict) -> RuleSignal:
    """LOW severity: location differs from most recent known customer location. Informational only."""
    triggered = bool(features.get("location_shift", False))
    return RuleSignal(
        rule_id="LOCATION_SHIFT",
        triggered=triggered,
        severity=RuleSeverity.LOW,
        detail=(
            "Location shift detected (informational only — no automatic escalation)"
            if triggered
            else "No location shift detected"
        ),
    )


def evaluate_all(features: dict, amount: float, config: Settings) -> list[RuleSignal]:
    """
    Run all rules and return a list of evidence signals.
    Returns signals for ALL rules (triggered=False for those that did not fire).
    Never returns a Decision — that is decision.py's responsibility.
    """
    return [
        evaluate_velocity_burst(features, config),
        evaluate_high_amount_new_device(features, amount, config),
        evaluate_new_merchant_deviation(features, config),
        evaluate_multi_device(features, config),
        evaluate_location_shift(features),
    ]
