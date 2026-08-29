import asyncio
from apps.risk_service.engine.rules import evaluate_all
from apps.risk_service.config import Settings
from apps.risk_service.tests.conftest import FEATURES_LOW_RISK, SAMPLE_TRANSACTION

print(evaluate_all(FEATURES_LOW_RISK, SAMPLE_TRANSACTION["amount"], Settings()))
from apps.risk_service.engine.decision import decide_fallback
print("Fallback:", decide_fallback(evaluate_all(FEATURES_LOW_RISK, SAMPLE_TRANSACTION["amount"], Settings())))
