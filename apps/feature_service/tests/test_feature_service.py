from datetime import datetime, timezone

from fastapi.testclient import TestClient

from pathlib import Path

import pandas as pd
import pytest

from apps.feature_service.batch import write_feature_dataset
from apps.feature_service.features.extractor import behavioral_features, extract_batch, extract_feature_vector
from apps.feature_service.features.schema import FEATURE_NAMES
from apps.feature_service.graph.scorer import RelationshipGraph
from apps.feature_service.main import app, store
from shared.schemas.transaction import ChannelEnum, Location, Transaction


def tx(
    transaction_id: str,
    customer_id: str = "c1",
    merchant_id: str = "m1",
    amount: float = 100.0,
    minute: int = 0,
    device_id: str | None = "d1",
    city: str = "New York",
) -> Transaction:
    return Transaction(
        transaction_id=transaction_id,
        customer_id=customer_id,
        merchant_id=merchant_id,
        amount=amount,
        currency="USD",
        timestamp=datetime(2026, 8, 22, 10, minute, tzinfo=timezone.utc),
        channel=ChannelEnum.CARD,
        device_id=device_id,
        location=Location(country="US", city=city),
        merchant_category="grocery",
    )


def envelope(data: dict[str, object]) -> dict[str, object]:
    return {
        "request_id": "REQ_test",
        "timestamp": "2026-08-22T10:00:00Z",
        "data": data,
    }


def test_behavioral_features_known_values() -> None:
    first = tx("tx1", amount=100.0, minute=0)
    second = tx("tx2", amount=150.0, minute=30)
    third = tx("tx3", merchant_id="m2", amount=300.0, minute=45, device_id="d2", city="Boston")

    features = behavioral_features(third, [first, second])

    assert features["customer_txn_count_60m"] == 2
    assert features["customer_amount_mean_prior"] == 125.0
    assert features["amount_deviation_ratio"] == 1.4
    assert features["is_new_device"] is True
    assert features["is_new_merchant"] is True
    assert features["location_shift"] is True


def test_graph_shared_device_counts() -> None:
    first = tx("tx1", customer_id="c1", device_id="shared")
    second = tx("tx2", customer_id="c2", device_id="shared", minute=5)
    candidate = tx("tx3", customer_id="c3", merchant_id="m2", device_id="shared", minute=10)

    score = RelationshipGraph.from_transactions([first, second]).score(candidate)

    assert score["device_customer_degree"] == 3
    assert score["shared_device_customer_count"] == 2
    assert score["relationship_risk_score"] == 0.7


def test_schema_matches_extraction_endpoint_fields() -> None:
    client = TestClient(app)
    store.clear()

    schema_response = client.get("/api/v1/features/schema").json()
    extract_response = client.post(
        "/api/v1/features/extract",
        json=envelope(tx("tx1").model_dump(mode="json")),
    ).json()

    schema_names = [field["name"] for field in schema_response["data"]["features"]]
    extracted_names = list(extract_response["data"]["features"].keys())

    assert schema_names == list(FEATURE_NAMES)
    assert extracted_names == schema_names


def test_offline_online_parity() -> None:
    transactions = [
        tx("tx1", amount=100.0, minute=0),
        tx("tx2", amount=150.0, minute=30),
        tx("tx3", merchant_id="m2", amount=300.0, minute=45, device_id="d2", city="Boston"),
    ]

    offline_rows = extract_batch(transactions)
    history: list[Transaction] = []
    online_vectors = []
    for transaction in transactions:
        online_vectors.append(extract_feature_vector(transaction, history))
        history.append(transaction)

    assert [row["transaction_id"] for row in offline_rows] == ["tx1", "tx2", "tx3"]
    assert [{name: row[name] for name in FEATURE_NAMES} for row in offline_rows] == online_vectors


def test_batch_output_preserves_member2_labels_for_member4(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data" / "fixtures" / "feature_known"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            tx("tx1").model_dump(mode="json"),
            tx("tx2", amount=150.0, minute=30).model_dump(mode="json"),
            tx("tx3", merchant_id="m2", amount=300.0, minute=45, device_id="d2").model_dump(mode="json"),
            tx("tx4", customer_id="c2", merchant_id="m3", amount=80.0, minute=50, device_id="d2").model_dump(mode="json"),
        ]
    ).to_parquet(dataset_dir / "transactions.parquet", index=False)
    (dataset_dir / "manifest.json").write_text(
        '{"dataset_id":"feature_known","rows":4,"schema_version":"1.0","provenance":"test_fixture"}',
        encoding="utf-8",
    )

    result = write_feature_dataset("feature_known", source_roots=[str(tmp_path)])
    output_path = Path(result["output_path"])
    feature_frame = pd.read_parquet(output_path)

    assert result["rows"] == 4
    assert {"is_fraud", "attack_id", "source_schema_version", "dataset_provenance"}.issubset(
        feature_frame.columns
    )
    assert list(feature_frame["is_fraud"]) == [False, False, False, False]
    assert all(name in feature_frame.columns for name in FEATURE_NAMES)


def test_can_process_member2_generated_fixture() -> None:
    member2_root = Path(__file__).resolve().parents[3]
    if not member2_root.exists():
        pytest.skip("Member 2 FraudGuard fixture tree is not present.")

    result = write_feature_dataset("DS_91c85fbe", source_roots=[str(member2_root)])
    output_path = Path(result["output_path"])
    feature_frame = pd.read_parquet(output_path)

    assert result["rows"] == 1000
    assert result["fraud_rows"] == 100
    assert result["attack_ids"] == ["ATO_001", "MULE_001"]
    assert feature_frame["is_fraud"].sum() == 100
    assert set(feature_frame.loc[feature_frame["is_fraud"], "attack_id"]) == {"ATO_001", "MULE_001"}
    assert all(name in feature_frame.columns for name in FEATURE_NAMES)


def test_malformed_transaction_returns_envelope_validation_error() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/features/extract",
        json=envelope({"transaction_id": "bad"}),
    )

    body = response.json()
    assert response.status_code == 422
    assert body["status"] == "error"
    assert body["error"]["code"] == "VALIDATION_ERROR"

