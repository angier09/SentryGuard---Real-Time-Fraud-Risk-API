from pathlib import Path

from starlette.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.risk_service import RiskService


def test_health_returns_status(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            model_path=tmp_path / "missing.joblib",
            threshold_path=tmp_path / "missing-threshold.json",
            metrics_path=tmp_path / "missing-metrics.json",
        )
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


def test_health_returns_healthy_when_model_is_loaded(
    risk_service: RiskService,
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(
            model_path=tmp_path / "missing.joblib",
            threshold_path=tmp_path / "missing-threshold.json",
            metrics_path=tmp_path / "missing-metrics.json",
        )
    )

    with TestClient(app) as client:
        app.state.risk_service = risk_service
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_model_info_returns_metadata(
    risk_service: RiskService,
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(
            model_path=tmp_path / "missing.joblib",
            threshold_path=tmp_path / "missing-threshold.json",
            metrics_path=tmp_path / "missing-metrics.json",
        )
    )

    with TestClient(app) as client:
        app.state.risk_service = risk_service
        response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["model_name"] == "xgboost"
    assert body["model_version"] == "xgb_v1.0.0"
    assert body["threshold"] == 0.72


def test_predict_returns_probability_and_decision(
    risk_service: RiskService,
    sample_transaction_payload: dict,
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(
            model_path=tmp_path / "missing.joblib",
            threshold_path=tmp_path / "missing-threshold.json",
            metrics_path=tmp_path / "missing-metrics.json",
        )
    )

    with TestClient(app) as client:
        app.state.risk_service = risk_service
        response = client.post("/v1/predict", json=sample_transaction_payload)

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["decision"] in {"approve", "monitor", "review", "block"}
    assert body["model_version"] == "xgb_v1.0.0"


def test_batch_predict_returns_predictions(
    risk_service: RiskService,
    sample_transaction_payload: dict,
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(
            model_path=tmp_path / "missing.joblib",
            threshold_path=tmp_path / "missing-threshold.json",
            metrics_path=tmp_path / "missing-metrics.json",
        )
    )

    with TestClient(app) as client:
        app.state.risk_service = risk_service
        response = client.post(
            "/v1/batch-predict",
            json={"transactions": [sample_transaction_payload]},
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 1
    assert body["model_version"] == "xgb_v1.0.0"
