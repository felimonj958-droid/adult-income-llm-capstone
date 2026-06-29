import pytest
from fastapi.testclient import TestClient

from src.llm.client import LLMClientError
from src.llm.dependencies import get_llm_client
from src.main import app


VALID_FEATURES = {
    "age": 39,
    "workclass": "State-gov",
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


class MockLLMClient:
    def chat(self, user_message: str, system_prompt: str | None = None, **kwargs):
        if system_prompt and "You convert a user's natural-language description into Adult Income features." in system_prompt:
            return (
                '{"age": 39, "workclass": "State-gov", "education": "Bachelors", '
                '"education-num": 13, "marital-status": "Never-married", '
                '"occupation": "Adm-clerical", "relationship": "Not-in-family", '
                '"race": "White", "sex": "Male", "capital-gain": 2174, '
                '"capital-loss": 0, "hours-per-week": 40, '
                '"native-country": "United-States"}'
            )
        return "Mocked explanation response."

    def parse_json(self, text: str):
        return VALID_FEATURES.copy()


class ClarifyingLLMClient:
    def chat(self, user_message: str, system_prompt: str | None = None, **kwargs):
        return (
            '{"needs_clarification": true, '
            '"explanation": "Please provide occupation and hours worked per week."}'
        )

    def parse_json(self, text: str):
        return {
            "needs_clarification": True,
            "explanation": "Please provide occupation and hours worked per week.",
        }


class InvalidJSONLLMClient:
    def chat(self, user_message: str, system_prompt: str | None = None, **kwargs):
        return "not valid json"

    def parse_json(self, text: str):
        raise LLMClientError("LLM returned invalid JSON")


class FailingLLMClient:
    def chat(self, user_message: str, system_prompt: str | None = None, **kwargs):
        raise LLMClientError("Upstream LLM failure")

    def parse_json(self, text: str):
        raise LLMClientError("Unable to parse LLM output")


@pytest.fixture
def client():
    app.dependency_overrides = {}
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides = {}


def override_llm_client():
    return MockLLMClient()


def override_clarifying_llm_client():
    return ClarifyingLLMClient()


def override_invalid_json_llm_client():
    return InvalidJSONLLMClient()


def override_failing_llm_client():
    return FailingLLMClient()


def test_chat_returns_structured_response(client):
    app.dependency_overrides[get_llm_client] = override_llm_client

    response = client.post(
        "/chat",
        json={
            "message": (
                "I am 39, work for the state government, have a bachelor's degree, "
                "work as an admin clerk, and live in the United States."
            )
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["needs_clarification"] is False
    assert data["parsed_features"]["age"] == 39
    assert data["parsed_features"]["education-num"] == 13
    assert data["parsed_features"]["native-country"] == "United-States"
    assert data["prediction"] in ["<=50K", ">50K"]
    assert 0.0 <= data["probability_gt_50k"] <= 1.0
    assert isinstance(data["model_name"], str)
    assert data["model_name"]
    assert data["explanation"] == "Mocked explanation response."


def test_chat_returns_clarification_when_features_missing(client):
    app.dependency_overrides[get_llm_client] = override_clarifying_llm_client

    response = client.post(
        "/chat",
        json={"message": "I am 39 and have a bachelor's degree."},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["needs_clarification"] is True
    assert data["explanation"] == "Please provide occupation and hours worked per week."
    assert data["parsed_features"] is None
    assert data["prediction"] is None
    assert data["probability_gt_50k"] is None
    assert data["model_name"] is None


def test_chat_returns_502_on_llm_failure(client):
    app.dependency_overrides[get_llm_client] = override_failing_llm_client

    response = client.post(
        "/chat",
        json={"message": "Summarize this project."},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Upstream LLM failure"


def test_chat_returns_502_on_invalid_llm_json(client):
    app.dependency_overrides[get_llm_client] = override_invalid_json_llm_client

    response = client.post(
        "/chat",
        json={"message": "I am 39, work in government, and have a bachelor's degree."},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "LLM returned invalid JSON"


def test_chat_rejects_blank_message(client):
    response = client.post("/chat", json={"message": "   "})

    assert response.status_code == 422


def test_explain_prediction_returns_prediction_and_explanation(client):
    app.dependency_overrides[get_llm_client] = override_llm_client

    response = client.post(
        "/explain-prediction",
        json=VALID_FEATURES,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in ["<=50K", ">50K"]
    assert 0.0 <= data["probability_gt_50k"] <= 1.0
    assert isinstance(data["model_name"], str)
    assert data["model_name"]
    assert data["explanation"] == "Mocked explanation response."


def test_explain_prediction_returns_502_on_llm_failure(client):
    app.dependency_overrides[get_llm_client] = override_failing_llm_client

    response = client.post(
        "/explain-prediction",
        json=VALID_FEATURES,
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Upstream LLM failure"
