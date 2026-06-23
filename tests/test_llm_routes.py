import pytest
from fastapi.testclient import TestClient

from src.llm.client import LLMClientError
from src.llm.dependencies import get_llm_client
from src.main import app


class MockLLMClient:
    def chat(self, user_message: str, system_prompt: str | None = None, **kwargs):
        if "You extract structured Adult Income features" in (system_prompt or ""):
            return '{"age": 39, "workclass": "State-gov", "education": "Bachelors", "education-num": 13, "marital-status": "Never-married", "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "White", "sex": "Male", "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States"}'
        return "Mocked explanation response."

    def parse_json(self, text: str):
        return {
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


class ClarifyingLLMClient:
    def chat(self, user_message: str, system_prompt: str | None = None, **kwargs):
        return '{"needs_clarification": true, "explanation": "Please provide occupation and hours worked per week."}'

    def parse_json(self, text: str):
        return {
            "needs_clarification": True,
            "explanation": "Please provide occupation and hours worked per week.",
        }


class FailingLLMClient:
    def chat(self, user_message: str, system_prompt: str | None = None, **kwargs):
        raise LLMClientError("Upstream LLM failure")

    def parse_json(self, text: str):
        raise LLMClientError("Unable to parse LLM output")


def override_llm_client():
    return MockLLMClient()


def override_clarifying_llm_client():
    return ClarifyingLLMClient()


def override_failing_llm_client():
    return FailingLLMClient()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_chat_returns_structured_response(client):
    app.dependency_overrides[get_llm_client] = override_llm_client

    response = client.post(
        "/chat",
        json={"message": "I am 39, work for the state government, have a bachelor's degree, and work as an admin clerk in the US."},
    )

    assert response.status_code == 200
    data = response.json()

    assert "parsed_features" in data
    assert "prediction" in data
    assert "probability_gt_50k" in data
    assert "model_name" in data
    assert "explanation" in data
    assert "needs_clarification" in data

    assert data["parsed_features"]["age"] == 39
    assert data["parsed_features"]["education-num"] == 13
    assert data["parsed_features"]["native-country"] == "United-States"
    assert data["needs_clarification"] is False
    assert isinstance(data["explanation"], str)
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
    assert "explanation" in data
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
    assert "detail" in response.json()


def test_explain_prediction_returns_prediction_and_explanation(client):
    app.dependency_overrides[get_llm_client] = override_llm_client

    response = client.post(
        "/explain-prediction",
        json={
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
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "probability_gt_50k" in data
    assert "model_name" in data
    assert "explanation" in data
    assert data["explanation"] == "Mocked explanation response."
