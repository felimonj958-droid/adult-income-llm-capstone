from fastapi.testclient import TestClient

from src.main import app
from src.llm.client import LLMClientError
from src.llm.dependencies import get_llm_client


class MockLLMClient:
    def chat(self, user_message: str, system_prompt: str | None = None, **kwargs):
        return "Mocked LLM response."


class FailingLLMClient:
    def chat(self, user_message: str, system_prompt: str | None = None, **kwargs):
        raise LLMClientError("Upstream LLM failure")


def override_llm_client():
    return MockLLMClient()


def override_failing_llm_client():
    return FailingLLMClient()


client = TestClient(app)


def test_chat_returns_message():
    app.dependency_overrides[get_llm_client] = override_llm_client

    response = client.post(
        "/chat",
        json={"message": "Summarize this project."},
    )

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Mocked LLM response."

    app.dependency_overrides.clear()


def test_chat_returns_502_on_llm_failure():
    app.dependency_overrides[get_llm_client] = override_failing_llm_client

    response = client.post(
        "/chat",
        json={"message": "Summarize this project."},
    )

    assert response.status_code == 502
    assert "detail" in response.json()

    app.dependency_overrides.clear()


def test_explain_prediction_returns_prediction_and_explanation():
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
    assert data["explanation"] == "Mocked LLM response."

    app.dependency_overrides.clear()
