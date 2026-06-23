from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Adult Income Prediction API is running."}


def test_predict_structured_success():
    payload = {
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

    response = client.post("/predict-structured", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability_gt_50k" in data
    assert "model_name" in data

    assert data["prediction"] in ["<=50K", ">50K"]
    assert 0.0 <= data["probability_gt_50k"] <= 1.0
    assert isinstance(data["model_name"], str)
    assert data["model_name"]


def test_predict_structured_invalid_input_returns_422():
    payload = {
        "age": -5,
        "workclass": "Private",
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Sales",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    response = client.post("/predict-structured", json=payload)

    assert response.status_code == 422


def test_predict_structured_missing_required_field_returns_422():
    payload = {
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
    }

    response = client.post("/predict-structured", json=payload)

    assert response.status_code == 422
