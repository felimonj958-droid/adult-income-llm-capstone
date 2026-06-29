
```md
# Adult Income LLM Capstone

A modular machine learning and FastAPI project that predicts whether annual income exceeds $50K and uses an LLM to explain predictions in plain English. The project combines a scikit-learn training pipeline, MLflow experiment tracking, DVC-based artifact versioning, and a refactored FastAPI application with structured prediction and chat-based explanation flows. 

## Overview

This project trains a classification model on the Adult Income dataset and serves predictions through a FastAPI API. It also includes LLM-backed explanation endpoints that translate model output into short, non-technical explanations for end users. 

The repository is organized so training, evaluation, prediction, LLM integration, and API routing remain separated. That structure improves maintainability, testing, and future extensibility for MLOps and MLSecOps-oriented enhancements. 

## Dataset

This project uses the **Adult (Census Income)** dataset from the UCI Machine Learning Repository to predict whether a person’s annual income exceeds $50,000 based on demographic and work-related attributes such as age, education, occupation, marital status, and hours worked per week. The dataset was extracted from the 1994 U.S. Census database by Barry Becker and is commonly used for binary classification and model comparison tasks.

## Dataset Suitability and Limitations

The Adult dataset is useful for demonstrating an end-to-end machine learning workflow because it contains a clear prediction target, a mix of categorical and numerical features, and a well-known benchmark structure that makes model comparison straightforward. It is especially useful here for showing preprocessing, classification, evaluation, explanation, and MLflow experiment tracking in a compact capstone project.

However, this dataset also has important limitations. It reflects historical U.S. census data rather than current labor-market conditions, and fairness-focused work has noted that the Adult dataset is old, has limited documentation, and includes outdated feature encodings that reduce its suitability for modern real-world decision systems.

Because of these constraints, this model should be interpreted as a **demonstration and educational tool**, not as a production-ready system for making real financial, hiring, or policy decisions. Its predictions may be less reliable when applied outside the dataset’s original population, time period, or social context.


## Project Structure

```text
src/
├── main.py
├── app.py
├── api/
│   ├── router.py
│   └── routes/
│       ├── health.py
│       ├── chat.py
│       └── prediction.py
├── core/
│   ├── logging.py
│   └── middleware.py
├── llm/
│   ├── client.py
│   └── dependencies.py
├── ml/
│   └── model_registry.py
├── schemas/
│   └── inference.py
├── services/
│   ├── explanation.py
│   └── predict.py
├── utils/
│   └── config.py
├── train.py
└── evaluate.py

tests/
└── test_llm_routes.py
```

## Quickstart

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
NEBIUS_API_KEY=your_api_key_here
LLM_API_BASE_URL=https://api.studio.nebius.ai
LLM_MODEL_NAME=deepseek-ai/DeepSeek-V3.2
MLFLOW_TRACKING_URI=file:./mlruns
MODEL_PATH=models/best_model.joblib
```

Do not commit real API keys. Keep `.env` ignored by Git. 

### 4. Train the model

```bash
python -m src.train
```

### 5. Evaluate the model

```bash
python -m src.evaluate
```

### 6. Run tests

```bash
python -m pytest tests -v
```

### 7. Start the FastAPI app

```bash
python -m uvicorn src.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## MLflow Tracking

MLflow is used to track experiments, parameters, and evaluation metrics across training runs. Local runs are stored in the `mlruns/` directory when `MLFLOW_TRACKING_URI=file:./mlruns` is configured. 

To launch the MLflow UI locally:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open:

```text
http://127.0.0.1:5000
```

## DVC Workflow

DVC is used to version larger data and model artifacts separately from Git, which helps keep the repository lightweight while preserving reproducibility. 

To fetch tracked artifacts from a configured remote:

```bash
dvc pull
```

To track a dataset or model artifact with DVC:

```bash
dvc add data/raw/adult.data
dvc add models/best_model.joblib
git add data/raw/adult.data.dvc models/best_model.joblib.dvc .gitignore
```

In this setup:

- Git tracks source code and `.dvc` pointer files.
- DVC tracks large data and model artifacts.
- MLflow tracks runs, parameters, and metrics. 

## API Endpoints

The service exposes the following primary routes: 

- `GET /` — root status endpoint confirming the API is running.
- `POST /predict-structured` — returns a structured income prediction, probability, and model name.
- `POST /explain-prediction` — returns a prediction plus a plain-English explanation from structured features.
- `POST /chat` — extracts features from natural-language input, runs prediction, and returns an LLM-backed explanation.

Use the interactive Swagger UI at `/docs` to test requests in the browser.

## Example: /predict-structured

Example request body:

```json
{
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
  "native-country": "United-States"
}
```

Example response (from a local run): 

```json
{
  "prediction": "<=50K",
  "probability_gt_50k": 0.10938373117856345,
  "model_name": "GradientBoostingClassifier"
}
```

## Example: /chat

Example request body:

```json
{
  "message": "I am 37, married, have a bachelor degree, work full-time in a professional job in the US, and want to know if the model thinks I earn more than 50k."
}
```

Example response shape:

```json
{
  "parsed_features": {
    "age": 37,
    "workclass": "Private",
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
  },
  "prediction": ">50K",
  "probability_gt_50k": 0.82,
  "model_name": "GradientBoostingClassifier",
  "explanation": "Plain-English explanation returned by the LLM.",
  "needs_clarification": false
}
```

The chat route orchestrates natural-language feature extraction, model prediction, and explanation generation in a single call. 

## Local Verification Checklist

Use this checklist to verify the project from a fresh local setup: 

1. Activate `.venv` and install dependencies.
2. Create and load `.env` with valid LLM credentials.
3. Start the API with `python -m uvicorn src.main:app --reload`.
4. Start MLflow with `mlflow ui --backend-store-uri ./mlruns`.
5. Open `/docs` and confirm the OpenAPI schema loads.
6. Test `GET /`.
7. Test `POST /predict-structured`.
8. Test `POST /explain-prediction`.
9. Test `POST /chat`.
10. Run `python -m pytest tests -v`.

## Docker Usage

Build the image from the repository root:

```bash
docker build -t adult-income-llm-capstone .
```

Run the container:

```bash
docker run --rm -p 8000:8000 --env-file .env adult-income-llm-capstone
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Testing

The project includes tests for model behavior, preprocessing, prediction routes, and LLM-backed routes.

Run the full test suite with:

```bash
python -m pytest tests -v
```

## Results Summary

The final deployed model is a Gradient Boosting classifier wrapped in a scikit-learn pipeline, and the live API returns `GradientBoostingClassifier` as the active model in prediction responses. 

The project demonstrates an end-to-end workflow for:

- training a tabular income classifier,
- tracking experiments with MLflow,
- versioning artifacts with DVC,
- serving predictions through FastAPI,
- and layering LLM explanations on top of structured ML outputs. 

## Reflection

This project reinforced that building a useful machine learning application involves more than training a model. The most valuable work included making the pipeline reproducible, comparing runs in MLflow, versioning artifacts with DVC, and exposing the final model through a structured FastAPI service. 

Another key lesson was that LLM integration introduces practical engineering challenges beyond the core classifier itself. Validating environment configuration, parsing model output safely, and handling provider response formats were important parts of making the chat workflow reliable. 

## Limitations

- LLM explanation quality depends on external API availability and model behavior.
- The project uses local MLflow and local Docker execution rather than a production cloud deployment.
- The current setup is optimized for demonstration and portfolio review rather than high-scale production traffic. 

## Future Improvements

- Add CI/CD for automated testing and container validation.
- Add hosted deployment for remote inference.
- Add monitoring, request tracing, and drift checks.
- Improve structured output guarantees for the LLM extraction workflow.
- Add stronger observability and error reporting around external LLM calls. 

## License

This project is licensed under the MIT License. See the `LICENSE` file for full details.
```
