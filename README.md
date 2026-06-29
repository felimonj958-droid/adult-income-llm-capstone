# Adult Income LLM Capstone

A modular machine learning and FastAPI project that predicts whether annual income exceeds $50K and uses an LLM to explain predictions in plain English. The project combines a scikit-learn training pipeline, MLflow experiment tracking, file-based artifact management, and a refactored FastAPI application with structured prediction and chat-based explanation flows.

## Overview

This project trains a classification model on the Adult Income dataset and serves predictions through a FastAPI API. It also includes LLM-backed explanation endpoints that translate model output into short, non-technical explanations for end users.

The repository is organized so training, evaluation, prediction, LLM integration, and API routing remain separated. That structure improves maintainability, testing, and future extensibility for MLOps and MLSecOps-oriented enhancements.

## Dataset

This project uses the Adult (Census Income) dataset from the UCI Machine Learning Repository to predict whether a person’s annual income exceeds $50,000 based on demographic and work-related attributes such as age, education, occupation, marital status, and hours worked per week.

## Dataset Suitability and Limitations

The Adult dataset is useful for demonstrating an end-to-end machine learning workflow because it contains a clear prediction target, a mix of categorical and numerical features, and a well-known benchmark structure that makes model comparison straightforward. It is especially useful here for showing preprocessing, classification, evaluation, explanation, and MLflow experiment tracking in a compact capstone project.

However, this dataset also has important limitations. It reflects historical U.S. census data rather than current labor-market conditions and should be treated as a demonstration dataset rather than a production decision system.

Because of these constraints, this model should be interpreted as an educational and engineering demonstration, not as a production-ready system for making real financial, hiring, or policy decisions.

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
├── preprocess.py
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
├── test_llm_routes.py
└── test_prediction_routes.py
```

## Features

- Structured prediction endpoint for validated Adult Income features.
- Explanation endpoint for plain-English model interpretation.
- Chat endpoint that converts natural language into structured Adult dataset features.
- MLflow experiment tracking for comparing runs and locating the best model.
- Modular FastAPI layout with separated routes, services, schemas, and model access.
- Docker support for containerized local deployment.

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/felimonj958-droid/adult-income-llm-capstone
cd adult-income-llm-capstone
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a local environment file

Create a `.env` file in the repository root:

```env
LLM_API_BASE_URL="https://api.tokenfactory.nebius.com/v1"
NEBIUS_API_KEY="YOUR_REAL_API_KEY"
LLM_MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
```

### 5. Start the API locally

```bash
python -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Local Development Notes

If using VS Code, a minimal `.vscode/settings.json` can help keep `.env` handling consistent during local development:

```json
{
  "python.analysis.typeCheckingMode": "basic",
  "python.envFile": "${workspaceFolder}/.env",
  "python.terminal.activateEnvironment": true,
  "python.terminal.useEnvFile": true
}
```

This helps local shells inside VS Code, but Docker and non-VS Code terminal sessions still require explicit environment configuration.

## MLflow

### Start MLflow UI

```bash
mlflow ui --backend-store-uri mlruns
```

Then open:

```text
http://127.0.0.1:5000
```

### Find the best model with MLflow

```python
import mlflow

best_run = mlflow.search_runs(
    experiment_names=["adult-income-experiments"],
    max_results=1,
    order_by=["metrics.accuracy DESC"],
)

print(best_run[["run_id", "metrics.accuracy", "params.model_name"]])
```

## API Endpoints

### `POST /chat`

Accepts a natural-language description, extracts Adult Income features, generates a prediction, and returns a plain-English explanation.

Example request:

```json
{
  "message": "I am 37, married, have a bachelor's degree, work full-time in a professional job in the US, and want to know if the model thinks I earn more than 50k."
}
```

### `POST /predict-structured`

Accepts a fully structured Adult Income feature payload and returns a prediction.

### `POST /explain-prediction`

Accepts a fully structured payload and returns a prediction plus a plain-English explanation.

## Running Tests

Run the test suite:

```bash
python -m pytest tests -v
```

## Docker

### 1. Build the image

```bash
docker build -t adult-income-llm-capstone:latest .
```

If Docker prints an image ID, record it here for reference:

```text
Docker Image ID: adult-income-llm-capstone:latest
                                c8400bb16eae       2.08GB   
```

### 2. Run the container

```bash
docker run --rm -p 8000:8000 \
  -e LLM_API_BASE_URL="https://api.tokenfactory.nebius.com/v1" \
  -e NEBIUS_API_KEY="YOUR_REAL_API_KEY" \
  -e LLM_MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct" \
  adult-income-llm-capstone:latest
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## GitHub Setup

Before pushing:

- Keep `.env` out of version control.
- Commit a `.env.example` with placeholder values only.
- Make sure `.gitignore` excludes `.env`, `.venv/`, and local-only artifacts.
- Verify the README commands match the real local workflow.

Recommended `.env.example`:

```env
LLM_API_BASE_URL="https://api.tokenfactory.nebius.com/v1"
NEBIUS_API_KEY="replace_me"
LLM_MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
```

## Hardest Engineering Challenge

The most difficult part of the project was not model training. The hardest part was making the LLM-assisted API flow reliable from environment configuration through response parsing and FastAPI validation.

Three issues mattered most:

1. API key handling across terminals and editor sessions.
2. Adapting the LLM client to the provider's OpenAI-compatible chat completion format.
3. Preventing natural-language extraction from failing validation when fields like `capital-gain` and `capital-loss` were missing or null.

## Learning Reflection

This project reinforced that production-style machine learning work is often less about the model itself and more about the interfaces around it. The model can be correct, yet the product still fails if environment variables are loaded incorrectly, if the LLM response parser is too strict, or if API validation expects values that natural-language inputs do not always provide.

The main resolution was to simplify and harden each layer:

- make local environment loading explicit;
- verify runtime values in the same shell used to launch the app;
- simplify the LLM response parser to match the actual provider response format;
- normalize extracted chat fields before validating them as structured features.

That process improved the reliability of the `/chat` route and made the overall system much smoother to demo and maintain.

## Troubleshooting

### Virtual environment not active

Check the active interpreter:

```bash
which python
python --version
```

If needed:

```bash
source .venv/bin/activate
```

### API key not loading correctly

Check what Python actually sees:

```bash
python - <<'PY'
import os
key = os.getenv("NEBIUS_API_KEY") or os.getenv("LLM_API_KEY")
print("KEY EXISTS:", bool(key))
print("KEY LENGTH:", len(key) if key else 0)
print("KEY LAST4:", key[-4:] if key else None)
print("BASE URL:", os.getenv("LLM_API_BASE_URL"))
print("MODEL:", os.getenv("LLM_MODEL_NAME"))
PY
```

If the shell has stale values, reset and export them manually:

```bash
unset NEBIUS_API_KEY
unset LLM_API_KEY
unset LLM_API_BASE_URL
unset LLM_MODEL_NAME

export LLM_API_BASE_URL="https://api.tokenfactory.nebius.com/v1"
export LLM_MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export NEBIUS_API_KEY='YOUR_REAL_API_KEY'
```

### `/chat` returns 422

This usually means natural-language extraction returned incomplete or invalid feature values.

Steps:
- verify `src/schemas/inference.py` defaults or validators are updated;
- verify `src/api/routes/chat.py` normalizes extracted fields before validation;
- restart Uvicorn completely;
- hard refresh `/docs` and test again.

### Hard restart and refresh

Stop and restart the server:

```bash
CTRL+C
python -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

Then hard refresh the docs page:

- Chrome / Edge on Mac: `Command + Shift + R`
- Safari on Mac: `Option + Command + R`

If needed, refresh the schema directly:

```text
http://127.0.0.1:8000/openapi.json
```

## Demo Video

[Watch the project demo video](
    https://drive.google.com/file/d/1qfaEl356QfkfhwU0VW2Scj4nyGKkIAU8/view?usp=drive_link

## Next Improvements

- Add stronger clarification behavior for ambiguous chat inputs.
- Add more robust LLM extraction tests for missing or partial feature descriptions.
- Register and alias the best model in MLflow for cleaner promotion workflows.
- Add container validation steps to confirm local and Docker behavior match.

## License

MIT License. See the `LICENSE` file for details.