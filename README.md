# Adult Income LLM Capstone

A modular machine learning and FastAPI project that predicts whether income exceeds $50K and uses an LLM to explain model predictions in plain English. The project combines a classical scikit-learn training pipeline, MLflow experiment tracking, a refactored FastAPI service layer, and request validation plus rate limiting for the LLM-backed endpoints.

## Overview

This project trains a classification model on the Adult Income dataset and serves predictions through a FastAPI API. It also includes an LLM-assisted explanation endpoint that turns structured model output into a short, non-technical explanation for end users.

The repository is organized so training, evaluation, prediction, LLM integration, and API routing stay separated. That makes the project easier to maintain, test, and extend as more MLOps or security features are added.

## Project structure

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

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## .env (MUST .gitignore REAL API KEY)
NEBIUS_API_KEY=your_api_key_here 
LLM_API_BASE_URL=https://api.studio.nebius.ai
LLM_MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct

## Training piepline/ Classical ML Model
python -m src.train

## Evaluate MLFlow
python -m src.evaluate

## FastAPI 
uvicorn src.main:app --reload
http://127.0.0.1:8000/docs

Quickstart” section in the README with these commands:
•	 python -m venv .venv 
•	 source .venv/bin/activate 
•	 pip install -r requirements.txt 
•	 python -m src.train 
•	 python -m src.evaluate 
•	 python -m pytest tests -v 
•	 python -m uvicorn src.main:app --reload 



