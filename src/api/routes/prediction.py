from fastapi import APIRouter, Depends, HTTPException

from src.llm.client import LLMClient, LLMClientError
from src.llm.dependencies import get_llm_client
from src.schemas.inference import ExplainPredictionResponse, Features, PredictionResult
from src.services.explanation import (
    EXPLANATION_SYSTEM_PROMPT,
    build_explanation_prompt,
)
from src.services.predict import predict_from_features

router = APIRouter(tags=["prediction"])


@router.post("/predict-structured", response_model=PredictionResult)
def predict_structured(features: Features) -> PredictionResult:
    try:
        return predict_from_features(features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/explain-prediction", response_model=ExplainPredictionResponse)
def explain_prediction(
    features: Features,
    llm_client: LLMClient = Depends(get_llm_client),
) -> ExplainPredictionResponse:
    try:
        prediction_result = predict_from_features(features)
        prompt = build_explanation_prompt(features, prediction_result)

        explanation = llm_client.chat(
            user_message=prompt,
            system_prompt=EXPLANATION_SYSTEM_PROMPT,
        )

        return ExplainPredictionResponse(
            prediction=prediction_result.prediction,
            probability_gt_50k=prediction_result.probability_gt_50k,
            model_name=prediction_result.model_name,
            explanation=explanation,
        )
    except LLMClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
