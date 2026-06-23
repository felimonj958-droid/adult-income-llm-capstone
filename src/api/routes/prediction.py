from fastapi import APIRouter, Depends, HTTPException

from src.llm.client import LLMClient, LLMClientError
from src.llm.dependencies import get_llm_client
from src.schemas.inference import ExplainPredictionResponse, Features, PredictionResult
from src.services.explanation import build_prediction_explanation
from src.services.predict import predict_from_features

router = APIRouter(tags=["prediction"])


@router.post("/predict-structured", response_model=PredictionResult)
def predict_structured(features: Features) -> PredictionResult:
    try:
        return predict_from_features(features)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the prediction.",
        ) from exc


@router.post("/explain-prediction", response_model=ExplainPredictionResponse)
def explain_prediction(
    features: Features,
    llm_client: LLMClient = Depends(get_llm_client),
) -> ExplainPredictionResponse:
    try:
        prediction_result = predict_from_features(features)

        explanation = build_prediction_explanation(
            llm_client=llm_client,
            features=features,
            prediction_result=prediction_result,
        )

        return ExplainPredictionResponse(
            prediction=prediction_result.prediction,
            probability_gt_50k=prediction_result.probability_gt_50k,
            model_name=prediction_result.model_name,
            explanation=explanation,
        )
    except LLMClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while explaining the prediction.",
        ) from exc
