from fastapi import APIRouter, Depends, HTTPException, status

from src.llm.client import LLMClient
from src.llm.dependencies import get_llm_client
from src.ml.model_registry import ModelRegistryError
from src.schemas.inference import ExplainPredictionResponse, Features, PredictionResult
from src.services.explanation import build_prediction_explanation
from src.services.predict import predict_from_features

router = APIRouter(tags=["prediction"])


def _run_prediction(features: Features) -> PredictionResult:
    try:
        return predict_from_features(features)
    except ModelRegistryError:
        raise
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating the prediction.",
        ) from exc


@router.post(
    "/predict-structured",
    response_model=PredictionResult,
    status_code=status.HTTP_200_OK,
)
def predict_structured(features: Features) -> PredictionResult:
    try:
        return _run_prediction(features)
    except ModelRegistryError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction model is unavailable: {exc}",
        ) from exc


@router.post(
    "/explain-prediction",
    response_model=ExplainPredictionResponse,
    status_code=status.HTTP_200_OK,
)
def explain_prediction(
    features: Features,
    llm_client: LLMClient = Depends(get_llm_client),
) -> ExplainPredictionResponse:
    try:
        prediction_result = _run_prediction(features)
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
    except ModelRegistryError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction model is unavailable: {exc}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while explaining the prediction.",
        ) from exc