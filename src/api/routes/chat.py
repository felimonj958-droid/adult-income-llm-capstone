from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError

from src.llm.client import LLMClient, LLMClientError
from src.llm.dependencies import get_llm_client
from src.ml.model_registry import ModelRegistryError
from src.schemas.inference import ChatRequest, ChatResponse, Features
from src.services.explanation import build_prediction_explanation
from src.services.predict import predict_from_features

router = APIRouter(tags=["chat"])

EXTRACTION_PROMPT = """
You convert a user's natural-language description into Adult Income dataset features.

Return valid JSON only.
Do not include markdown.
Do not include commentary.
Do not include reasoning.
Do not include any text before or after the JSON.

Required keys:
age
workclass
education
education-num
marital-status
occupation
relationship
race
sex
capital-gain
capital-loss
hours-per-week
native-country

Rules:
- Use only information explicitly stated by the user.
- Preserve Adult Income key names exactly as written above.
- Use 0 for capital-gain and capital-loss if the user does not mention them.
- Do not invent other missing values.
- If any required field other than capital-gain or capital-loss is missing or ambiguous, return exactly:
  {"needs_clarification": true, "explanation": "brief message describing what is missing"}
- If all required fields are present, return exactly one JSON object containing only the required keys.
- Output JSON only.
""".strip()


def _build_clarification_response(message: str) -> ChatResponse:
    cleaned = message.strip() if isinstance(message, str) and message.strip() else (
        "I need a few more details to extract the required Adult Income features."
    )
    return ChatResponse(
        explanation=cleaned,
        needs_clarification=True,
    )


def _normalize_extracted_features(parsed: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(parsed)
    normalized["capital-gain"] = normalized.get("capital-gain") or 0
    normalized["capital-loss"] = normalized.get("capital-loss") or 0
    return normalized

def _parse_features(parsed: dict[str, Any]) -> Features:
    try:
        return Features(**parsed)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Unable to parse the extracted input into valid Adult Income features.",
                "errors": exc.errors(),
            },
        ) from exc


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def chat(
    payload: ChatRequest,
    llm_client: LLMClient = Depends(get_llm_client),
) -> ChatResponse:
    try:
        
        parsed = llm_client.chat_json(
           user_message=payload.message,
           system_prompt=EXTRACTION_PROMPT,
           temperature=0.0,
           max_tokens=220,
        )

        if parsed.get("needs_clarification") is True:
            return _build_clarification_response(
                parsed.get(
                    "explanation",
                    "I need a few more details to understand your request."
                )
            )

        normalized = _normalize_extracted_features(parsed)
        features = _parse_features(normalized)
        prediction_result = predict_from_features(features)

        explanation = build_prediction_explanation(
            llm_client=llm_client,
            features=features,
            prediction_result=prediction_result,
            user_message=payload.message,
        )

        return ChatResponse(
            parsed_features=features,
            prediction=prediction_result.prediction,
            probability_gt_50k=prediction_result.probability_gt_50k,
            model_name=prediction_result.model_name,
            explanation=explanation,
            needs_clarification=False,
        )

    except LLMClientError:
        return _build_clarification_response(
            "The language model is temporarily unavailable. Please try again or use the structured prediction endpoint."
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
            detail="An unexpected error occurred while processing the chat request.",
        ) from exc