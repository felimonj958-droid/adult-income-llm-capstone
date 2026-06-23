from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from src.llm.client import LLMClient, LLMClientError
from src.llm.dependencies import get_llm_client
from src.schemas.inference import ChatRequest, ChatResponse, ParsedFeatures
from src.services.explanation import build_prediction_explanation
from src.services.predict import predict_from_features

router = APIRouter(tags=["chat"])


EXTRACTION_PROMPT = """
You extract structured Adult Income features from a user's natural-language message.

Return valid JSON only.

Required keys:
age, workclass, education, education-num, marital-status, occupation,
relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country

Rules:
- Use only information explicitly provided by the user.
- Do not guess missing values.
- If any required field is missing or ambiguous, return:
  {"needs_clarification": true, "explanation": "brief message describing what is missing"}
- If all required fields are present, return a JSON object with exactly the required keys.
- capital-gain and capital-loss may be 0 only if your application policy explicitly allows that default.
- Do not include markdown, commentary, or extra text.
""".strip()


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    llm_client: LLMClient = Depends(get_llm_client),
) -> ChatResponse:
    try:
        extraction = llm_client.chat(
            user_message=payload.message,
            system_prompt=EXTRACTION_PROMPT,
        )
        parsed = llm_client.parse_json(extraction)

        if parsed.get("needs_clarification"):
            return ChatResponse(
                explanation=parsed.get(
                    "explanation",
                    "I need a few more details to extract the required Adult Income features.",
                ),
                needs_clarification=True,
            )

        features = ParsedFeatures(**parsed)
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

    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unable to parse the request into valid model features.",
                "errors": exc.errors(),
            },
        ) from exc
    except LLMClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the chat request.",
        ) from exc
