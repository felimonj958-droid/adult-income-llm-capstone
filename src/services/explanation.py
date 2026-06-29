from __future__ import annotations

from src.llm.client import LLMClient, LLMClientError
from src.schemas.inference import Features, PredictionResult

MAX_USER_MESSAGE_CHARS = 300

EXPLANATION_SYSTEM_PROMPT = """
You explain Adult Income model predictions in plain English for non-technical users.
Write 3 to 4 short sentences.
Say this is a model prediction, not a fact.
Mention a few relevant profile details from the provided inputs.
Do not invent facts or claim certainty.
""".strip()


def _sanitize_user_message(user_message: str | None) -> str | None:
    if not user_message:
        return None
    cleaned = user_message.strip()
    if not cleaned:
        return None
    if len(cleaned) > MAX_USER_MESSAGE_CHARS:
        cleaned = cleaned[:MAX_USER_MESSAGE_CHARS].rstrip() + "..."
    return cleaned


def build_explanation_prompt(
    features: Features,
    prediction_result: PredictionResult,
    user_message: str | None = None,
) -> str:
    safe_user_message = _sanitize_user_message(user_message)

    user_context = ""
    if safe_user_message:
        user_context = (
            "User message for context only:\n"
            f"{safe_user_message}\n\n"
        )

    return f"""
Explain this Adult Income model result for a non-technical user.

{user_context}Profile:
- age: {features.age}
- workclass: {features.workclass}
- education: {features.education}
- marital_status: {features.marital_status}
- occupation: {features.occupation}
- relationship: {features.relationship}
- race: {features.race}
- sex: {features.sex}
- capital_gain: {features.capital_gain}
- capital_loss: {features.capital_loss}
- hours_per_week: {features.hours_per_week}
- native_country: {features.native_country}

Prediction:
- prediction: {prediction_result.prediction}
- probability_gt_50k: {prediction_result.probability_gt_50k:.4f}
- model_name: {prediction_result.model_name}
""".strip()


def _fallback_explanation(
    features: Features,
    prediction_result: PredictionResult,
) -> str:
    label = prediction_result.prediction
    probability = round(prediction_result.probability_gt_50k, 2)

    return (
        f"This is a model prediction, not a statement of fact. "
        f"Based on the provided profile, the model predicts {label} "
        f"with an estimated probability above 50K of {probability}. "
        f"Relevant inputs include education, occupation, hours worked per week, "
        f"and marital relationship status."
    )


def build_prediction_explanation(
    llm_client: LLMClient,
    features: Features,
    prediction_result: PredictionResult,
    user_message: str | None = None,
) -> str:
    prompt = build_explanation_prompt(
        features=features,
        prediction_result=prediction_result,
        user_message=user_message,
    )

    try:
        explanation = llm_client.chat(
            user_message=prompt,
            system_prompt=EXPLANATION_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=180,
        ).strip()

        if len(explanation) < 20:
            return _fallback_explanation(features, prediction_result)

        return explanation

    except LLMClientError:
        return _fallback_explanation(features, prediction_result)
    except Exception:
        return _fallback_explanation(features, prediction_result)