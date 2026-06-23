from src.llm.client import LLMClient, LLMClientError
from src.schemas.inference import Features, PredictionResult

EXPLANATION_SYSTEM_PROMPT = (
    "You are a helpful assistant that explains machine learning "
    "predictions in plain English. "
    "Be clear, cautious, and concise. "
    "Do not claim certainty, and do not imply the model knows facts it was not given."
)


def build_explanation_prompt(
    features: Features,
    prediction_result: PredictionResult,
    user_message: str | None = None,
) -> str:
    user_context = f"\nOriginal user message:\n{user_message}\n" if user_message else ""

    return f"""
You are explaining an Adult Income model prediction to a non-technical user.{user_context}

Input features:
- age: {features.age}
- workclass: {features.workclass}
- education: {features.education}
- education_num: {features.education_num}
- marital_status: {features.marital_status}
- occupation: {features.occupation}
- relationship: {features.relationship}
- race: {features.race}
- sex: {features.sex}
- capital_gain: {features.capital_gain}
- capital_loss: {features.capital_loss}
- hours_per_week: {features.hours_per_week}
- native_country: {features.native_country}

Model output:
- prediction: {prediction_result.prediction}
- probability_gt_50k: {prediction_result.probability_gt_50k:.4f}
- model_name: {prediction_result.model_name}

Instructions:
- Explain the result in 3 to 5 sentences.
- Use plain English.
- Be cautious and avoid claiming certainty.
- Say this is a model prediction, not a statement of fact.
- Mention a few influential-looking profile factors, but do not invent feature importance.
- Do not mention internal implementation details unless they help the user understand the result.
""".strip()


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

    explanation = llm_client.chat(
        user_message=prompt,
        system_prompt=EXPLANATION_SYSTEM_PROMPT,
    )

    explanation = explanation.strip()
    if not explanation:
        raise LLMClientError("Explanation generation returned an empty response.")

    return explanation
