from src.schemas.inference import Features, PredictionResult

EXPLANATION_SYSTEM_PROMPT = (
    "You are a helpful assistant that explains machine learning "
    "predictions in plain English."
)


def build_explanation_prompt(
    features: Features,
    prediction_result: PredictionResult,
) -> str:
    return f"""
You are explaining an Adult Income model prediction to a non-technical user.

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
- probability_gt_50k: {prediction_result.probability_gt_50k}
- model_name: {prediction_result.model_name}

Explain this result in 3 to 5 sentences. Be clear, cautious, and avoid claiming certainty.
""".strip()
