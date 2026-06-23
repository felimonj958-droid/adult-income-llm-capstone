from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Features(BaseModel):
    age: int = Field(..., ge=17, le=90)
    workclass: str
    education: str
    education_num: int = Field(..., ge=1, le=20, alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., ge=0, alias="capital-gain")
    capital_loss: int = Field(..., ge=0, alias="capital-loss")
    hours_per_week: int = Field(..., ge=1, le=100, alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    @field_validator(
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
        mode="before",
    )
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("must be a string")
        v = v.strip()
        if not v:
            raise ValueError("must not be empty")
        return v

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        },
    }


class ParsedFeatures(Features):
    pass


class PredictionResult(BaseModel):
    prediction: str = Field(..., description="Predicted income class label, e.g. <=50K or >50K.")
    probability_gt_50k: float = Field(..., ge=0.0, le=1.0)
    model_name: str

    @field_validator("prediction")
    @classmethod
    def validate_prediction(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("prediction must not be empty")
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("model_name must not be empty")
        return v


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User message sent to the chat endpoint.",
    )
    system_prompt: Optional[str] = Field(
        default="You are a concise assistant for the Adult Income capstone project.",
        max_length=500,
        description="Optional system prompt used to guide the assistant.",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty or whitespace")
        return v

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        return v or None

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": (
                    "I am 37, married, have a bachelor's degree, work full-time "
                    "in a professional job in the US, and want to know if the "
                    "model thinks I earn more than 50k."
                ),
                "system_prompt": "You are a concise assistant for the Adult Income capstone project.",
            }
        }
    }


class ChatResponse(BaseModel):
    parsed_features: Optional[ParsedFeatures] = None
    prediction: Optional[str] = None
    probability_gt_50k: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_name: Optional[str] = None
    explanation: str
    needs_clarification: bool = False

    @field_validator("explanation")
    @classmethod
    def validate_explanation(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("explanation must not be empty")
        return v

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "parsed_features": {
                    "age": 37,
                    "workclass": "Private",
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Married-civ-spouse",
                    "occupation": "Prof-specialty",
                    "relationship": "Husband",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 0,
                    "capital-loss": 0,
                    "hours-per-week": 45,
                    "native-country": "United-States",
                },
                "prediction": ">50K",
                "probability_gt_50k": 0.82,
                "model_name": "GradientBoostingClassifier",
                "explanation": (
                    "Based on the provided profile, the model predicts income "
                    "above $50K. This appears to be influenced by education, "
                    "occupation, marital status, and hours worked per week."
                ),
                "needs_clarification": False,
            }
        },
    }


class ExplainPredictionResponse(BaseModel):
    prediction: str
    probability_gt_50k: float = Field(..., ge=0.0, le=1.0)
    model_name: str
    explanation: str

    @field_validator("prediction", "model_name", "explanation")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be empty")
        return v
