from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _clean_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


class Features(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={
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
    )

    age: int = Field(..., ge=17, le=90)
    workclass: str
    education: str
    education_num: int = Field(..., ge=1, le=20, alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(0, ge=0, alias="capital-gain")
    capital_loss: int = Field(0, ge=0, alias="capital-loss")
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
    def validate_text_fields(cls, value: str, info):
        return _clean_text(value, info.field_name)


class PredictionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: str = Field(..., description="Predicted income class label.")
    probability_gt_50k: float = Field(..., ge=0.0, le=1.0)
    model_name: str

    @field_validator("prediction", "model_name", mode="before")
    @classmethod
    def validate_text_fields(cls, value: str, info):
        return _clean_text(value, info.field_name)


class ExplainPredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: str
    probability_gt_50k: float = Field(..., ge=0.0, le=1.0)
    model_name: str
    explanation: str

    @field_validator("prediction", "model_name", "explanation", mode="before")
    @classmethod
    def validate_text_fields(cls, value: str, info):
        return _clean_text(value, info.field_name)


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "message": (
                    "I am 37, married, have a bachelor's degree, work full-time "
                    "in a professional job in the US, and want to know if the model "
                    "thinks I earn more than 50k."
                )
            }
        },
    )

    message: str = Field(..., min_length=1, max_length=1000)

    @field_validator("message", mode="before")
    @classmethod
    def validate_message(cls, value: str):
        return _clean_text(value, "message")


class ChatResponse(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    parsed_features: Optional[Features] = None
    prediction: Optional[str] = None
    probability_gt_50k: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_name: Optional[str] = None
    explanation: str
    needs_clarification: bool = False

    @field_validator("prediction", "model_name", mode="before")
    @classmethod
    def validate_optional_text_fields(cls, value: Optional[str], info):
        if value is None:
            return value
        return _clean_text(value, info.field_name)

    @field_validator("explanation", mode="before")
    @classmethod
    def validate_explanation(cls, value: str):
        return _clean_text(value, "explanation")