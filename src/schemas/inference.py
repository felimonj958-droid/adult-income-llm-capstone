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


class PredictionResult(BaseModel):
    prediction: str
    probability_gt_50k: float = Field(..., ge=0.0, le=1.0)
    model_name: str


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User message sent to the chat endpoint."
    )
    system_prompt: Optional[str] = Field(
        default="You are a concise assistant for the Adult Income capstone project.",
        max_length=500,
        description="Optional system prompt used to guide the assistant."
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
                "message": "Summarize the Adult Income capstone in 2 sentences.",
                "system_prompt": "You are a concise assistant for the Adult Income capstone project."
            }
        }
    }


class ChatResponse(BaseModel):
    message: str


class ExplainPredictionResponse(BaseModel):
    prediction: str
    probability_gt_50k: float = Field(..., ge=0.0, le=1.0)
    model_name: str
    explanation: str

