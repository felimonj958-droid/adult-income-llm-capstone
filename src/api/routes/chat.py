from fastapi import APIRouter, Depends, HTTPException

from src.llm.client import LLMClient, LLMClientError
from src.llm.dependencies import get_llm_client
from src.schemas.inference import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    llm_client: LLMClient = Depends(get_llm_client),
) -> ChatResponse:
    try:
        message = llm_client.chat(
            user_message=payload.message,
            system_prompt=payload.system_prompt,
        )
        return ChatResponse(message=message)
    except LLMClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
