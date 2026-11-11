from common.schemas.chat import ChatRequest, ChatResponse
from fastapi import APIRouter

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def ask_corpus(req: ChatRequest) -> ChatResponse:
    return ChatResponse(
        answer="В корпусе нет подтверждённых данных по данному вопросу.",
        has_answer_in_corpus=False,
        citations=[],
    )
