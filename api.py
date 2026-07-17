from fastapi import FastAPI
from pydantic import BaseModel, Field

from moderador import censurar_mensagem


app = FastAPI(
    title="SafeChat Moderation API",
    version="1.0.0"
)


class ModerationRequest(BaseModel):
    content: str = Field(min_length=1, max_length=500)


class ModerationResponse(BaseModel):
    original_content: str
    content: str
    moderated: bool


@app.get("/health")
def verificar_saude() -> dict[str, str]:
    """Informa ao SafeChat que o serviço de moderação está disponível."""
    return {"status": "ok"}


@app.post("/moderate", response_model=ModerationResponse)
def moderar(
    request: ModerationRequest
) -> ModerationResponse:
    """Recebe o conteúdo do SafeChat e devolve o resultado da moderação."""
    moderated_content = censurar_mensagem(
        request.content
    )

    return ModerationResponse(
        original_content=request.content,
        content=moderated_content,
        moderated=moderated_content != request.content
    )
