from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    service: Optional[str] = None  # 'ration_card', 'birth_certificate', or None for all

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class AskRequest(BaseModel):
    query: str
    top_k: int = 3
    include_sources: bool = False
    service: Optional[str] = None
    history: List[ChatMessage] = []  
    next_steps: List[str] = [] 


class ChunkResponse(BaseModel):
    service: str
    state: str
    section: str
    text: str
    score: float


class AskResponse(BaseModel):
    query: str
    answer: str
    language: str = "en"
    sources: List[ChunkResponse] = []
    service: Optional[str] = None
    next_steps: List[str] = []   # ✅ REQUIRED
