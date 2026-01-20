from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    service: Optional[str] = None  # 'ration_card', 'birth_certificate', or None for all


class AskRequest(BaseModel):
    """Request model for the /ask endpoint with LLM synthesis"""
    query: str
    top_k: int = 5  # Retrieve more chunks for better context
    include_sources: bool = False  # Whether to include source chunks in response
    service: Optional[str] = None  # 'ration_card', 'birth_certificate', or None for all


class ChunkResponse(BaseModel):
    service: str
    state: str
    section: str
    text: str
    score: float


class AskResponse(BaseModel):
    """Response model for the /ask endpoint"""
    query: str
    answer: str  # The synthesized answer from LLM
    language: str = "en"  # 'en' for English, 'ml' for Malayalam
    sources: List[ChunkResponse] = []  # Optional: source chunks used
