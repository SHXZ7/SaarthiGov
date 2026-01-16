from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class AskRequest(BaseModel):
    """Request model for the /ask endpoint with LLM synthesis"""
    query: str
    top_k: int = 5  # Retrieve more chunks for better context
    include_sources: bool = False  # Whether to include source chunks in response


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
    sources: List[ChunkResponse] = []  # Optional: source chunks used
