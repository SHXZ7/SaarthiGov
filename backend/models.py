from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class ChunkResponse(BaseModel):
    service: str
    state: str
    section: str
    text: str
    score: float
