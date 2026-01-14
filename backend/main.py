from fastapi import FastAPI
from retrieval import retrieve_chunks
from models import QueryRequest
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Kerala Government Services Assistant",
    description="Multilingual retrieval API for government services",
    version="1.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/retrieve")
def retrieve(request: QueryRequest):
    results = retrieve_chunks(request.query, request.top_k)
    return {
        "query": request.query,
        "results": results
    }
