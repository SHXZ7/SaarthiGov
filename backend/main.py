from fastapi import FastAPI
from retrieval import retrieve_chunks
from models import QueryRequest, AskRequest, AskResponse
from llm import synthesize_answer
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


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Main endpoint: Retrieves relevant chunks and synthesizes a clear answer using LLM.
    This is the recommended endpoint for end users.
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve_chunks(request.query, request.top_k)
    
    # Step 2: Synthesize answer using LLM
    answer = synthesize_answer(request.query, chunks)
    
    # Step 3: Return synthesized answer (optionally include sources)
    return AskResponse(
        query=request.query,
        answer=answer,
        sources=chunks if request.include_sources else []
    )


@app.post("/retrieve")
def retrieve(request: QueryRequest):
    """
    Raw retrieval endpoint: Returns chunks without LLM synthesis.
    Useful for debugging or when you want raw search results.
    """
    results = retrieve_chunks(request.query, request.top_k)
    return {
        "query": request.query,
        "results": results
    }
