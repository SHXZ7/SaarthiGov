from fastapi import FastAPI
from retrieval import retrieve_chunks
from models import QueryRequest, AskRequest, AskResponse
from llm import (
    synthesize_answer,
    is_malayalam,
    translate_ml_to_en,
    translate_en_to_ml
)
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
    Supports Malayalam queries with automatic translation.
    """
    original_query = request.query
    
    # Step 1: Detect language
    malayalam = is_malayalam(original_query)
    
    # Step 2: Translate query if Malayalam
    if malayalam:
        query_for_rag = translate_ml_to_en(original_query)
        print(f"Translated query: {query_for_rag}")
    else:
        query_for_rag = original_query
    
    # Step 3: Retrieve relevant chunks (using English query)
    chunks = retrieve_chunks(query_for_rag, service=request.service, k=request.top_k)
    
    # Step 4: Synthesize answer using LLM (in English)
    english_answer = synthesize_answer(query_for_rag, chunks)
    
    # Step 5: Translate answer back if original was Malayalam
    if malayalam:
        final_answer = translate_en_to_ml(english_answer)
    else:
        final_answer = english_answer
    
    # Step 6: Return response
    return AskResponse(
        query=original_query,
        answer=final_answer,
        language="ml" if malayalam else "en",
        sources=chunks if request.include_sources else []
    )


@app.post("/retrieve")
def retrieve(request: QueryRequest):
    """
    Raw retrieval endpoint: Returns chunks without LLM synthesis.
    Useful for debugging or when you want raw search results.
    """
    results = retrieve_chunks(request.query, service=request.service, k=request.top_k)
    return {
        "query": request.query,
        "service": request.service,
        "results": results
    }
