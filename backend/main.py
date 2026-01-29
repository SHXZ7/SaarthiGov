from fastapi import FastAPI
from retrieval import retrieve_chunks
from models import QueryRequest, AskRequest, AskResponse
from llm import (
    synthesize_answer,
    is_malayalam,
    rewrite_query,  
    translate_ml_to_en,
    translate_en_to_ml
)
from fastapi.middleware.cors import CORSMiddleware
from service_detection import detect_service
from next_step_recommender import recommend_next_steps
from utils import detect_current_intent


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
    original_query = request.query
    history = request.history or []

    # 🌐 Language detection
    malayalam = is_malayalam(original_query)

    if malayalam:
        query_for_rag = translate_ml_to_en(original_query)
    else:
        query_for_rag = original_query

    # 🧠 Rewrite follow-up into standalone query
    standalone_query = rewrite_query(query_for_rag, history)

# 🔍 Detect service from question
# 🔍 Detect service from question (ONLY as fallback)
    service = request.service
    detected_service = detect_service(standalone_query)

# ✅ Auto-detect ONLY if dropdown is NOT selected
    if service is None and detected_service:
        service = detected_service

    # 🛑 HARD STOP if service still unknown
    if not service:
        return AskResponse(
            query=original_query,
            answer=(
                "I can help with ration card, birth certificate, or unemployment allowance. "
                "Please specify the service."
            ),
            language="ml" if malayalam else "en",
            sources=[],
            service=None,
            next_steps=[]
        )

    # 📥 STEP 1: Retrieve chunks (STRICT service)
    chunks = retrieve_chunks(
        standalone_query,
        service=service,
        k=request.top_k
    )

    # 🧭 STEP 2: Detect intent EARLY
    current_intent = detect_current_intent(chunks)

    # 🔒 STEP 3: Intent-based chunk filtering
    if current_intent:
        intent_filtered_chunks = [
            c for c in chunks
            if current_intent in c["section"].lower()
        ]

        # Safety: ensure at least one chunk survives
        if intent_filtered_chunks:
            chunks = intent_filtered_chunks
        else:
            chunks = chunks[:1]

    # 🤖 STEP 4: NOW synthesize answer
    english_answer = synthesize_answer(
        standalone_query,
        chunks,
        history
    )

    # 🌍 Translate back if needed
    final_answer = (
        translate_en_to_ml(english_answer)
        if malayalam else english_answer
    )

    # 🔮 Next step recommendation (unchanged)
    next_steps = (
        recommend_next_steps(service, current_intent)
        if current_intent else []
    )

    return AskResponse(
        query=original_query,
        answer=final_answer,
        language="ml" if malayalam else "en",
        sources=chunks if request.include_sources else [],
        service=service,
        next_steps=next_steps
    )

@app.post("/retrieve")
def retrieve(request: QueryRequest):
    """
    Raw retrieval endpoint: Returns chunks without LLM synthesis.
    STRICT: service must be specified.
    """

    if request.service is None:
        return {
            "error": "Service must be specified for retrieval.",
            "available_services": ["ration_card", "birth_certificate", "unemployment_allowance"]
        }

    results = retrieve_chunks(
        request.query,
        service=request.service,
        k=request.top_k
    )

    return {
        "query": request.query,
        "service": request.service,
        "results": results
    }
