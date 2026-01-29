import os
import httpx
import time
from typing import List, Dict
from dotenv import load_dotenv
from matplotlib.style import context

# Load environment variables from .env file
load_dotenv()

# --- Language Detection ---
def is_malayalam(text: str) -> bool:
    """Check if text contains Malayalam characters (Unicode range: U+0D00 to U+0D7F)"""
    return any('\u0D00' <= ch <= '\u0D7F' for ch in text)

# OpenRouter API configuration
# Get your API key at: https://openrouter.ai/keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free models to try (in order of preference)
# High-quality free models from https://openrouter.ai/models
FREE_MODELS = [
    "mistralai/mistral-7b-instruct",
    "mistralai/mixtral-8x7b-instruct",

    # LLaMA-based strong general models
    "meta-llama/llama-3-8b-instruct",
    "meta-llama/llama-2-13b-chat",

    # Good for longer reasoning / explanations
    "huggingfaceh4/zephyr-7b-beta",
    "togethercomputer/redpajama-incite-chat-3b",

    # Code + reasoning friendly
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "codellama/codellama-7b-instruct",

    # Lightweight / fast (lower cost, quick replies)
    "google/gemma-7b-it",
    "tiiuae/falcon-7b-instruct",
]



SYSTEM_PROMPT = """You are a helpful Kerala Government Services Assistant. Your job is to answer questions about government services based ONLY on the provided context.

SUPPORTED SERVICES:
- Ration Card (ration_card)
- Birth Certificate (birth_certificate)
- Unemployment Allowance (unemployment_allowance)

⚠️ CRITICAL RULES - FOLLOW STRICTLY:
1. Use ONLY information from the provided context chunks - NEVER make up, guess, or add information
2. If the context does not contain the answer, respond: "I don't have enough information to answer this question. Please try asking about a different aspect of the service."
3. Do NOT invent fees, timelines, document names, or office addresses that are not in the context
4. Do NOT use your general knowledge - only use what is explicitly stated in the chunks
5. Keep answers concise and actionable
6. When chunks from different services appear, focus on the most relevant one
7. Always respond in English (translation is handled separately)

ANSWER TEMPLATES - Use the appropriate format based on question type:

📄 FOR "DOCUMENTS NEEDED" QUESTIONS:
**Documents Required for [Service Name]**
• Document 1
• Document 2
• Document 3
⚠️ Note: [Any important notes from context]

🔄 FOR "PROCESS/HOW TO APPLY" QUESTIONS:
**How to Apply for [Service Name]**

*Online Process:*
1. Step 1 (mention portal name if in context)
2. Step 2
3. Step 3

*Offline Process:*
1. Step 1
2. Step 2

⏱️ Processing Time: [Only if mentioned in context]

✅ FOR "ELIGIBILITY" QUESTIONS:
**Eligibility for [Service Name]**
• Criteria 1
• Criteria 2
• Criteria 3

📍 FOR "WHERE/LOCATION" QUESTIONS:
**Where to Apply**
• Location/Office name (only from context)
• Website or portal (only if mentioned)
• Timings (only if mentioned)

⏰ FOR "TIMELINE/DEADLINE" QUESTIONS:
**Important Timelines**
• Only include timelines explicitly mentioned in context

💡 FOR GENERAL QUESTIONS:
Provide a brief, clear answer with bullet points for key information found in context.

🚫 IF INFORMATION NOT FOUND:
"I don't have enough information about [topic] in my current knowledge base. Please try rephrasing your question or ask about a specific aspect like eligibility, documents required, or application process.\""""


def synthesize_answer(
    query: str,
    chunks: List[Dict],
    history: list = None
) -> str:
    
    history_text = ""
    if history:
        history_text = "\n".join(
        f"{m.role}: {m.content}" for m in history[-4:]
    )


    """
    Takes retrieved chunks and synthesizes a coherent answer using LLM via OpenRouter.
    
    Args:
        query: The user's question
        chunks: List of retrieved chunks with text, service, section, etc.
    
    Returns:
        Synthesized answer string
    """
    if not chunks:
        return "I couldn't find any relevant information for your question. Please try rephrasing or ask about a different government service."
    
    # 🔒 HARD STOP: documents-only questions
    if "document" in query.lower():
        chunks = [
            c for c in chunks
            if "document" in c["section"].lower()
        ] or chunks[:1]
        
    # Format chunks into context string
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Chunk {i}] Service: {chunk['service']} | Section: {chunk['section']}\n{chunk['text']}"
        )
    
    context = "\n\n".join(context_parts)
    
    # Build the user message with context
    user_message = f"""CONTEXT CHUNKS:
        {context}

        CONVERSATION CONTEXT (for reference only):
        {history_text}

        USER QUESTION:
        {query}

    Provide a clear, helpful answer based ONLY on the context above:"""

    try:
        # Try multiple free models with retry logic
        last_error = None
        for model in FREE_MODELS:
            try:
                print(f"Trying model: {model}...")
                response = httpx.post(
                    OPENROUTER_BASE_URL,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "Kerala Government Services Assistant"
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_message}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 350
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                
                result = response.json()
                print(f"Success with model: {model}")
                return result["choices"][0]["message"]["content"]
            except httpx.TimeoutException as e:
                last_error = e
                print(f"Model {model} timed out, trying next...")
                continue
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in [429, 402, 404, 503]:
                    # Rate limited, payment required, model not found, or service unavailable
                    print(f"Model {model} unavailable ({e.response.status_code}), trying next...")
                    time.sleep(0.5)
                    time.sleep(0.5)
                    continue
                raise  # Other errors, don't retry
        
        # All models failed
        raise last_error or Exception("All models failed")
    except Exception as e:
        # Fallback: return formatted chunks if LLM fails
        print(f"LLM synthesis failed: {e}")
        return fallback_response(query, chunks)


def fallback_response(query: str, chunks: List[Dict]) -> str:
    """
    Intent-aware fallback response when LLM is unavailable.
    Returns ONLY the most relevant chunk instead of dumping everything.
    """

    if not chunks:
        return (
            "I don't have enough information to answer this question. "
            "Please try asking about a specific aspect like documents, "
            "eligibility, or application process."
        )

    # Pick the highest-ranked chunk (already sorted by score)
    top_chunk = chunks[0]

    return (
        f"**{top_chunk['section']}**\n"
        f"{top_chunk['text']}"
    )


# --- Reusable LLM Call Function ---
def call_llm(prompt: str, system_prompt: str = None, max_tokens: int = 512) -> str:
    """
    Generic LLM call function for translation and other tasks.
    
    Args:
        prompt: The user prompt/message
        system_prompt: Optional system prompt (defaults to simple assistant)
        max_tokens: Maximum tokens in response
    
    Returns:
        LLM response string
    """
    if system_prompt is None:
        system_prompt = "You are a helpful assistant. Follow instructions precisely."
    
    last_error = None
    for model in FREE_MODELS:
        try:
            response = httpx.post(
                OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "Kerala Government Services Assistant"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": max_tokens
                },
                timeout=60.0
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            last_error = e
            time.sleep(0.5)
            continue
    
    raise last_error or Exception("All models failed")


# --- Translation Functions ---
def translate_ml_to_en(text: str) -> str:
    """Translate Malayalam text to English using LLM."""
    prompt = f"""Translate the following Malayalam text to English.
Do not add, remove, or explain anything.
Only translate.

Text:
{text}"""
    try:
        return call_llm(prompt, max_tokens=256)
    except Exception as e:
        print(f"Translation ML->EN failed: {e}")
        return text  # Return original if translation fails


def translate_en_to_ml(text: str) -> str:
    """Translate English text to Malayalam using LLM."""
    prompt = f"""Translate the following English text to Malayalam.
Keep it clear and simple.
Do not add extra information.

Text:
{text}"""
    try:
        return call_llm(prompt, max_tokens=512)
    except Exception as e:
        print(f"Translation EN->ML failed: {e}")
        return text  # Return original if translation fails

def rewrite_query(user_query: str, history: list) -> str:
    """
    Rewrite a follow-up question into a standalone question using chat history.
    """
    if not history:
        return user_query

    recent = history[-4:]  # limit context
    history_text = "\n".join(
        f"{m.role}: {m.content}" for m in recent
    )

    prompt = f"""
You are a query rewriter for a Kerala government services assistant.

Conversation so far:
{history_text}

User's new question:
{user_query}

Rewrite the user's question into a standalone question.
Do NOT answer.
Do NOT add new information.
"""

    try:
        rewritten = call_llm(prompt, max_tokens=128)
        return rewritten.strip()
    except Exception:
        return user_query
