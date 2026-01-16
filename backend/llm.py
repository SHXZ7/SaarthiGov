import os
import httpx
import time
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenRouter API configuration
# Get your API key at: https://openrouter.ai/keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free models to try (in order of preference)
# High-quality free models from https://openrouter.ai/models
FREE_MODELS = [
    "xiaomi/mimo-v2-flash:free",        # Best - 309B MoE, top performance
    "mistralai/devstral-2-2512:free",   # 123B, great for code/agents
    "tngtech/deepseek-r1t2-chimera:free", # 671B MoE, strong reasoning
    "tngtech/deepseek-r1t-chimera:free",  # R1+V3 merge
    "zhipu-ai/glm-4.5-air:free",        # GLM 4.5 Air
]


SYSTEM_PROMPT = """You are a helpful Kerala Government Services Assistant. Your job is to answer questions about government services based on the provided context.

RULES:
1. Use ONLY information from the provided context - NEVER make up information
2. If context is insufficient, say "I don't have complete information about this"
3. If the question is in Malayalam, respond in Malayalam
4. Keep answers concise and actionable

ANSWER TEMPLATES - Use the appropriate format based on question type:

📄 FOR "DOCUMENTS NEEDED" QUESTIONS:
**Documents Required for [Service Name]**
• Document 1
• Document 2
• Document 3
⚠️ Note: [Any important notes about Aadhaar/mandatory docs]

🔄 FOR "PROCESS/HOW TO APPLY" QUESTIONS:
**How to Apply for [Service Name]**

*Online Process:*
1. Step 1
2. Step 2
3. Step 3

*Offline Process:*
1. Step 1
2. Step 2

⏱️ Processing Time: [X days/weeks]

✅ FOR "ELIGIBILITY" QUESTIONS:
**Eligibility for [Service Name]**
• Criteria 1
• Criteria 2
• Criteria 3

📍 FOR "WHERE/LOCATION" QUESTIONS:
**Where to Apply**
• Location/Office name
• Address or website
• Timings (if available)

💡 FOR GENERAL QUESTIONS:
Provide a brief, clear answer with bullet points for key information."""


def synthesize_answer(query: str, chunks: List[Dict]) -> str:
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

USER QUESTION: {query}

Provide a clear, helpful answer based on the context above:"""

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
                        "max_tokens": 512
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
    Fallback response when LLM is unavailable.
    Returns formatted chunks directly.
    """
    lines = [f"Here's what I found about your question:\n"]
    
    for chunk in chunks:
        lines.append(f"**{chunk['section']}** ({chunk['service']})")
        lines.append(chunk['text'])
        lines.append("")
    
    return "\n".join(lines)
