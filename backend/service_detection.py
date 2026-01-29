from sentence_transformers import SentenceTransformer
import numpy as np

SERVICE_DESCRIPTIONS = {
    "ration_card": """
    Kerala ration card services including eligibility criteria,
    required documents, card types, online application through
    Civil Supplies portal, Akshaya centre offline process,
    fees, timelines, corrections, and member changes.
    """,

    "birth_certificate": """
    Kerala birth certificate registration including eligibility,
    registration timelines, required documents for hospital and
    home births, online portals like K-SMART and ILGMS,
    offline registration at local bodies, late registration,
    corrections, duplicates, and special cases.
    """,

    "unemployment_allowance": """
    Kerala unemployment allowance schemes including Unemployment
    Allowance Scheme and MGNREGA unemployment allowance,
    eligibility conditions, required documents, application
    process through local bodies, benefit rules, appeals,
    and legal framework.
    """
}


_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

SERVICE_EMBEDDINGS = {
    service: _model.encode(desc, normalize_embeddings=True)
    for service, desc in SERVICE_DESCRIPTIONS.items()
}


def detect_service(query: str) -> str:
    """
    Detect the most relevant government service for a query
    using embedding similarity.
    """
    query_vec = _model.encode(query, normalize_embeddings=True)

    best_service = None
    best_score = -1.0

    for service, service_vec in SERVICE_EMBEDDINGS.items():
        score = float(np.dot(query_vec, service_vec))
        if score > best_score:
            best_score = score
            best_service = service

    return best_service
