def detect_current_intent(chunks):
    """
    Infer current intent from the top retrieved chunk section.
    """
    if not chunks:
        return None

    section = chunks[0]["section"].lower()

    if "document" in section:
        return "documents"
    if "eligibility" in section:
        return "eligibility"
    if "process" in section or "apply" in section:
        return "process"
    if "time" in section or "timeline" in section:
        return "timeline"
    if "fee" in section:
        return "fees"
    if "correction" in section:
        return "correction"

    return None
