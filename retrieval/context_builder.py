"""
Context builder: assembles retrieved code chunks into a structured
prompt for the LLM, including source attribution.
"""

from retrieval.retriever import RetrievedChunk


def build_context(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Builds the full LLM prompt from the question and retrieved chunks.

    The prompt structure:
    1. System instruction (role, constraints)
    2. Retrieved code context (numbered, with source attribution)
    3. User question
    """
    # Build the code context section
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = _format_source(chunk)
        context_parts.append(
            f"--- Source {i}: {source} ---\n"
            f"{chunk.content}\n"
        )

    context_text = "\n".join(context_parts)

    prompt = (
        f"You are a senior software engineer who is an expert on the DriveStream platform — "
        f"a Kafka-inspired event streaming system built with Java 21 that uses Google Drive "
        f"as the persistent storage backend.\n\n"
        f"Answer the user's question using ONLY the code context provided below. "
        f"Be specific, reference actual class names, method signatures, and explain the "
        f"code logic clearly. If the answer isn't in the provided context, say so honestly.\n\n"
        f"When referencing code, mention the file and class name so the user can find it.\n\n"
        f"═══ RELEVANT CODE CONTEXT ═══\n\n"
        f"{context_text}\n"
        f"═══ USER QUESTION ═══\n\n"
        f"{question}\n\n"
        f"═══ YOUR ANSWER ═══\n\n"
    )

    return prompt


def build_context_for_class_explanation(class_name: str, chunks: list[RetrievedChunk]) -> str:
    """Builds a prompt specifically for explaining a class in detail."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = _format_source(chunk)
        context_parts.append(
            f"--- Source {i}: {source} ---\n"
            f"{chunk.content}\n"
        )

    context_text = "\n".join(context_parts)

    prompt = (
        f"You are a senior software engineer explaining the DriveStream codebase.\n\n"
        f"Provide a comprehensive explanation of the `{class_name}` class, covering:\n"
        f"1. Its purpose and role in the architecture\n"
        f"2. Key fields and their meaning\n"
        f"3. Important methods and what they do\n"
        f"4. How it interacts with other classes\n"
        f"5. Any notable design patterns used\n\n"
        f"Use the code context below. Be specific and reference actual code.\n\n"
        f"═══ CODE CONTEXT ═══\n\n"
        f"{context_text}\n"
        f"═══ EXPLANATION ═══\n\n"
    )

    return prompt


def format_sources_for_response(chunks: list[RetrievedChunk]) -> list[dict]:
    """Formats retrieved chunks as source citations for the API response."""
    sources = []
    seen = set()
    for chunk in chunks:
        key = (chunk.file_path, chunk.class_name, chunk.method_name)
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "file": chunk.file_path,
            "class": chunk.class_name,
            "method": chunk.method_name,
            "type": chunk.chunk_type,
            "relevance": round(1.0 - chunk.score, 3),  # convert distance to similarity
        })
    return sources


def _format_source(chunk: RetrievedChunk) -> str:
    """Formats a chunk's source info for display in the prompt."""
    parts = []
    if chunk.file_path:
        # Show just the filename
        parts.append(chunk.file_path.split("/")[-1])
    if chunk.class_name:
        parts.append(chunk.class_name)
    if chunk.method_name:
        parts.append(f".{chunk.method_name}()")
    if chunk.docstring:
        parts.append(f'"{chunk.docstring[:80]}"')
    return " → ".join(parts) if parts else chunk.id
