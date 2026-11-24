"""Handle actions sent by the embedded Folio web interface."""

from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

from evaluator import evaluate_answer_similarity


VALID_PAGES = {"ask", "activity", "evaluation", "document", "visuals"}


def _set_notice(session_state, kind: str, message: str) -> None:
    session_state.folio_notice = {"kind": kind, "message": message}


def _retrieve(retriever, query: str, top_k: int = 5) -> list[str]:
    return retriever.retrieve_answer_chunks(query, top_k=top_k)


def _chunk_page(chunk: str) -> str | None:
    match = re.match(r"\s*\[Page\s*([^\]|]+)", str(chunk), re.IGNORECASE)
    return match.group(1).strip() if match else None


def _ask(event, session_state, retriever, llm, settings) -> None:
    question = str(event.get("question", "")).strip()
    if not question:
        _set_notice(session_state, "error", "Enter a question first.")
        return
    if not session_state.active_document:
        _set_notice(session_state, "error", "Upload a PDF before asking questions.")
        return

    start_time = datetime.now()
    chunks = retriever.retrieve_answer_chunks(
        question,
        top_k=settings["top_k"],
        rerank=settings["rerank"],
        deduplicate=settings["deduplicate"],
    )
    retrieval_time = (datetime.now() - start_time).total_seconds()
    session_state.retrieval_metrics.append({
        "question": question,
        "timestamp": datetime.now().isoformat(),
        "chunks_retrieved": len(chunks),
        "retrieval_time": retrieval_time,
        "top_k": settings["top_k"],
    })
    session_state.folio_last_context = chunks

    citations = []
    confidence = None
    generation_time = 0
    if chunks:
        generation_start = datetime.now()
        if settings["show_citations"]:
            answer, citations, confidence = llm.generate_answer_with_citations(
                chunks, question
            )
        else:
            answer = llm.generate_answer("\n\n".join(chunks[:10]), question)
        generation_time = (datetime.now() - generation_start).total_seconds()
    else:
        answer = "No relevant content was retrieved, so an answer could not be generated."

    source_pages = list(dict.fromkeys(
        page for page in (_chunk_page(chunk) for chunk in chunks) if page
    ))
    source_pages.extend(
        page for page in (
            str(citation.get("page", "")).strip()
            for citation in citations
        )
        if page and page not in source_pages
    )
    session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "answer_generated": bool(chunks),
        "timestamp": datetime.now().isoformat(),
        "chunks_used": len(chunks),
        "retrieved_chunks": chunks,
        "source_pages": source_pages,
        "confidence": confidence,
        "citations": citations,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "evaluation": {},
    })
    session_state.folio_last_retrieval = {
        "chunks": len(chunks),
        "time": retrieval_time,
    }
    if not chunks:
        _set_notice(
            session_state,
            "error",
            "No relevant content was found. The question was saved for evaluation.",
        )
        return

    if (
        confidence is not None
        and confidence / 100 < settings.get("confidence_threshold", 0)
    ):
        _set_notice(
            session_state,
            "warning",
            f"Low confidence answer ({confidence:.1f}%). Verify it against the cited sources.",
        )


def _evaluate_user_query(event, session_state) -> None:
    try:
        query_index = int(event.get("query_index", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid user query selection.") from exc

    chats = session_state.chat_history
    if query_index < 0 or query_index >= len(chats):
        raise ValueError("The selected user query is no longer available.")

    chat = chats[query_index]
    chunks = chat.get("retrieved_chunks", [])
    reference_answer = str(event.get("reference_answer", "")).strip()
    expected_page_text = str(event.get("expected_pages", "")).strip()
    expected_pages = set()
    if expected_page_text:
        page_tokens = re.split(r"[,;\s]+", expected_page_text)
        if any(not token.isdigit() or int(token) < 1 for token in page_tokens):
            _set_notice(
                session_state,
                "error",
                "Expected pages must be positive page numbers separated by commas.",
            )
            return
        expected_pages = {int(token) for token in page_tokens}

    raw_indices = event.get("relevant_chunk_indices", [])
    if not isinstance(raw_indices, list):
        raise ValueError("Relevant chunk selections must be a list.")
    try:
        relevant_indices = sorted({int(index) for index in raw_indices})
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid relevant chunk selection.") from exc
    if any(index < 0 or index >= len(chunks) for index in relevant_indices):
        raise ValueError("A selected relevant chunk is no longer available.")

    relevance_reviewed = bool(event.get("relevance_reviewed", False))
    retrieval_precision = None
    if relevance_reviewed and chunks:
        retrieval_precision = len(relevant_indices) / len(chunks) * 100

    page_precision = None
    if expected_pages and chunks:
        retrieved_pages = [_chunk_page(chunk) for chunk in chunks]
        relevant_pages = sum(
            1 for page in retrieved_pages
            if page is not None and page.isdigit() and int(page) in expected_pages
        )
        page_precision = relevant_pages / len(chunks) * 100

    answer_generated = chat.get("answer_generated", bool(chat.get("answer")))
    answer_metrics = (
        evaluate_answer_similarity(chat.get("answer", ""), reference_answer)
        if reference_answer and answer_generated
        else None
    )
    chat["evaluation"] = {
        "reference_answer": reference_answer,
        "expected_pages": sorted(expected_pages),
        "relevance_reviewed": relevance_reviewed,
        "relevant_chunk_indices": relevant_indices,
        "retrieval_precision": retrieval_precision,
        "page_precision": page_precision,
        "answer_metrics": answer_metrics,
    }

    if not answer_generated:
        _set_notice(
            session_state,
            "warning",
            "Retrieval evaluation saved. Answer scoring is unavailable because no answer was generated.",
        )
    elif answer_metrics is None:
        _set_notice(
            session_state,
            "warning",
            "Retrieval evaluation saved. Answer evaluation is pending a reference answer.",
        )
    else:
        _set_notice(session_state, "success", "User query evaluation saved.")


def _run_evaluation(session_state, retriever, llm) -> None:
    from evaluator import MultiModalEvaluator

    evaluator = MultiModalEvaluator()
    session_state.evaluation_results = evaluator.run_full_evaluation(
        retriever.retrieve_answer_chunks,
        llm.generate_answer,
    )
    _set_notice(session_state, "success", "Evaluation completed.")


def _load_sample_results(session_state) -> None:
    session_state.evaluation_results = pd.DataFrame({
        "question": [
            "What are the main findings?",
            "Which delivery options are mentioned?",
            "What is the key recommendation?",
        ],
        "modality": ["text", "text", "table"],
        "retrieval_score": [85, 92, 78],
        "answer_similarity": [90, 88, 82],
        "exact_match": [True, True, False],
    })
    _set_notice(session_state, "success", "Sample evaluation results loaded.")


def _full_summary(session_state, retriever, llm) -> None:
    queries = [
        "main purpose",
        "objectives",
        "key findings",
        "recommendations",
        "deliverables",
        "evaluation criteria",
        "conclusions",
        "important statistics",
        "tables and figures",
        "key takeaways",
    ]
    chunks = []
    for query in queries:
        chunks.extend(_retrieve(retriever, query, top_k=2))

    unique_chunks = list(dict.fromkeys(chunks))
    if not unique_chunks:
        _set_notice(session_state, "error", "No document content was found to summarize.")
        return

    session_state.folio_outputs["full_summary"] = {
        "text": llm.generate_document_summary(unique_chunks[:15]),
        "sources": unique_chunks[:8],
    }


def _quick_summary(action: str, session_state, retriever, llm) -> None:
    options = {
        "quick_overview": (
            "main purpose and key topics",
            "Give a concise overview of this document.",
            "Quick overview",
        ),
        "evaluation_summary": (
            "evaluation criteria and weights",
            "Summarize the evaluation criteria and their weights.",
            "Evaluation criteria",
        ),
        "features_summary": (
            "expected features and capabilities",
            "List and briefly describe the expected features.",
            "Key features",
        ),
        "deliverables_summary": (
            "deliverables and required outputs",
            "What are the deliverables?",
            "Deliverables",
        ),
    }
    query, question, label = options[action]
    chunks = _retrieve(retriever, query, top_k=5)
    if not chunks:
        _set_notice(session_state, "error", "No matching content was found in the PDF.")
        return

    answer = llm.generate_answer("\n".join(chunks[:3]), question)
    session_state.folio_outputs[action] = {
        "title": label,
        "text": answer,
    }


def _extract_numbers(session_state, retriever) -> None:
    queries = [
        "percentage", "weight", "score", "metric", "criteria",
        "dates", "deadlines", "duration", "totals", "amounts",
    ]
    chunks = []
    for query in queries:
        chunks.extend(_retrieve(retriever, query, top_k=2))

    references = []
    seen = set()
    for chunk in chunks:
        percentages = re.findall(r"(\d+%)", chunk)
        numbers = re.findall(r"\b(\d+)\b", chunk)
        if not percentages and not numbers:
            continue
        preview = chunk[:250] + ("..." if len(chunk) > 250 else "")
        fingerprint = preview[:50]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        references.append({
            "text": preview,
            "percentages": percentages,
            "numbers": [number for number in numbers if int(number) > 9],
        })

    session_state.folio_outputs["numbers"] = references


def _analyze_structure(session_state, retriever) -> None:
    queries = ["sections", "headers", "title", "heading", "contents"]
    chunks = []
    for query in queries:
        chunks.extend(_retrieve(retriever, query, top_k=2))

    headers = set()
    for chunk in chunks:
        for line in chunk.splitlines():
            candidate = line.strip()
            if (
                candidate.startswith("#")
                or (len(candidate) < 100 and candidate.isalpha() and candidate.isupper())
                or "=====" in candidate
            ):
                if candidate:
                    headers.add(candidate[:100])

    session_state.folio_outputs["structure"] = sorted(headers)[:20]


def _visual_search(event, session_state) -> None:
    cross_modal = session_state.get("cross_modal_retriever")
    if cross_modal is None:
        _set_notice(
            session_state,
            "error",
            "Visual search needs an indexed PDF and enabled CLIP embeddings.",
        )
        return

    query = str(event.get("query", "")).strip()
    if not query:
        _set_notice(session_state, "error", "Enter a visual search query.")
        return

    session_state.visual_results = cross_modal.retrieve_visual_content(
        query,
        top_k=max(1, min(int(event.get("top_k", 3)), 10)),
    )
    session_state.visual_query_submitted = query
    session_state.visual_analysis = None


def _analyze_image(event, session_state, llm) -> None:
    results = session_state.get("visual_results", [])
    index = int(event.get("index", -1))
    question = str(event.get("question", "")).strip()
    if index < 0 or index >= len(results):
        _set_notice(session_state, "error", "Choose a valid search result first.")
        return
    if not question:
        _set_notice(session_state, "error", "Enter a question about the selected image.")
        return

    selected = results[index]
    context = (
        f"This image was extracted from page {selected.get('page', 'Unknown')} "
        "of the source PDF.\n"
        f"OCR text associated with the image:\n{selected.get('ocr_text', '')}\n"
        f"CLIP retrieval score: {selected.get('score', 0.0):.4f}"
    )
    session_state.visual_analysis = llm.generate_answer_with_images(
        question=question,
        image_paths=[selected["image_path"]],
        context=context,
    )


def handle_ui_event(event, session_state, retriever, llm, settings) -> None:
    """Apply one UI event to session state and the existing RAG services."""
    action = event.get("action")
    session_state.folio_notice = None

    if action == "navigate":
        page = event.get("page")
        if page not in VALID_PAGES:
            raise ValueError(f"Unknown UI page: {page}")
        session_state.folio_page = page
        return

    if action == "ask":
        _ask(event, session_state, retriever, llm, settings)
    elif action == "clear_history":
        session_state.chat_history = []
        session_state.retrieval_metrics = []
        session_state.folio_last_retrieval = None
        session_state.folio_last_context = []
    elif action == "run_evaluation":
        _run_evaluation(session_state, retriever, llm)
    elif action == "evaluate_user_query":
        _evaluate_user_query(event, session_state)
    elif action == "load_sample_results":
        _load_sample_results(session_state)
    elif action == "full_summary":
        _full_summary(session_state, retriever, llm)
    elif action in {
        "quick_overview",
        "evaluation_summary",
        "features_summary",
        "deliverables_summary",
    }:
        _quick_summary(action, session_state, retriever, llm)
    elif action == "extract_numbers":
        _extract_numbers(session_state, retriever)
    elif action == "analyze_structure":
        _analyze_structure(session_state, retriever)
    elif action == "visual_search":
        _visual_search(event, session_state)
    elif action == "analyze_image":
        _analyze_image(event, session_state, llm)
    else:
        raise ValueError(f"Unknown UI action: {action}")
