import base64
import io
import json
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from llm import get_llm
from retriever import EnhancedRetriever
from ui_actions import handle_ui_event


ROOT = Path(__file__).resolve().parent
folio_component = components.declare_component(
    "folio_web_ui",
    path=str(ROOT / "web_ui"),
)

st.set_page_config(
    page_title="Folio | Document Intelligence",
    layout="wide",
    page_icon="📚",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: #0c111b;
        color: #edf3fa;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1600px;
        padding: 1rem 1rem 2rem;
    }

    [data-testid="stSidebar"] {
        background: #111a27;
        border-right: 1px solid #27364a;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: #111a27;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #edf3fa;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #91a1b5;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background: #182334;
        border: 1px dashed #40516a;
        border-radius: 12px;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section * {
        color: #edf3fa;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] button,
    [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {
        background: #9af0c8;
        border: 1px solid #9af0c8;
        color: #10251e !important;
        font-weight: 700;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover,
    [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button:hover {
        background: #b6f7d8;
        border-color: #b6f7d8;
        color: #10251e !important;
    }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #dce6f1;
    }

    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
        border-radius: 10px;
    }

    [data-testid="stSidebar"] [data-testid="stForm"] {
        padding: 0.75rem;
        border: 1px solid #27364a;
        border-radius: 12px;
        background: #131b29;
    }

    [data-testid="stSidebar"] [data-testid="stAlert"] {
        border-radius: 10px;
    }

    [data-testid="stSidebar"] hr {
        border-color: #27364a;
    }

    iframe[title="folio_web_ui"] {
        border: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def init_llm():
    return get_llm()


def _initialize_session_state():
    defaults = {
        "rag_session_id": uuid.uuid4().hex,
        "active_document": None,
        "chat_history": [],
        "retrieval_metrics": [],
        "evaluation_results": None,
        "visual_results": [],
        "visual_analysis": None,
        "visual_query_submitted": "",
        "folio_page": "ask",
        "folio_outputs": {},
        "folio_notice": None,
        "folio_last_event_id": None,
        "folio_last_retrieval": None,
        "folio_last_context": [],
        "cross_modal_retriever": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _image_data_uri(image_path):
    from PIL import Image

    with Image.open(image_path) as image:
        image.thumbnail((1200, 1200))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _component_payload():
    document = st.session_state.active_document
    results = st.session_state.evaluation_results
    if isinstance(results, pd.DataFrame):
        evaluation_records = json.loads(results.to_json(orient="records"))
    else:
        evaluation_records = results or []

    visual_results = []
    for result in st.session_state.visual_results:
        component_result = {
            key: value
            for key, value in result.items()
            if key != "image_path"
        }
        image_path = result.get("image_path")
        if image_path:
            try:
                component_result["image_data"] = _image_data_uri(image_path)
            except (OSError, ValueError) as exc:
                component_result["image_error"] = f"Image preview unavailable: {exc}"
        visual_results.append(component_result)

    cross_modal = st.session_state.cross_modal_retriever
    vision_documents = getattr(cross_modal, "vision_documents", []) if cross_modal else []
    return {
        "page": st.session_state.folio_page,
        "document": (
            {
                "filename": document.get("filename"),
                "chunk_count": document.get("chunk_count", 0),
            }
            if document
            else None
        ),
        "chat_history": st.session_state.chat_history,
        "retrieval_metrics": st.session_state.retrieval_metrics,
        "last_retrieval": st.session_state.folio_last_retrieval,
        "evaluation_results": evaluation_records,
        "outputs": st.session_state.folio_outputs,
        "retrieved_context": st.session_state.folio_last_context,
        "visual_results": visual_results,
        "visual_analysis": st.session_state.visual_analysis,
        "visual_query": st.session_state.visual_query_submitted,
        "visual_ready": cross_modal is not None,
        "visual_count": len(vision_documents),
        "visual_inventory": [
            {
                "image_id": image.get("image_id", ""),
                "page": image.get("page", ""),
                "image_index": image.get("image_index", ""),
                "ocr_text": image.get("ocr_text", ""),
            }
            for image in vision_documents
        ],
        "notice": st.session_state.folio_notice,
        "show_confidence": st.session_state.show_confidence_setting,
        "show_debug": st.session_state.show_debug_setting,
    }


_initialize_session_state()
active_document = st.session_state.active_document
retriever = EnhancedRetriever(
    session_id=st.session_state.rag_session_id,
    document_id=active_document["document_id"] if active_document else None,
)
llm = init_llm()

with st.sidebar:
    st.markdown("## FOLIO")
    st.caption("Document intelligence workspace")
    st.subheader("Your source")
    st.caption("One active PDF per session. Uploading another replaces this one.")

    with st.form("pdf_upload_form", clear_on_submit=True):
        uploaded_pdf = st.file_uploader(
            "Upload a PDF to ask questions about",
            type=["pdf"],
        )
        upload_submitted = st.form_submit_button(
            "Upload and index PDF",
            use_container_width=True,
        )

    if upload_submitted:
        if uploaded_pdf is None:
            st.error("Choose a PDF file before uploading.")
        else:
            previous_document_id = (
                active_document["document_id"] if active_document else None
            )
            try:
                from upload_manager import ingest_uploaded_pdf

                with st.spinner("Extracting and indexing your PDF..."):
                    upload_result = ingest_uploaded_pdf(
                        uploaded_pdf.getvalue(),
                        session_id=st.session_state.rag_session_id,
                        filename=uploaded_pdf.name,
                        previous_document_id=previous_document_id,
                    )

                st.session_state.active_document = upload_result
                st.session_state.chat_history = []
                st.session_state.retrieval_metrics = []
                st.session_state.evaluation_results = None
                st.session_state.visual_results = []
                st.session_state.visual_analysis = None
                st.session_state.visual_query_submitted = ""
                st.session_state.folio_outputs = {}
                st.session_state.folio_last_retrieval = None
                st.session_state.folio_last_context = []
                st.session_state.folio_notice = None

                vision_error = None
                if upload_result["vision_indexed"]:
                    try:
                        from cross_modal_retriever import CrossModalRetriever

                        st.session_state.cross_modal_retriever = CrossModalRetriever(
                            data_dir=upload_result["asset_directory"]
                        )
                    except Exception as exc:
                        st.session_state.cross_modal_retriever = None
                        vision_error = f"Image index could not be loaded: {exc}"
                else:
                    st.session_state.cross_modal_retriever = None

                st.success(
                    f"Indexed {upload_result['filename']} "
                    f"({upload_result['chunk_count']} text chunks)."
                )
                if not upload_result["vision_indexed"] and upload_result["image_count"]:
                    st.warning(
                        "Image search is unavailable because CLIP embeddings are not "
                        "enabled or could not be loaded. Extracted image OCR is still searchable."
                    )
                if vision_error:
                    st.warning(vision_error)
                if upload_result["cleanup_warning"]:
                    st.warning(upload_result["cleanup_warning"])

                active_document = upload_result
                retriever.set_document_scope(upload_result["document_id"])
            except Exception as exc:
                st.error(f"PDF upload failed: {exc}")

    if active_document:
        st.caption(f"Active PDF: {active_document['filename']}")
    else:
        st.info("Upload a PDF to enable document questions and analysis.")

    st.divider()
    st.subheader("Retrieval settings")
    top_k = st.slider(
        "Number of chunks",
        5,
        100,
        20,
        help="How many text chunks to retrieve from the document.",
    )
    left, right = st.columns(2)
    with left:
        rerank_enabled = st.checkbox("Rerank", True)
    with right:
        deduplicate_enabled = st.checkbox("Deduplicate", True)

    st.subheader("Display settings")
    show_citations = st.checkbox("Show citations", True)
    show_confidence = st.checkbox("Show confidence", True)
    show_debug = st.checkbox("Show retrieved context", False)
    st.session_state.show_confidence_setting = show_confidence
    st.session_state.show_debug_setting = show_debug
    with st.expander("Advanced settings"):
        confidence_threshold = st.slider(
            "Confidence threshold",
            0.0,
            1.0,
            0.3,
            0.05,
            help="Show a warning when an answer is below this confidence.",
        )
        max_context_length = st.number_input(
            "Max context length",
            1000,
            10000,
            4000,
        )

    st.divider()
    st.caption("Answers · Gemini")
    st.caption("Text vectors · MiniLM")
    st.caption("Scope · Private to this session")

component_event = folio_component(
    data=_component_payload(),
    key="folio_web_ui",
    default=None,
)

if isinstance(component_event, dict):
    event_id = component_event.get("event_id")
    if event_id and event_id != st.session_state.folio_last_event_id:
        st.session_state.folio_last_event_id = event_id
        settings = {
            "top_k": top_k,
            "rerank": rerank_enabled,
            "deduplicate": deduplicate_enabled,
            "show_citations": show_citations,
            "show_confidence": show_confidence,
            "confidence_threshold": confidence_threshold,
            "max_context_length": max_context_length,
        }
        try:
            with st.spinner("Working with your document..."):
                handle_ui_event(
                    component_event,
                    st.session_state,
                    retriever,
                    llm,
                    settings,
                )
        except Exception as exc:
            st.session_state.folio_notice = {
                "kind": "error",
                "message": f"Could not complete the requested action: {exc}",
            }
        st.rerun()
