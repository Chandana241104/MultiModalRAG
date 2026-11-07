# MultiModalRAG

A multi-modal Retrieval-Augmented Generation (RAG) system for PDF-based document intelligence. The project combines text retrieval, OCR, image extraction, and vision-aware search to answer questions about complex documents using local vector indexing and an LLM backend.

This repository is a full-stack demo for querying multi-modal document content through a Streamlit interface. It supports:

- text retrieval from document chunks
- image and chart extraction from PDFs
- OCR-based visual interpretation
- cross-modal reranking and deduplication
- answer generation with Gemini and offline fallback
- evaluation dashboards for retrieval and answer quality

## Project Overview

The system is designed to work with multi-page PDFs and structured document repositories. It extracts both textual and visual information, indexes them in a vector store, and retrieves the most relevant evidence for a user question before generating a grounded answer.

The app is built around a Streamlit front end and a Python backend that handles ingestion, embedding, retrieval, reranking, and answer generation.

## Key Features

- PDF ingestion and chunking
- semantic text retrieval with embeddings
- visual indexing using CLIP-style image embeddings
- OCR extraction for charts and images
- hybrid reranking of retrieved results
- LLM-based answer generation using Gemini
- fallback offline answer behavior when no API key is available
- analytics and evaluation panels inside the UI

## Repository Structure

- `app.py` — main Streamlit application and UI
- `retriever.py` — enhanced retrieval pipeline with deduplication and reranking
- `vector_store.py` — vector database querying utilities
- `embeddings.py` — text embedding generation
- `vision_embeddings.py` — image embedding support
- `vision_processor.py` — vision preprocessing utilities
- `cross_modal_retriever.py` — retrieval across text and visual content
- `cross_modal_reranker.py` — reranking logic for multimodal results
- `ingest.py` — document ingestion pipeline
- `ingest_local.py` — local ingestion variant
- `ingest_utils.py` — helper utilities used during ingestion
- `llm.py` — Gemini-backed LLM wrapper and offline fallback logic
- `evaluator.py` — evaluation routines for retrieval and answer quality
- `build_vision_index.py` — creates a visual index from extracted images
- `setup.py` — environment setup utility
- `requirements.txt` — Python dependency list
- `data/` — ingestion data directory
- `report/` — project report or generated outputs
- `Multi-Modal-Rag.mp4` — demo video
- `Technical Report Multi Modal RAG.docx` — project technical report

## Architecture

### 1. Ingestion layer
The pipeline ingests documents, chunks them into segments, and prepares both text and visual evidence for indexing.

- PDFs are processed into text chunks
- images and charts are extracted
- OCR text is captured from visual content
- embeddings are generated for both modality types
- vector representations are stored in Pinecone or a local-compatible vector backend

### 2. Retrieval layer
The retriever:

- expands the user query
- embeds the query
- searches the vector store
- deduplicates near-duplicate matches
- applies relevance filtering
- reranks results by similarity and quality

This logic is implemented in `retriever.py` and utilities in `vector_store.py`.

### 3. Generation layer
`llm.py` wraps the Gemini API and provides methods to:

- answer questions from retrieved context
- answer questions using images
- generate document summaries
- parse structured responses with citations
- fallback to an offline answer when the API is unavailable

### 4. UI layer
`app.py` is a Streamlit application with:

- chat interface
- document summary tools
- analytics dashboard
- evaluation panel
- visual content analysis workspace

## Setup

### Requirements

- Python 3.8+
- Tesseract OCR installed on the system
- Access to Pinecone and Gemini credentials if you want full cloud-backed retrieval and generation

### Install dependencies

```bash
pip install -r requirements.txt
```

### Create environment variables

Create a `.env` file in the project root with your keys:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=multi-modal-rag
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=optional_openai_key

PDF_PATH=data/documents/sample.pdf
CACHE_DIR=cache

EMBEDDING_MODEL=paraphrase-MiniLM-L3-v2
VISION_MODEL=openai/clip-vit-base-patch32
CHUNK_SIZE=500
OVERLAP=100
```

### Install Tesseract

- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- Windows: install from the Tesseract GitHub releases or official project page

## Usage

### 1. Set up the project structure

```bash
python setup.py
```

This creates needed directories such as:

- `data/documents`
- `data/evaluation`
- `data/images`
- `logs`
- `cache`

### 2. Ingest the source documents

```bash
python ingest.py
```

This prepares the content for retrieval and indexing.

### 3. Run the application

```bash
streamlit run app.py
```

Open the local Streamlit URL in the browser to start asking questions about the document.

## Example Workflow

1. Upload or point the system to a PDF document
2. Ingest the document into the vector index
3. Ask a question in the Streamlit chat panel
4. The retriever fetches the most relevant chunks/images
5. The LLM answers using only the retrieved evidence
6. Results and metrics are displayed in the dashboard

## Evaluation Features

The repository includes experimental evaluation support for:

- retrieval accuracy
- answer similarity
- exact-match checks
- modality-based performance comparisons

These are surfaced in the `Evaluation` tab of the app and are driven by logic in `evaluator.py`.

## Visual Search

The project also supports visual retrieval and interpretation. It can:

- extract charts and images from PDFs
- perform image similarity search
- read OCR text from visual content
- ask Gemini to explain a selected image based on the image itself and OCR metadata

This functionality is implemented through `build_vision_index.py`, `vision_embeddings.py`, `cross_modal_retriever.py`, and the visual panel in `app.py`.

## Dependencies

The project uses libraries such as:

- Streamlit
- PyMuPDF
- Pillow
- sentence-transformers
- Pinecone client
- OpenCV / OCR libraries
- Transformers / Torch
- Plotly
- pandas / numpy
- python-dotenv
- Google GenAI

See `requirements.txt` for the full set.

## Notes

- The system is designed to operate with rich PDF document collections, especially documents with tables, charts, and images.
- If Gemini credentials are not configured, the app still includes offline generation logic, though it is limited compared to the cloud-backed mode.
- The repository contains both production-style app files and experimental utilities, so it is best suited for research and prototyping workflows.

## License

This repository does not currently include a license file. If you plan to distribute or publish the project publicly, consider adding an open-source license such as MIT or Apache 2.0.

## Summary

MultiModalRAG is a document intelligence system built around retrieval-augmented generation across text and visual content. It helps users ask natural-language questions about PDFs and other document data while grounding answers in retrieved evidence and multimodal context.
