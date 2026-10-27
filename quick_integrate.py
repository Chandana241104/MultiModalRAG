"""Quick integration of all multi-modal components"""
from unified_embeddings import UnifiedEmbedder
from cross_modal_reranker import CrossModalReranker
from ingest_utils import extract_structured_tables, extract_chart_metadata

def enhance_existing_system():
    # 1. Process tables
    tables = extract_structured_tables("data/documents/sample.pdf")
    
    # 2. Process charts
    charts = extract_chart_metadata("data/documents/sample.pdf")
    
    # 3. Initialize unified embedder
    embedder = UnifiedEmbedder()
    
    # 4. Initialize reranker
    from cross_modal_reranker import CrossModalReranker

    # Change this:
    reranker = CrossModalReranker()
    
    # To this (if you want singleton):
    from cross_modal_reranker import get_reranker
    reranker = get_reranker()
    
    print(f"Extracted {len(tables)} tables")
    print(f"Extracted {len(charts)} charts")
    print("Multi-modal components ready")
    
    return embedder, reranker, tables, charts
