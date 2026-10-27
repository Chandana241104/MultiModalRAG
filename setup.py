# setup.py
import os
import sys
import subprocess

def setup_environment():
    """Setup script for installation"""
    
    print("Setting up Multi-Modal RAG System")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8+ required")
        sys.exit(1)
    
    # Create directory structure
    directories = [
        "data/documents",
        "data/evaluation",
        "data/images",
        "logs",
        "cache"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")
    
    # Check for Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print("Tesseract found")
    except:
        print("Tesseract not found. Install:")
        print("  macOS: brew install tesseract")
        print("  Ubuntu: sudo apt install tesseract-ocr")
        print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
    
    # Create .env template
    env_template = """# API Keys
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=multi-modal-rag
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=optional_openai_key

# Paths
PDF_PATH=data/documents/sample.pdf
CACHE_DIR=cache

# Settings
EMBEDDING_MODEL=paraphrase-MiniLM-L3-v2
VISION_MODEL=openai/clip-vit-base-patch32
CHUNK_SIZE=500
OVERLAP=100
"""
    
    if not os.path.exists(".env"):
        with open(".env.template", "w") as f:
            f.write(env_template)
        print("Created .env.template - Fill with your API keys")
    
    print("\nSetup complete!")
    print("Next steps:")
    print("1. Fill in .env.template with your API keys")
    print("2. Rename to .env")
    print("3. Run: python ingest.py")
    print("4. Run: streamlit run app_enhanced.py")
