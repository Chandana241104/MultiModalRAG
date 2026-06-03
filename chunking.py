def chunk_text(text, chunk_size=500):
    """Chunk text with overlap for better context coverage."""
    words = text.split()
    chunks = []
    overlap = 100  # words

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks
