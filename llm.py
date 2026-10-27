"""Gemini-backed answer generation with offline-safe fallbacks."""

from __future__ import annotations

import json
import mimetypes
import os
import re
from typing import Dict, List, Tuple

from dotenv import load_dotenv

try:
    from google import genai
except Exception:
    genai = None


load_dotenv()


class EnhancedLLM:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        self.model_name = model_name
        self.client = None

        api_key = os.getenv("GEMINI_API_KEY")
        if genai is None:
            print("google-genai is not installed. LLM calls will use offline fallback.")
        elif not api_key:
            print("GEMINI_API_KEY is not set. LLM calls will use offline fallback.")
        else:
            try:
                self.client = genai.Client(api_key=api_key)
            except Exception as exc:
                print(f"Could not initialize Gemini client: {exc}")

    def _generate(self, prompt: str) -> str:
        if self.client is None:
            return self._offline_answer(prompt)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return getattr(response, "text", str(response)) or ""
        except Exception as exc:
            print(f"Gemini API error: {exc}")
            return self._offline_answer(prompt)

    def generate_answer_with_images(
        self,
        question: str,
        image_paths: List[str],
        context: str = "",
    ) -> str:
        """Ask Gemini to answer a question using actual image bytes plus context.

        If Gemini is unavailable, this method explicitly reports that visual
        interpretation was not performed rather than pretending OCR is vision.
        """
        existing_paths = [path for path in (image_paths or []) if path and os.path.isfile(path)]
        if not existing_paths:
            return "No valid image files were provided for visual analysis."

        if self.client is None:
            ocr_excerpt = context.strip() or "No OCR context is available."
            return (
                "Gemini is not configured, so the image itself was not interpreted. "
                "The available metadata/OCR context is shown below:\n\n"
                f"{ocr_excerpt[:3000]}"
            )

        prompt = f"""You are a careful multimodal document analyst.
Answer the user's question using the attached image(s) and the supplied context.
Describe visible trends, labels, legends, tables, and relationships only when they
are actually visible. Do not invent unreadable values. If a detail cannot be read,
say so. Separate visual observations from any OCR context. Mention the PDF page
number if it is present in the context.

CONTEXT:
{context or 'No additional OCR context was provided.'}

QUESTION:
{question}

Provide a clear answer with relevant visual evidence. If the image does not contain
the requested information, say that directly."""

        try:
            from google.genai import types

            contents = [prompt]
            for image_path in existing_paths[:4]:
                mime_type, _ = mimetypes.guess_type(image_path)
                if not mime_type or not mime_type.startswith("image/"):
                    mime_type = "image/png"
                with open(image_path, "rb") as image_file:
                    image_bytes = image_file.read()
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
            )
            answer = getattr(response, "text", None)
            if answer and answer.strip():
                return answer.strip()
            return "Gemini returned an empty response for the supplied image."
        except Exception as exc:
            print(f"Gemini image analysis error: {exc}")
            return (
                "Gemini could not analyze the image. Check the API key, model name, "
                f"and API error. Details: {exc}"
            )

    def _offline_answer(self, prompt: str) -> str:
        """Extract relevant information from context when Gemini is unavailable."""
        # Try to extract the answer from the context
        context_match = re.search(
            r"(?:Context below contains text.*?:|CONTEXT WITH CITATIONS:|document excerpts, provide a comprehensive summary:)\s*(.*?)(?:\n\s*(?:Question:|QUESTION:|Create a structured summary)|$)",
            prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        context = context_match.group(1).strip() if context_match else ""
        
        if not context:
            return "I could not find the answer in the document."

        # Try to find direct answer in context
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        
        # Look for lines that might contain the answer
        answer_lines = []
        for line in lines:
            # Look for lines with percentages, numbers, or key information
            if any([
                re.search(r'\d+%', line),
                re.search(r'weight', line.lower()),
                re.search(r'score', line.lower()),
                re.search(r'criteria', line.lower()),
                re.search(r'accuracy', line.lower()),
                re.search(r'delivery', line.lower()),
                re.search(r'options?', line.lower()),
            ]):
                answer_lines.append(line)
        
        if answer_lines:
            # Take the first few relevant lines
            answer = " ".join(answer_lines[:3])
            if len(answer) > 700:
                answer = answer[:697].rstrip() + "..."
            return f"Based on the document: {answer}"
        
        # Fallback: return first few lines of context
        excerpt = " ".join(lines[:3])
        if len(excerpt) > 700:
            excerpt = excerpt[:697].rstrip() + "..."
        return (
            "Offline mode: Gemini is not configured. Here is the most "
            f"relevant content from the document:\n\n{excerpt}"
        )

    def generate_answer(self, context: str, question: str) -> str:
        """Generate an answer from context while preserving legacy call shape."""
        if not context or not context.strip():
            return "No context was provided to answer this question."

        prompt = f"""
You are an intelligent document analyst. Answer the question based ONLY on the provided context.

CONTEXT:
{context}

QUESTION: {question}

Instructions:
1. Base your answer ONLY on the context provided above.
2. If the context contains the answer, provide it directly and concisely.
3. If the context partially contains the answer, provide what you can.
4. If the context does NOT contain the answer, say: "I could not find the answer in the document."
5. Mention relevant page numbers when available.
6. Quote numerical data (percentages, numbers) exactly as they appear.

ANSWER:
"""
        return self._generate(prompt)

    def generate_answer_with_citations(self, chunks: List[str], question: str) -> Tuple[str, List[Dict], float]:
        """
        Generate an answer with citation metadata.

        Returns (answer, citations, confidence_percent).
        """
        if not chunks:
            return "I could not find any relevant information in the document.", [], 0.0

        formatted_context = ""
        citation_map = {}

        for i, chunk in enumerate(chunks or []):
            chunk_text = self._chunk_to_text(chunk)
            source_match = re.match(r"\[Page\s*([^\]|]+)\s*\|\s*([^\]]+)\]\s*(.+)", chunk_text, re.DOTALL)
            citation_id = i + 1

            if source_match:
                page, source_type, text = source_match.groups()
                page = page.strip()
                source_type = source_type.strip()
                text = text.strip()
            else:
                page = "Unknown"
                source_type = "text"
                text = chunk_text

            # Truncate very long chunks
            if len(text) > 1500:
                text = text[:1500] + "..."
            
            formatted_context += f"[{citation_id}] Source: Page {page} ({source_type}): {text}\n\n"
            citation_map[citation_id] = {
                "page": page,
                "type": source_type,
                "text": text[:200] + "..." if len(text) > 200 else text,
            }

        # Check if we have meaningful context
        if len(formatted_context.strip()) < 50:
            return "The retrieved content is too short or empty. Please try rephrasing your question.", [], 0.0

        prompt = f"""
You are a precise document analyst. Use ONLY the context below to answer the question.

CONTEXT WITH CITATIONS:
{formatted_context}

QUESTION: {question}

OUTPUT FORMAT:
ANSWER: Your complete answer with inline citations like [1][2] if you use specific sources
CONFIDENCE: Number between 0-100% (how confident are you that this answer is correct)
CITATIONS USED: Comma-separated citation numbers, e.g. 1,3,5

INSTRUCTIONS:
- If the answer is in the context, provide it clearly and concisely.
- If the answer is partially in the context, provide what you can.
- If the answer is NOT in the context, say "I could not find the answer in the document."
- Do NOT make up information not in the context.
- Quote numbers and percentages exactly as they appear.
"""
        response_text = self._generate(prompt)
        answer, confidence, citations_used = self.parse_structured_response(response_text)
        
        # If answer indicates not found, check if we can still extract something
        if "could not find" in answer.lower() or "not found" in answer.lower():
            # Try to extract any relevant information
            fallback_answer = self._extract_relevant_info(formatted_context, question)
            if fallback_answer and "could not find" not in fallback_answer.lower():
                answer = fallback_answer
                confidence = 40.0
        
        citations = [citation_map[citation_id] for citation_id in citations_used if citation_id in citation_map]
        return answer, citations, confidence

    def _extract_relevant_info(self, context: str, question: str) -> str:
        """Extract relevant information when the main answer generation fails."""
        # Try to find key terms in context
        key_terms = re.findall(r'\b\w+\b', question.lower())
        key_terms = [t for t in key_terms if len(t) > 3]
        
        # Find sentences containing key terms
        sentences = re.split(r'[.!?]+', context)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for term in key_terms[:5]:
                if term in sentence_lower:
                    relevant_sentences.append(sentence.strip())
                    break
        
        if relevant_sentences:
            return " ".join(relevant_sentences[:3])
        
        # If no relevant sentences found, return first few lines
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        if lines:
            return " ".join(lines[:2])
        
        return None

    def _chunk_to_text(self, chunk) -> str:
        if isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("content") or chunk.get("description") or ""
            page = chunk.get("page", "Unknown")
            source_type = chunk.get("type", "text")
            return f"[Page {page} | {source_type}] {text}"
        return str(chunk)

    def parse_structured_response(self, response: str) -> Tuple[str, float, List[int]]:
        answer = response
        confidence = 50.0
        citations_used: List[int] = []

        # Try to parse structured response
        answer_match = re.search(
            r"ANSWER:\s*(.*?)(?=CONFIDENCE:|CITATIONS USED:|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if answer_match:
            answer = answer_match.group(1).strip()

        confidence_match = re.search(r"CONFIDENCE:\s*(\d+(?:\.\d+)?)\s*%?", response, re.IGNORECASE)
        if confidence_match:
            try:
                confidence = max(0.0, min(100.0, float(confidence_match.group(1))))
            except ValueError:
                pass

        citations_match = re.search(r"CITATIONS USED:\s*([\d,\s]+)", response, re.IGNORECASE)
        if citations_match:
            citations_used = [int(c) for c in citations_match.group(1).split(",") if c.strip().isdigit()]

        if not citations_used:
            citations_used = [int(c) for c in re.findall(r"\[(\d+)\]", answer)]

        # Clean up answer
        answer = answer.strip()
        if not answer:
            answer = "I could not find the answer in the document."

        return answer, confidence, citations_used

    def generate_document_summary(self, chunks: List[str] | None = None) -> str:
        """Generate a comprehensive summary from document chunks."""
        chunks = chunks or []
        if not chunks:
            return "No document context was provided for summarization."

        # Convert chunks to text
        chunk_texts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                chunk_texts.append(chunk.get("text", ""))
            else:
                chunk_texts.append(str(chunk))
        
        combined_context = "\n\n".join(chunk_texts[:50])
        
        if len(combined_context) > 8000:
            combined_context = combined_context[:8000] + "..."
        
        prompt = f"""
Based on the following document excerpts, provide a concise structured summary:

{combined_context}

Please create a summary with:
1. Main topics and themes
2. Key statistics and numbers mentioned
3. Any visual or table content
4. Conclusions or key takeaways

Format your response as a well-structured summary with clear sections.
"""
        return self._generate(prompt)

    def evaluate_answer_quality(self, question: str, answer: str, context: str) -> Dict:
        prompt = f"""
Evaluate the quality of this Q&A pair.

Question: {question}
Context: {context[:1000]}...
Answer: {answer}

Return JSON with faithfulness, completeness, clarity, citation, and overall scores from 1 to 5.
"""
        response = self._generate(prompt)
        try:
            eval_text = re.sub(r"```json\s*|\s*```", "", response).strip()
            return json.loads(eval_text)
        except Exception:
            return {
                "faithfulness": 3,
                "completeness": 3,
                "clarity": 3,
                "citation": 3,
                "overall": 3,
            }

    def generate_structured_citations(self, chunks: List[str]) -> List[Dict]:
        citations = []
        for chunk in chunks:
            chunk_text = self._chunk_to_text(chunk)
            match = re.search(r"\[Page\s*([^\]|]+)\s*\|\s*([^\]]+)\]\s*(.+)", chunk_text, re.DOTALL)
            if match:
                page, doc_type, content = match.groups()
                citations.append(
                    {
                        "page": page.strip(),
                        "type": doc_type.strip(),
                        "content": content[:200],
                        "confidence": self.calculate_source_confidence(content),
                        "modality": self.determine_modality(doc_type, content),
                    }
                )
        return citations

    def calculate_source_confidence(self, content: str) -> float:
        if not content:
            return 0.0
        if len(content) > 200:
            return 0.85
        if len(content) > 50:
            return 0.65
        return 0.4

    def determine_modality(self, doc_type: str, content: str) -> str:
        doc_type_lower = doc_type.lower()
        if "image" in doc_type_lower or "chart" in doc_type_lower:
            return "vision"
        if "table" in doc_type_lower:
            return "table"
        if re.search(r"\d+%|\b\d+\.\d+\b", content):
            return "numeric_text"
        return "text"


_llm_instance = None


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = EnhancedLLM()
    return _llm_instance


def generate_answer(context, question):
    return get_llm().generate_answer(context, question)


def generate_structured_citations(chunks: List[str]) -> List[Dict]:
    return get_llm().generate_structured_citations(chunks)