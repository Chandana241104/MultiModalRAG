# llm.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
import re
from typing import List, Dict, Tuple
import json

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class EnhancedLLM:
    def __init__(self, model_name: str = 'gemini-2.5-flash'):
        self.model = genai.GenerativeModel(model_name)
        self.conversation_history = []
    
    def generate_answer(self, context: str, question: str) -> str:
        """
        Generate answer from context with improved prompting.
        Maintains backward compatibility.
        """
        try:
            # Robust Prompting with citation awareness
            prompt = f"""
            You are an intelligent document analyst. 
            
            Context below contains text from a PDF with source citations in format [Page X | Type]:
            {context}
            
            Question: {question}
            
            Instructions:
            1. Base your answer ONLY on the provided context
            2. If the context has source markers like [Page X | Type], mention the relevant page numbers
            3. For numerical data (percentages, dates, scores), quote them exactly
            4. If multiple sources mention the same fact, indicate all relevant sources
            5. If you cannot find the answer, say "I could not find the answer in the document."
            6. Structure your answer clearly and cite sources naturally
            
            Format your answer with:
            - Clear answer first
            - Supporting evidence from context
            - Source references when applicable
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ AI Error: {str(e)}"
    
    def generate_answer_with_citations(self, chunks: List[str], question: str) -> Tuple[str, List[Dict], float]:
        """
        Enhanced answer generation with structured citations and confidence scoring.
        Returns: (answer, citations, confidence_score)
        """
        try:
            # Prepare context with explicit citation markers
            formatted_context = ""
            citation_map = {}
            
            for i, chunk in enumerate(chunks):
                # Extract source info from chunk format [Page X | Type] text
                source_match = re.match(r'\[Page\s*(\d+)\s*\|\s*(\w+)\]\s*(.+)', chunk)
                if source_match:
                    page, source_type, text = source_match.groups()
                    citation_id = i + 1
                    formatted_context += f"[{citation_id}] Source: Page {page} ({source_type}): {text}\n\n"
                    citation_map[citation_id] = {
                        'page': page,
                        'type': source_type,
                        'text': text[:200] + "..." if len(text) > 200 else text
                    }
                else:
                    # Fallback for chunks without proper formatting
                    citation_id = i + 1
                    formatted_context += f"[{citation_id}] Source: {chunk[:150]}...\n\n"
                    citation_map[citation_id] = {
                        'page': 'Unknown',
                        'type': 'text',
                        'text': chunk[:200] + "..." if len(chunk) > 200 else chunk
                    }
            
            # Enhanced prompt for structured output
            prompt = f"""
            You are a precise document analyst. Use ONLY the context below to answer the question.
            
            CONTEXT WITH CITATIONS:
            {formatted_context}
            
            QUESTION: {question}
            
            CRITICAL INSTRUCTIONS:
            1. Answer must be based SOLELY on the provided context
            2. Include citation markers like [1], [2] after relevant statements
            3. If context doesn't contain answer, admit it and provide confidence 0%
            4. Estimate your confidence in the answer (0-100%)
            5. List all citations used at the end
            
            OUTPUT FORMAT (STRICT):
            ANSWER: [Your complete answer with inline citations like [1][2]]
            CONFIDENCE: [Number between 0-100]%
            CITATIONS USED: [Comma-separated citation numbers, e.g., 1,3,5]
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Parse structured response
            answer, confidence, citations_used = self.parse_structured_response(response_text)
            
            # Get citation details
            citations = []
            for cit_id in citations_used:
                if cit_id in citation_map:
                    citations.append(citation_map[cit_id])
            
            return answer, citations, confidence
            
        except Exception as e:
            return f"Error: {str(e)}", [], 0.0
    
    def parse_structured_response(self, response: str) -> Tuple[str, float, List[int]]:
        """Parse the structured LLM response"""
        
        # Initialize defaults
        answer = response
        confidence = 50.0  # Default medium confidence
        citations_used = []
        
        # Try to extract ANSWER section
        answer_match = re.search(r'ANSWER:\s*(.*?)(?=CONFIDENCE:|CITATIONS USED:|$)', 
                                response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
        
        # Try to extract CONFIDENCE
        confidence_match = re.search(r'CONFIDENCE:\s*(\d+\.?\d*)%', response, re.IGNORECASE)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
            except:
                pass
        
        # Try to extract CITATIONS USED
        citations_match = re.search(r'CITATIONS USED:\s*([\d,\s]+)', response, re.IGNORECASE)
        if citations_match:
            citations_str = citations_match.group(1)
            try:
                citations_used = [int(c.strip()) for c in citations_str.split(',') if c.strip().isdigit()]
            except:
                pass
        
        # If no structured format found, extract citations from answer text
        if not citations_used:
            citations_in_text = re.findall(r'\[(\d+)\]', answer)
            citations_used = [int(c) for c in citations_in_text if c.isdigit()]
        
        return answer, confidence, citations_used
    
    def generate_document_summary(self, chunks: List[str]) -> str:
        """Generate a comprehensive summary of the document"""
        try:
            # Combine chunks with source info
            combined_context = "\n\n".join(chunks[:50])  # Limit for token count
            
            prompt = f"""
            Based on the following document excerpts, provide a comprehensive summary:
            
            {combined_context}
            
            Create a structured summary including:
            1. Main topics and themes
            2. Key statistics and figures
            3. Important tables or charts mentioned
            4. Conclusions or recommendations
            5. Document structure and organization
            
            Format as a clear, well-structured report.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def evaluate_answer_quality(self, question: str, answer: str, context: str) -> Dict:
        """Self-evaluate the quality of the generated answer"""
        try:
            prompt = f"""
            Evaluate the quality of this Q&A pair:
            
            Question: {question}
            Context: {context[:1000]}...
            Answer: {answer}
            
            Rate on a scale of 1-5 for:
            1. Faithfulness to context (1=hallucinated, 5=perfectly grounded)
            2. Completeness (1=incomplete, 5=fully answered)
            3. Clarity (1=confusing, 5=very clear)
            4. Citation quality (1=no citations, 5=perfect citations)
            
            Return as JSON: {{"faithfulness": X, "completeness": X, "clarity": X, "citation": X, "overall": X}}
            Calculate overall as average of the four scores.
            """
            
            response = self.model.generate_content(prompt)
            
            # Try to parse JSON
            try:
                eval_text = response.text.strip()
                # Remove markdown code blocks if present
                eval_text = re.sub(r'```json\s*|\s*```', '', eval_text)
                evaluation = json.loads(eval_text)
                return evaluation
            except:
                # Fallback to simple scoring
                return {
                    "faithfulness": 3,
                    "completeness": 3,
                    "clarity": 3,
                    "citation": 3,
                    "overall": 3
                }
                
        except Exception as e:
            return {"error": str(e), "overall": 0}

# Initialize singleton instance
_llm_instance = None

def get_llm():
    """Get singleton LLM instance"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = EnhancedLLM()
    return _llm_instance

# Backward compatibility function
def generate_answer(context, question):
    """Legacy function for backward compatibility"""
    llm = get_llm()
    return llm.generate_answer(context, question)
def generate_structured_citations(self, chunks: List[str]) -> List[Dict]:
    """Extract structured source information from chunks"""
    citations = []
    
    for chunk in chunks:
        # Parse [Page X | Type] format
        import re
        match = re.search(r'\[Page\s*(\d+)\s*\|\s*(\w+)\]\s*(.+)', chunk)
        if match:
            page, doc_type, content = match.groups()
            citations.append({
                "page": int(page),
                "type": doc_type,
                "content": content[:200],
                "confidence": self.calculate_source_confidence(content),
                "modality": self.determine_modality(doc_type, content)
            })
    
    return citations