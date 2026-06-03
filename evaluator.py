# evaluator.py - Complete evaluation module for Multi-Modal RAG
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st
import plotly.express as px
from datetime import datetime
import os

class MultiModalEvaluator:
    def __init__(self, test_queries_path: str = "data/evaluation/queries.json"):
        """Initialize evaluator with test queries"""
        self.test_queries = self.load_test_queries(test_queries_path)
        self.results = []
        
    def load_test_queries(self, path: str) -> Dict:
        """Load benchmark queries for evaluation"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Sample evaluation queries based on your document
        sample_queries = {
            "text_queries": [
                {
                    "question": "What is the weight of Accuracy & Faithfulness?",
                    "expected_answer": "25%",
                    "modality": "text",
                    "expected_page": 1,
                    "difficulty": "easy"
                },
                {
                    "question": "What are the delivery options?",
                    "expected_answer": "24-hour and 48-hour challenges",
                    "modality": "text",
                    "expected_page": 2,
                    "difficulty": "easy"
                },
                {
                    "question": "What is the project about?",
                    "expected_answer": "multi-modal RAG system for document intelligence",
                    "modality": "text",
                    "expected_page": 1,
                    "difficulty": "medium"
                }
            ],
            "table_queries": [
                {
                    "question": "What is the weight for Code Quality?",
                    "expected_answer": "10%",
                    "modality": "table",
                    "expected_page": 2,
                    "difficulty": "easy"
                },
                {
                    "question": "What is multi-modal coverage weight?",
                    "expected_answer": "20%",
                    "modality": "table",
                    "expected_page": 2,
                    "difficulty": "easy"
                },
                {
                    "question": "List all evaluation criteria with weights",
                    "expected_answer": "Accuracy (25%), Multi-modal (20%), System Design (20%), Innovation (15%), Code Quality (10%), Presentation (10%)",
                    "modality": "table",
                    "expected_page": 2,
                    "difficulty": "hard"
                }
            ]
        }
        
        # Save to file
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(sample_queries, f, indent=2, ensure_ascii=False)
            print(f"✅ Created evaluation queries at: {path}")
        except Exception as e:
            print(f"⚠️  Could not save queries: {e}")
        
        return sample_queries
    
    def evaluate_retrieval(self, retrieved_chunks: List[str], 
                          expected_page: int, modality: str) -> Dict:
        """Evaluate retrieval performance"""
        metrics = {
            "retrieved_count": len(retrieved_chunks),
            "correct_modality": 0,
            "correct_page": 0,
            "precision": 0,
            "page_precision": 0
        }
        
        for chunk in retrieved_chunks:
            # Check if chunk contains the right modality
            if modality in chunk.lower():
                metrics["correct_modality"] += 1
            
            # Check if chunk contains the right page
            if (f"page {expected_page}" in chunk.lower() or 
                f"[page {expected_page}" in chunk.lower() or
                f"| page {expected_page}" in chunk.lower()):
                metrics["correct_page"] += 1
        
        # Calculate precision
        if retrieved_chunks:
            metrics["precision"] = metrics["correct_modality"] / len(retrieved_chunks)
            if expected_page > 0:
                metrics["page_precision"] = metrics["correct_page"] / len(retrieved_chunks)
        
        return metrics
    
    def evaluate_answer_similarity(self, predicted: str, expected: str) -> Dict:
        """Evaluate answer quality using text similarity"""
        if not predicted or not expected:
            return {
                "exact_match": False,
                "key_numbers_matched": 0,
                "jaccard_similarity": 0,
                "combined_score": 0
            }
        
        predicted_lower = predicted.lower()
        expected_lower = expected.lower()
        
        # Check for exact match of key terms
        exact_match = expected_lower in predicted_lower
        
        # Check for partial matches (key numbers like percentages)
        key_numbers = [word for word in expected.split() if '%' in word or word.isdigit()]
        number_matches = 0
        for num in key_numbers:
            if num in predicted:
                number_matches += 1
        
        # Calculate Jaccard similarity
        expected_words = set(expected_lower.split())
        predicted_words = set(predicted_lower.split())
        
        if expected_words and predicted_words:
            intersection = expected_words.intersection(predicted_words)
            union = expected_words.union(predicted_words)
            jaccard_similarity = len(intersection) / len(union) if union else 0
        else:
            jaccard_similarity = 0
        
        # Calculate combined score
        key_numbers_score = number_matches / len(key_numbers) if key_numbers else 0
        combined_score = (exact_match * 0.4 + key_numbers_score * 0.3 + jaccard_similarity * 0.3) * 100
        
        return {
            "exact_match": exact_match,
            "key_numbers_matched": key_numbers_score,
            "jaccard_similarity": jaccard_similarity,
            "combined_score": combined_score
        }
    
    def run_single_evaluation(self, question: Dict, retriever_func, answer_func) -> Dict:
        """Run evaluation for a single question"""
        try:
            # Step 1: Retrieve chunks
            retrieved_chunks = retriever_func(question["question"], top_k=10)
            
            # Step 2: Evaluate retrieval
            retrieval_metrics = self.evaluate_retrieval(
                retrieved_chunks, 
                question.get("expected_page", 0),
                question.get("modality", "text")
            )
            
            # Step 3: Generate answer
            context = "\n".join(retrieved_chunks[:3]) if retrieved_chunks else ""
            answer = answer_func(context, question["question"]) if context else "No context found"
            
            # Step 4: Evaluate answer
            answer_metrics = self.evaluate_answer_similarity(answer, question["expected_answer"])
            
            # Combine results
            result = {
                "question": question["question"],
                "modality": question.get("modality", "text"),
                "difficulty": question.get("difficulty", "medium"),
                "expected_answer": question["expected_answer"],
                "retrieved_answer": answer[:200] + "..." if len(answer) > 200 else answer,
                "retrieved_count": retrieval_metrics["retrieved_count"],
                "retrieval_precision": round(retrieval_metrics["precision"] * 100, 2),
                "answer_similarity": round(answer_metrics["combined_score"], 2),
                "exact_match": answer_metrics["exact_match"],
                "key_numbers_matched": round(answer_metrics["key_numbers_matched"] * 100, 2),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"Error evaluating question '{question.get('question', 'unknown')}': {e}")
            return {
                "question": question.get("question", "unknown"),
                "modality": question.get("modality", "text"),
                "error": str(e),
                "retrieved_count": 0,
                "retrieval_precision": 0,
                "answer_similarity": 0,
                "exact_match": False,
                "key_numbers_matched": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def run_full_evaluation(self, retriever_func, answer_func, max_questions: int = 6) -> pd.DataFrame:
        """Run complete evaluation suite"""
        self.results = []
        
        # Collect all questions
        questions_to_test = []
        for modality, queries in self.test_queries.items():
            questions_to_test.extend(queries[:2])  # Test 2 from each category
        
        questions_to_test = questions_to_test[:max_questions]
        
        print(f"Running evaluation on {len(questions_to_test)} questions...")
        
        for i, question in enumerate(questions_to_test):
            print(f"  Evaluating question {i+1}/{len(questions_to_test)}: {question['question'][:50]}...")
            result = self.run_single_evaluation(question, retriever_func, answer_func)
            self.results.append(result)
            print(f"    Retrieval Precision: {result.get('retrieval_precision', 0)}%")
            print(f"    Answer Similarity: {result.get('answer_similarity', 0)}%")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save results to file
        try:
            os.makedirs("data/evaluation", exist_ok=True)
            results_df.to_csv("data/evaluation/evaluation_results.csv", index=False)
            results_df.to_json("data/evaluation/evaluation_results.json", orient="records", indent=2)
            print(f"✅ Saved evaluation results to data/evaluation/")
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")
        
        return results_df
    
    def generate_summary_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Generate summary metrics from evaluation results"""
        if results_df.empty:
            return {}
        
        summary = {
            "total_questions": len(results_df),
            "avg_retrieval_precision": round(results_df["retrieval_precision"].mean(), 2),
            "avg_answer_similarity": round(results_df["answer_similarity"].mean(), 2),
            "exact_match_rate": round(results_df["exact_match"].mean() * 100, 2),
            "avg_key_numbers_matched": round(results_df["key_numbers_matched"].mean(), 2),
            "total_correct_answers": sum(results_df["answer_similarity"] > 70),
            "success_rate": round((sum(results_df["answer_similarity"] > 70) / len(results_df)) * 100, 2)
        }
        
        # By modality
        if 'modality' in results_df.columns:
            modality_stats = results_df.groupby('modality').agg({
                'retrieval_precision': 'mean',
                'answer_similarity': 'mean',
                'exact_match': 'mean'
            }).round(2)
            summary['modality_stats'] = modality_stats.to_dict()
        
        # By difficulty
        if 'difficulty' in results_df.columns:
            difficulty_stats = results_df.groupby('difficulty').agg({
                'retrieval_precision': 'mean',
                'answer_similarity': 'mean'
            }).round(2)
            summary['difficulty_stats'] = difficulty_stats.to_dict()
        
        return summary

# Dashboard functions
def create_evaluation_dashboard(results_df: pd.DataFrame):
    """Create visualization dashboard for evaluation metrics"""
    if results_df.empty:
        st.warning("No evaluation results available. Run evaluation first.")
        return
    
    # Generate summary metrics
    evaluator = MultiModalEvaluator()
    summary = evaluator.generate_summary_metrics(results_df)
    
    st.title("📊 Evaluation Dashboard")
    
    # Summary metrics
    st.subheader("📈 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Questions", summary.get("total_questions", 0))
    with col2:
        st.metric("Avg Retrieval Precision", f"{summary.get('avg_retrieval_precision', 0):.1f}%")
    with col3:
        st.metric("Avg Answer Similarity", f"{summary.get('avg_answer_similarity', 0):.1f}%")
    with col4:
        st.metric("Success Rate", f"{summary.get('success_rate', 0):.1f}%")
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Scores by question
        fig1 = px.bar(results_df, x='question', y=['retrieval_precision', 'answer_similarity'],
                     title="Performance by Question", barmode='group',
                     labels={'value': 'Score (%)', 'variable': 'Metric', 'question': 'Question'},
                     height=400)
        fig1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_chart2:
        # Modality comparison
        if 'modality' in results_df.columns:
            modality_means = results_df.groupby('modality')[['retrieval_precision', 'answer_similarity']].mean().reset_index()
            fig2 = px.bar(modality_means, x='modality', y=['retrieval_precision', 'answer_similarity'],
                         title="Performance by Modality", barmode='group',
                         labels={'value': 'Score (%)', 'variable': 'Metric', 'modality': 'Modality'},
                         height=400)
            st.plotly_chart(fig2, use_container_width=True)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Format the table for better display
    display_df = results_df.copy()
    if 'retrieved_answer' in display_df.columns:
        display_df['retrieved_answer'] = display_df['retrieved_answer'].apply(lambda x: x[:100] + "..." if len(str(x)) > 100 else x)
    
    st.dataframe(display_df[['question', 'modality', 'retrieval_precision', 'answer_similarity', 'exact_match']])
    
    # Expandable detailed view
    with st.expander("🔍 View Complete Results"):
        st.dataframe(results_df)
        
        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="evaluation_results.csv",
                mime="text/csv"
            )
        with col_dl2:
            json_data = results_df.to_json(orient="records", indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name="evaluation_results.json",
                mime="application/json"
            )
    
    # Recommendations
    st.subheader("🎯 Recommendations")
    
    avg_retrieval = summary.get('avg_retrieval_precision', 0)
    avg_answer = summary.get('avg_answer_similarity', 0)
    
    recommendations = []
    
    if avg_retrieval < 60:
        recommendations.append("Improve retrieval by adjusting chunk size or overlap")
    if avg_answer < 70:
        recommendations.append("Enhance answer generation with better prompting")
    if 'exact_match_rate' in summary and summary['exact_match_rate'] < 50:
        recommendations.append("Focus on exact answer matching for key facts")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.success("System is performing well! Consider adding more complex test cases.")

# Quick test function
def quick_test():
    """Quick test to verify evaluator works"""
    print("🧪 Testing MultiModalEvaluator...")
    
    # Create dummy retriever and answer functions for testing
    def dummy_retriever(query, top_k=10):
        return [f"Test chunk about {query} [Page 1 | text]"]
    
    def dummy_answer(context, question):
        return f"Test answer for: {question}"
    
    # Create evaluator
    evaluator = MultiModalEvaluator()
    
    # Run test
    results = evaluator.run_full_evaluation(dummy_retriever, dummy_answer, max_questions=2)
    
    if not results.empty:
        print("✅ Evaluator test passed!")
        print(f"Results shape: {results.shape}")
        return True
    else:
        print("❌ Evaluator test failed")
        return False

if __name__ == "__main__":
    # Run quick test if file is executed directly
    quick_test()