# app.py 
import streamlit as st
from retriever_local import get_retriever, retrieve_answer_chunks  
from llm import get_llm, generate_answer  
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import re
from cross_modal_reranker import get_reranker 
# Set page config
st.set_page_config(
    page_title="Multi-Modal RAG Pro", 
    layout="wide",
    page_icon="🤖"
)

# Initialize components with caching
@st.cache_resource
def init_components():
    retriever = get_retriever()
    llm = get_llm()
    return retriever, llm

retriever, llm = init_components()

# Initialize session state for history and metrics
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'retrieval_metrics' not in st.session_state:
    st.session_state.retrieval_metrics = []
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Retrieval settings
    st.subheader("Retrieval Settings")
    top_k = st.slider("Number of chunks", 5, 100, 20, 
                     help="How many text chunks to retrieve from document")
    
    col1, col2 = st.columns(2)
    with col1:
        rerank_enabled = st.checkbox("Rerank", True, 
                                    help="Use smart reranking for better results")
    with col2:
        deduplicate_enabled = st.checkbox("Deduplicate", True,
                                         help="Remove duplicate content")
    
    # Display settings
    st.subheader("Display Settings")
    show_citations = st.checkbox("Show citations", True)
    show_confidence = st.checkbox("Show confidence", True)
    show_debug = st.checkbox("Debug mode", False)
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.05,
                                        help="Minimum confidence score to show answer")
        max_context_length = st.number_input("Max context length", 1000, 10000, 4000,
                                           help="Maximum tokens for context")
        
    st.divider()
    
    # System info
    st.subheader("System Info")
    st.info("""
    **Status**: Ready
    **Model**: Gemini 2.5 Flash
    **Embeddings**: MiniLM-L3-v2
    **Storage**: Local Vector Store
    """)
    
    # Quick actions
    st.divider()
    st.subheader("Quick Actions")
    
    if st.button("📊 Run Evaluation", use_container_width=True):
        st.session_state.run_evaluation = True
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.retrieval_metrics = []
        st.rerun()
    
    if st.button("📄 Generate Summary", use_container_width=True):
        st.session_state.generate_summary = True

# Main title
st.title("🤖 Multi-Modal Document Intelligence Assistant")
st.caption("Ask questions about the multi-modal RAG assignment document")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Analytics", "🔍 Evaluation", "📄 Document"])

# Tab 1: Chat Interface
with tab1:
    # Question input with example questions
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        question = st.text_input(
            "Ask a question:",
            placeholder="e.g., What is the weight of Accuracy & Faithfulness?",
            key="question_input"
        )
    
    with col2:
        search_clicked = st.button("🔍 Search", use_container_width=True, type="primary")
    
    with col3:
        # Example questions dropdown
        example_questions = [
            "What is the weight of Accuracy & Faithfulness?",
            "What are the delivery options?",
            "What is multi-modal coverage weight?",
            "What evaluation criteria are mentioned?",
            "What are the expected features?"
        ]
        example_selected = st.selectbox(
            "Examples",
            ["Select example..."] + example_questions,
            key="example_select"
        )
        
        if example_selected != "Select example...":
            question = example_selected
    
    # Process question
    if (question and (search_clicked or question != st.session_state.get('last_question', ''))):
        st.session_state.last_question = question
        
        with st.spinner("🔍 Searching document..."):
            # Store start time for performance metrics
            start_time = datetime.now()
            
            # Retrieve chunks with enhanced retriever
            chunks = retriever.retrieve_answer_chunks(
                question, 
                top_k=top_k,
                rerank=rerank_enabled,
                deduplicate=deduplicate_enabled
            )
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            # Store retrieval metrics
            retrieval_metric = {
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "chunks_retrieved": len(chunks),
                "retrieval_time": retrieval_time,
                "top_k": top_k
            }
            st.session_state.retrieval_metrics.append(retrieval_metric)
        
        if chunks:
            # Show retrieval stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Chunks Retrieved", len(chunks))
            with col2:
                # Count content types
                content_types = {}
                for chunk in chunks:
                    if '|' in chunk:
                        content_type = chunk.split('|')[1].split(']')[0].strip().lower()
                        content_types[content_type] = content_types.get(content_type, 0) + 1
                st.metric("Content Types", len(content_types))
            with col3:
                st.metric("Retrieval Time", f"{retrieval_time:.2f}s")
            with col4:
                if chunks:
                    avg_len = sum(len(c) for c in chunks) / len(chunks)
                    st.metric("Avg Chunk Length", f"{int(avg_len)} chars")
            
            # Display retrieved chunks in expander
            with st.expander(f"🔍 Retrieved Context ({len(chunks)} chunks)", expanded=show_debug):
                if chunks:
                    for i, chunk in enumerate(chunks[:10]):  # Show first 10
                        # Extract source info
                        source_info = ""
                        if '[' in chunk and ']' in chunk:
                            source_part = chunk.split(']')[0] + ']'
                            content = chunk.split(']', 1)[1] if ']' in chunk else chunk
                            source_info = source_part
                        else:
                            content = chunk
                        
                        # Display chunk
                        st.markdown(f"**Chunk {i+1}** {source_info}")
                        st.text(content[:300] + "..." if len(content) > 300 else content)
                        st.divider()
                else:
                    st.warning("No chunks retrieved")
            
            # Generate answer
            with st.spinner("🤔 Generating answer..."):
                generation_start = datetime.now()
                
                if show_citations:
                    # Use enhanced answer generation with citations
                    answer, citations, confidence = llm.generate_answer_with_citations(chunks, question)
                    generation_time = (datetime.now() - generation_start).total_seconds()
                    
                    # Check confidence threshold
                    if confidence/100 < confidence_threshold:
                        st.warning(f"⚠️ Low confidence answer ({confidence:.1f}%) - Verify with sources below")
                    
                    # Display answer
                    st.subheader("💡 Answer")
                    
                    # Style answer based on confidence
                    if confidence >= 80:
                        st.success(answer)
                    elif confidence >= 60:
                        st.info(answer)
                    else:
                        st.warning(answer)
                    
                    # Display confidence score
                    if show_confidence:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.progress(confidence/100, 
                                       text=f"Confidence: {confidence:.1f}%")
                        with col2:
                            st.caption(f"Generated in {generation_time:.2f}s")
                    
                    # Display citations
                    if show_citations and citations:
                        st.subheader("📚 Sources")
                        for i, citation in enumerate(citations):
                            with st.container():
                                col1, col2 = st.columns([1, 4])
                                with col1:
                                    st.markdown(f"**Source {i+1}**")
                                    st.caption(f"Page {citation['page']}")
                                    st.caption(f"Type: {citation['type']}")
                                with col2:
                                    st.text(citation['text'][:500] + "..." if len(citation['text']) > 500 else citation['text'])
                                st.divider()
                else:
                    # Use simple answer generation (backward compatible)
                    context = "\n\n".join(chunks[:10])  # Limit context
                    answer = llm.generate_answer(context, question)
                    generation_time = (datetime.now() - generation_start).total_seconds()
                    
                    st.subheader("💡 Answer")
                    st.markdown(answer)
                    st.caption(f"Generated in {generation_time:.2f}s")
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer[:500] + "..." if len(answer) > 500 else answer,
                "timestamp": datetime.now().isoformat(),
                "chunks_used": len(chunks),
                "confidence": confidence if show_citations else None,
                "generation_time": generation_time
            })
            
        else:
            st.error("❌ No relevant content found for your question.")
            st.info("""
            Try:
            1. Rephrasing your question
            2. Using simpler keywords
            3. Checking if the document contains this information
            """)
    
    # Display chat history
    if st.session_state.chat_history:
        st.divider()
        st.subheader("💭 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            with st.expander(f"Q: {chat['question'][:50]}...", expanded=False):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                if chat.get('confidence'):
                    st.caption(f"Confidence: {chat['confidence']:.1f}%")
                st.caption(f"Time: {chat['timestamp'][11:19]} | Chunks: {chat['chunks_used']}")

# Tab 2: Analytics Dashboard
with tab2:
    st.header("📊 System Analytics")
    
    if st.session_state.retrieval_metrics:
        # Convert to DataFrame
        metrics_df = pd.DataFrame(st.session_state.retrieval_metrics)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_retrieval = metrics_df['retrieval_time'].mean()
            st.metric("Avg Retrieval Time", f"{avg_retrieval:.2f}s")
        with col2:
            avg_chunks = metrics_df['chunks_retrieved'].mean()
            st.metric("Avg Chunks Retrieved", f"{avg_chunks:.0f}")
        with col3:
            total_queries = len(metrics_df)
            st.metric("Total Queries", total_queries)
        with col4:
            if 'confidence' in st.session_state.chat_history[0] if st.session_state.chat_history else False:
                avg_conf = sum([c.get('confidence', 0) for c in st.session_state.chat_history]) / len(st.session_state.chat_history)
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            else:
                st.metric("Chat History", len(st.session_state.chat_history))
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Retrieval time over time
            if len(metrics_df) > 1:
                metrics_df['time_index'] = range(len(metrics_df))
                fig1 = px.line(metrics_df, x='time_index', y='retrieval_time',
                             title="Retrieval Time Trend",
                             labels={'retrieval_time': 'Time (seconds)', 'time_index': 'Query Number'})
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Chunks retrieved distribution
            fig2 = px.histogram(metrics_df, x='chunks_retrieved',
                              title="Chunks Retrieved Distribution",
                              nbins=10)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed metrics table
        with st.expander("📋 Detailed Metrics"):
            st.dataframe(metrics_df[['question', 'chunks_retrieved', 'retrieval_time', 'timestamp']])
    
    else:
        st.info("No analytics data yet. Start asking questions to see metrics.")

# Tab 3: Evaluation Dashboard
with tab3:
    st.header("🔍 Evaluation Dashboard")
    
    # Evaluation controls
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🔄 Run New Evaluation", type="primary"):
            with st.spinner("Running evaluation..."):
                # Import evaluator
                try:
                    from evaluator import MultiModalEvaluator
                    evaluator = MultiModalEvaluator()
                    results = evaluator.run_full_evaluation(
                        retriever.retrieve_answer_chunks,
                        llm.generate_answer
                    )
                    st.session_state.evaluation_results = results
                    st.success("Evaluation completed!")
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
    
    with col2:
        if st.button("📥 Load Sample Results"):
            # Create sample evaluation data
            sample_data = {
                "question": [
                    "What is the weight of Accuracy & Faithfulness?",
                    "What are the delivery options?",
                    "What is multi-modal coverage weight?"
                ],
                "modality": ["text", "text", "table"],
                "retrieval_score": [85, 92, 78],
                "answer_similarity": [90, 88, 82],
                "exact_match": [True, True, False]
            }
            st.session_state.evaluation_results = pd.DataFrame(sample_data)
    
    # In app.py, find the Evaluation Dashboard section (around line 386):

# Display evaluation results
if st.session_state.evaluation_results is not None:
    results_df = st.session_state.evaluation_results
    
    # Summary metrics
    st.subheader("📈 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Use the correct column name - check what's available
        if 'retrieval_precision' in results_df.columns:
            avg_retrieval = results_df['retrieval_precision'].mean()
        elif 'retrieval_score' in results_df.columns:
            avg_retrieval = results_df['retrieval_score'].mean()
        else:
            avg_retrieval = 0
        st.metric("Avg Retrieval Score", f"{avg_retrieval:.1f}%")
    
    with col2:
        if 'answer_similarity' in results_df.columns:
            avg_answer = results_df['answer_similarity'].mean()
        else:
            avg_answer = 0
        st.metric("Avg Answer Similarity", f"{avg_answer:.1f}%")
    
    with col3:
        if 'exact_match' in results_df.columns:
            exact_match_rate = results_df['exact_match'].mean() * 100
        else:
            exact_match_rate = 0
        st.metric("Exact Match Rate", f"{exact_match_rate:.1f}%")
    
    with col4:
        total_tests = len(results_df)
        st.metric("Total Tests", total_tests)
    
    # Charts - update column names here too
    col1, col2 = st.columns(2)
    
    with col1:
        # Scores by question
        y_columns = []
        if 'retrieval_precision' in results_df.columns:
            y_columns.append('retrieval_precision')
        elif 'retrieval_score' in results_df.columns:
            y_columns.append('retrieval_score')
        
        if 'answer_similarity' in results_df.columns:
            y_columns.append('answer_similarity')
        
        if y_columns:
            fig1 = px.bar(results_df, x='question', y=y_columns,
                         title="Performance by Question", barmode='group',
                         labels={'value': 'Score (%)', 'variable': 'Metric'},
                         height=400)
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Modality comparison
        if 'modality' in results_df.columns and len(y_columns) > 0:
            modality_scores = results_df.groupby('modality')[y_columns].mean().reset_index()
            fig2 = px.bar(modality_scores, x='modality', y=y_columns,
                         title="Performance by Modality", barmode='group',
                         labels={'value': 'Score (%)', 'variable': 'Metric'},
                         height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed results
            with st.expander("📋 Detailed Evaluation Results"):
                st.dataframe(results_df)

                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="evaluation_results.csv",
                    mime="text/csv"
                )
    
        else:
                st.info("Run an evaluation to see metrics here.")

# Tab 4: Document Analysis - UPDATED
with tab4:
    st.header("📄 Document Analysis")
    
    # Document summary section
    st.subheader("Document Summary")
    
    # Create columns for summary options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📖 Generate Full Summary", key="full_summary_btn", use_container_width=True):
            with st.spinner("Analyzing document..."):
                # Use terms that ARE in the document
                summary_queries = [
                    "project context",
                    "objectives",
                    "expected features",
                    "deliverables", 
                    "evaluation criteria",
                    "duration options",
                    "multi-modal RAG",
                    "assignment challenge",
                    "weight percentage",
                    "accuracy faithfulness"
                ]
                
                summary_chunks = []
                for query in summary_queries:
                    try:
                        chunks = retriever.retrieve_answer_chunks(query, top_k=2)
                        summary_chunks.extend(chunks)
                    except:
                        continue
                
                # Deduplicate
                seen = set()
                unique_chunks = []
                for chunk in summary_chunks:
                    # Simple hash of first 100 chars
                    chunk_hash = hash(chunk[:100])
                    if chunk_hash not in seen:
                        seen.add(chunk_hash)
                        unique_chunks.append(chunk)
                
                if unique_chunks:
                    # Generate summary
                    summary = llm.generate_document_summary(unique_chunks[:15])
                    
                    # Display summary
                    st.markdown("### 📖 Comprehensive Document Summary")
                    st.markdown(summary)
                    
                    # Show sources
                    with st.expander("🔍 Sources Used for Summary"):
                        for i, chunk in enumerate(unique_chunks[:8]):
                            st.markdown(f"**Source {i+1}**")
                            st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                            st.divider()
                else:
                    st.warning("Could not retrieve content for summary.")
                    st.info("Try asking specific questions first to populate the retriever cache.")
    
    with col2:
        if st.button("⚡ Quick Overview", key="quick_overview_btn", use_container_width=True):
            with st.spinner("Generating quick overview..."):
                # Direct query for overview
                chunks = retriever.retrieve_answer_chunks("project about multi-modal RAG", top_k=5)
                if chunks:
                    context = "\n".join(chunks[:3])
                    answer = llm.generate_answer(context, "What is this project about? Provide a brief overview.")
                    st.markdown("### 🎯 Project Overview")
                    st.markdown(answer)
                else:
                    st.info("Ask specific questions first to build context.")
    
    # Quick summary sections
    st.subheader("Quick Summaries")
    
    col_quick1, col_quick2, col_quick3 = st.columns(3)
    
    with col_quick1:
        if st.button("📊 Evaluation Summary", key="eval_summary_btn", use_container_width=True):
            with st.spinner("Extracting evaluation info..."):
                chunks = retriever.retrieve_answer_chunks("evaluation criteria weight", top_k=5)
                if chunks:
                    context = "\n".join(chunks[:3])
                    answer = llm.generate_answer(context, "Summarize the evaluation criteria with weights")
                    st.markdown("### ⚖️ Evaluation Criteria")
                    st.markdown(answer)
    
    with col_quick2:
        if st.button("🎯 Features Summary", key="features_summary_btn", use_container_width=True):
            with st.spinner("Extracting features..."):
                chunks = retriever.retrieve_answer_chunks("expected features", top_k=5)
                if chunks:
                    context = "\n".join(chunks[:3])
                    answer = llm.generate_answer(context, "List and briefly describe the expected features")
                    st.markdown("### 🔧 Expected Features")
                    st.markdown(answer)
    
    with col_quick3:
        if st.button("📦 Deliverables Summary", key="deliverables_summary_btn", use_container_width=True):
            with st.spinner("Extracting deliverables..."):
                chunks = retriever.retrieve_answer_chunks("deliverables", top_k=5)
                if chunks:
                    context = "\n".join(chunks[:3])
                    answer = llm.generate_answer(context, "What are the deliverables?")
                    st.markdown("### 📦 Deliverables")
                    st.markdown(answer)
    
    # Statistics extraction
    st.subheader("📊 Key Statistics & Numbers")
    
    if st.button("Extract Key Numbers", key="stats_btn") or st.session_state.get('generate_summary', False):
        st.session_state.generate_summary = False
        
        with st.spinner("Finding numerical data..."):
            # Look for percentages and numbers - use specific terms from document
            stats_queries = [
                "25%", "20%", "15%", "10%", "percentage",
                "weight", "score", "metric", "criteria",
                "24-hour", "48-hour", "duration", "options"
            ]
            
            all_stats_chunks = []
            for query in stats_queries:
                try:
                    chunks = retriever.retrieve_answer_chunks(query, top_k=2)
                    all_stats_chunks.extend(chunks)
                except:
                    continue
            
            if all_stats_chunks:
                # Extract numbers and percentages
                numerical_data = []
                for chunk in all_stats_chunks:
                    # Find percentages
                    percentages = re.findall(r'(\d+%)', chunk)
                    # Find numbers
                    numbers = re.findall(r'\b(\d+)\b', chunk)
                    
                    if percentages or numbers:
                        # Get context around the numbers
                        text_preview = chunk[:250] + "..." if len(chunk) > 250 else chunk
                        
                        numerical_data.append({
                            "text": text_preview,
                            "percentages": percentages,
                            "numbers": [n for n in numbers if int(n) > 9],  # Filter small numbers
                        })
                
                if numerical_data:
                    # Deduplicate
                    unique_data = []
                    seen_texts = set()
                    for data in numerical_data:
                        text_hash = hash(data['text'][:50])
                        if text_hash not in seen_texts:
                            seen_texts.add(text_hash)
                            unique_data.append(data)
                    
                    st.write(f"**📈 Found {len(unique_data)} numerical references:**")
                    
                    # Group by type
                    percentages_data = [d for d in unique_data if d['percentages']]
                    numbers_data = [d for d in unique_data if d['numbers'] and not d['percentages']]
                    
                    if percentages_data:
                        st.markdown("### 🔢 Percentages Found")
                        for data in percentages_data[:8]:
                            with st.container():
                                st.markdown(f"**Percentages:** {', '.join(data['percentages'])}")
                                st.text(data['text'])
                                st.divider()
                    
                    if numbers_data:
                        st.markdown("### 🔢 Other Numbers Found")
                        for data in numbers_data[:5]:
                            with st.container():
                                st.markdown(f"**Numbers:** {', '.join(data['numbers'][:3])}")
                                st.text(data['text'])
                                st.divider()
                    
                    # Create a summary
                    all_percentages = []
                    for data in percentages_data:
                        all_percentages.extend(data['percentages'])
                    
                    if all_percentages:
                        st.markdown("### 📋 Percentage Summary")
                        st.write(f"Found {len(all_percentages)} percentage references:")
                        for pct in sorted(set(all_percentages), key=lambda x: int(x[:-1]), reverse=True):
                            st.write(f"- {pct}")
                
                else:
                    st.info("No numerical statistics found in the retrieved content.")
                    st.info("Try asking specific questions like 'What is accuracy weight?' first.")
            else:
                st.warning("Could not retrieve content for statistics extraction.")
                st.info("""
                **Tips:**
                1. First ask specific questions like "What is accuracy weight?"
                2. Try queries like "25%" or "percentage"
                3. Check if ingestion was successful
                """)
    
    # Document structure analysis
    st.subheader("📑 Document Structure")
    
    if st.button("Analyze Document Structure", key="structure_btn"):
        with st.spinner("Analyzing document structure..."):
            # Look for section headers and structure
            structure_queries = [
                "page 1", "page 2", "section", "header",
                "title", "heading", "###", "##", "#"
            ]
            
            structure_chunks = []
            for query in structure_queries:
                try:
                    chunks = retriever.retrieve_answer_chunks(query, top_k=2)
                    structure_chunks.extend(chunks)
                except:
                    continue
            
            if structure_chunks:
                # Look for potential headers
                potential_headers = []
                for chunk in structure_chunks:
                    lines = chunk.split('\n')
                    for line in lines:
                        line = line.strip()
                        if (line.startswith('#') or 
                            (len(line) < 100 and line.upper() == line and line.isalpha()) or
                            '=====' in line):
                            potential_headers.append(line[:100])
                
                if potential_headers:
                    st.markdown("### 🏗️ Document Structure Detected")
                    st.write("Potential sections/headers found:")
                    for header in list(set(potential_headers))[:10]:
                        if header:  # Skip empty
                            st.write(f"- {header}")
                else:
                    st.info("No clear document structure detected.")
            else:
                st.info("Could not analyze document structure.")
# In app.py, add a new tab
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "📊 Analytics", "🔍 Evaluation", "📄 Document", "🖼️ Visual"])

# Tab 5: Visual Content Analysis
with tab5:
    st.header("🖼️ Visual Content Analysis")
    
    st.subheader("Chart & Image Retrieval")
    
    visual_query = st.text_input(
        "Ask about charts, graphs, or visual content:",
        placeholder="e.g., Find charts about evaluation criteria",
        key="visual_query"
    )
    
    if st.button("🔍 Search Visual Content", key="visual_search_btn"):
        if visual_query:
            with st.spinner("Searching visual content..."):
                try:
                    from cross_modal_retriever import get_cross_modal_retriever
                    cross_modal = get_cross_modal_retriever()
                    
                    # Get visual results
                    visual_results = cross_modal.retrieve_visual_content(visual_query, top_k=5)
                    
                    if visual_results:
                        st.success(f"Found {len(visual_results)} visual results")
                        
                        for i, result in enumerate(visual_results):
                            with st.expander(f"📈 Visual Result {i+1} (Score: {result['score']:.2f})", expanded=True):
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    st.markdown(f"**Type:** {result['chart_type']}")
                                    st.markdown(f"**Page:** {result['page']}")
                                    st.markdown(f"**Similarity:** {result['score']:.2f}")
                                
                                with col2:
                                    st.markdown("**Description:**")
                                    st.info(result['description'])
                                    
                                    # Show metadata
                                    if 'metadata' in result:
                                        metadata = result['metadata']
                                        with st.expander("📋 Chart Metadata"):
                                            if metadata.get('has_axes'):
                                                st.markdown("✅ Has axes labels")
                                            if metadata.get('has_grid'):
                                                st.markdown("✅ Has grid lines")
                                            if metadata.get('has_legend'):
                                                st.markdown("✅ Has legend")
                                            if metadata.get('ocr_text'):
                                                st.markdown(f"**OCR Text:** {metadata['ocr_text'][:200]}...")
                                
                                st.divider()
                    else:
                        st.info("No visual content found for your query.")
                        
                except Exception as e:
                    st.error(f"Visual search error: {e}")
                    st.info("Try text-based search in the Chat tab.")
        else:
            st.warning("Please enter a query about visual content.")
    
    # Visual content statistics
    st.subheader("📊 Visual Content Statistics")
    
    if st.button("Analyze Visual Content", key="analyze_visual_btn"):
        with st.spinner("Analyzing document for visual content..."):
            try:
                from cross_modal_retriever import get_cross_modal_retriever
                cross_modal = get_cross_modal_retriever()
                
                if hasattr(cross_modal, 'chart_data'):
                    chart_types = {}
                    for chart in cross_modal.chart_data:
                        chart_type = chart.get("chart_type", "unknown")
                        chart_types[chart_type] = chart_types.get(chart_type, 0) + 1
                    
                    if chart_types:
                        # Create visualization
                        import plotly.express as px
                        
                        df = pd.DataFrame({
                            'Chart Type': list(chart_types.keys()),
                            'Count': list(chart_types.values())
                        })
                        
                        fig = px.pie(df, values='Count', names='Chart Type',
                                    title="Distribution of Chart Types in Document")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("### 📋 Visual Content Summary")
                        st.write(f"**Total charts/figures found:** {len(cross_modal.chart_data)}")
                        
                        for chart_type, count in chart_types.items():
                            st.write(f"- **{chart_type}:** {count}")
                    else:
                        st.info("No charts detected in the document.")
                else:
                    st.info("Chart data not available. Run ingestion with chart extraction.")
                    
            except Exception as e:
                st.error(f"Visual analysis error: {e}")
# Footer
st.divider()
footer_cols = st.columns(4)
with footer_cols[0]:
    st.caption("🤖 Powered by Gemini AI")
with footer_cols[1]:
    st.caption("🔍 Multi-Modal Retrieval")
with footer_cols[2]:
    st.caption("📊 Enhanced Analytics")
with footer_cols[3]:
    st.caption(f"🕐 {datetime.now().strftime('%H:%M')}")

# Run evaluation if triggered from sidebar
if st.session_state.get('run_evaluation', False):
    st.session_state.run_evaluation = False
    st.rerun()