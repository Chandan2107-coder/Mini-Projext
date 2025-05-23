import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from langchain.vectorstores import FAISS
from langchain_community.llms import Cohere
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os
import cohere
import time
from datetime import datetime

# Load environment variables
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

# Page configuration
st.set_page_config(
    page_title="DocuChat Pro - AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 3rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .sidebar-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Status Cards */
    .status-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    .status-card.warning {
        border-left-color: #f59e0b;
        background: #fefbf3;
    }
    
    .status-card.error {
        border-left-color: #ef4444;
        background: #fef2f2;
    }
    
    .status-card.success {
        border-left-color: #10b981;
        background: #f0fdf4;
    }
    
    /* Main Content Cards */
    .content-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f1f5f9;
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin: 0;
    }
    
    /* Chat Interface */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background: #f8fafc;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    .chat-message {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .chat-question {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .chat-answer {
        color: #475569;
        line-height: 1.6;
    }
    
    /* Progress Bar Custom */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* File Uploader */
    .uploadedFile {
        background: white;
        border-radius: 8px;
        border: 2px dashed #cbd5e1;
        padding: 1rem;
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        background: #f8fafc;
        border-radius: 12px;
        margin-top: 3rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# Check API key
if not cohere_api_key:
    st.error("🚫 Cohere API Key not found. Please set your COHERE_API_KEY in the .env file.")
    st.stop()

# Initialize Cohere client
co = cohere.Client(cohere_api_key)

# All your existing functions remain the same
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=600,
        length_function=len
    )
    return text_splitter.split_text(text)

class CohereEmbeddingsCustom:
    def _init_(self, api_key, model="embed-english-v3.0"):
        self.client = cohere.Client(api_key)
        self.model = model
    
    def embed_documents(self, texts):
        try:
            batch_size = 96
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type="search_document"
                )
                all_embeddings.extend(response.embeddings)
            
            return all_embeddings
        except Exception as e:
            st.error(f"Error embedding documents: {str(e)}")
            return []
    
    def embed_query(self, text):
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_query"
            )
            return response.embeddings[0]
        except Exception as e:
            st.error(f"Error embedding query: {str(e)}")
            return []
    
    def _call_(self, text):
        if isinstance(text, list):
            return self.embed_documents(text)
        else:
            return self.embed_query(text)

def get_vector_store(text_chunks):
    try:
        embeddings = CohereEmbeddingsCustom(cohere_api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

def get_conversational_chain():
    prompt_template = """
    Based on the provided context, answer the question in detail. If the answer cannot be found in the context, 
    respond with "Answer is not available in the provided context."

    Context: {context}

    Question: {question}

    Detailed Answer:
    """
    
    model = Cohere(
        cohere_api_key=cohere_api_key,
        model="command",
        temperature=0.1,
        max_tokens=1000
    )
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def summarize_text_with_chat(text):
    try:
        max_chunk_size = 40000
        
        if len(text) <= max_chunk_size:
            response = co.chat(
                message=f"""Provide a concise but comprehensive summary of this text. Include key points and main ideas:

                {text}""",
                model="command",
                temperature=0.1,
                max_tokens=800
            )
            return response.text
        else:
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            summaries = []
            
            for i, chunk in enumerate(chunks[:3]):
                response = co.chat(
                    message=f"""Summarize this text part {i+1}:

                    {chunk}""",
                    model="command",
                    temperature=0.1,
                    max_tokens=500
                )
                summaries.append(response.text)
            
            if len(chunks) > 3:
                combined_summary = "\n\n".join(summaries) + f"\n\n[Note: Summarized first 3 parts of {len(chunks)} total parts for faster processing]"
            else:
                combined_summary = "\n\n".join(summaries)
            
            return combined_summary
            
    except Exception as e:
        st.error(f"Error in summarization: {str(e)}")
        return "Error occurred during summarization."

def process_user_input(user_question):
    try:
        embeddings = CohereEmbeddingsCustom(cohere_api_key)
        
        vector_store = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        docs = vector_store.similarity_search(user_question, k=2)
        
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question}, 
            return_only_outputs=True
        )
        
        return response["output_text"]
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        return "Sorry, I encountered an error while processing your question."

def vector_store_exists():
    return os.path.exists("faiss_index")

# Main Application
def main():
    # Header
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🤖 DocuChat Pro</h1>
        <p>Advanced AI-Powered Document Analysis & Chat Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'pdf_texts' not in st.session_state:
        st.session_state['pdf_texts'] = {}
    if 'summary' not in st.session_state:
        st.session_state['summary'] = ""
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'processed_files' not in st.session_state:
        st.session_state['processed_files'] = []

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">📁 Document Management</div>
        </div>
        """, unsafe_allow_html=True)
        
        pdf_docs = st.file_uploader(
            "Upload PDF Documents",
            accept_multiple_files=True,
            type=['pdf'],
            help="Select one or more PDF files to analyze"
        )
        
        if pdf_docs:
            st.markdown(f"📄 {len(pdf_docs)} file(s) selected**")
            for pdf in pdf_docs:
                st.markdown(f"• {pdf.name}")
        
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    progress_container = st.empty()
                    
                    # Progress tracking
                    progress_container.markdown("*Step 1:* Extracting text from PDFs...")
                    progress_bar = st.progress(0)
                    time.sleep(0.5)
                    
                    raw_text = get_pdf_text(pdf_docs)
                    progress_bar.progress(30)
                    
                    progress_container.markdown("*Step 2:* Creating text chunks...")
                    time.sleep(0.5)
                    text_chunks = get_text_chunks(raw_text)
                    progress_bar.progress(60)
                    
                    progress_container.markdown("*Step 3:* Building AI knowledge base...")
                    time.sleep(0.5)
                    success = get_vector_store(text_chunks)
                    progress_bar.progress(100)
                    
                    if success:
                        st.session_state['processed_files'] = [pdf.name for pdf in pdf_docs]
                        st.success("✅ Documents processed successfully!")
                        progress_container.empty()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Processing failed. Please try again.")
            else:
                st.warning("Please upload PDF files first.")
        
        # Status Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">📊 System Status</div>
        </div>
        """, unsafe_allow_html=True)
        
        if vector_store_exists():
            st.markdown("""
            <div class="status-card success">
                ✅ AI Knowledge Base: Ready
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state['processed_files']:
                st.markdown("📚 Processed Documents:")
                for file in st.session_state['processed_files']:
                    st.markdown(f"• {file}")
        else:
            st.markdown("""
            <div class="status-card warning">
                ⏳ Please process documents first
            </div>
            """, unsafe_allow_html=True)
        
        # Statistics
        if st.session_state['chat_history']:
            st.markdown("""
            <div class="metric-card">
                <p class="metric-value">{}</p>
                <p class="metric-label">Questions Asked</p>
            </div>
            """.format(len(st.session_state['chat_history'])), unsafe_allow_html=True)

    # Main Content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="content-card fade-in">
            <div class="card-header">
                <span style="font-size: 1.5rem;">📋</span>
                <h2 class="card-title">Document Summary</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if pdf_docs and vector_store_exists():
            pdf_names = [pdf.name for pdf in pdf_docs]
            
            selected_pdfs = st.multiselect(
                "Select documents to summarize:",
                options=pdf_names,
                default=pdf_names,
                help="Choose which documents to include in the summary"
            )
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📝 Generate Summary", type="secondary", use_container_width=True):
                    if selected_pdfs:
                        with st.spinner("Creating intelligent summary..."):
                            text = ""
                            for pdf_name, pdf_file in zip(pdf_names, pdf_docs):
                                if pdf_name in selected_pdfs:
                                    if pdf_name not in st.session_state['pdf_texts']:
                                        st.session_state['pdf_texts'][pdf_name] = get_pdf_text([pdf_file])
                                    text += st.session_state['pdf_texts'][pdf_name]
                            
                            if text:
                                summary = summarize_text_with_chat(text)
                                st.session_state['summary'] = summary
                                st.success("✅ Summary generated successfully!")
                                time.sleep(1)
                                st.rerun()
            
            with col_btn2:
                if st.button("🗑 Clear Summary", use_container_width=True):
                    st.session_state['summary'] = ""
                    st.rerun()
        
        # Display Summary
        if st.session_state['summary']:
            st.markdown("""
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea; margin-top: 1rem;">
            """, unsafe_allow_html=True)
            st.markdown("### 📄 Document Summary")
            st.markdown(st.session_state['summary'])
            st.markdown("</div>", unsafe_allow_html=True)
        elif vector_store_exists():
            st.info("💡 Generate a summary to see document insights here.")
        else:
            st.info("📤 Upload and process documents to generate summaries.")
    
    with col2:
        st.markdown("""
        <div class="content-card fade-in">
            <div class="card-header">
                <span style="font-size: 1.5rem;">💬</span>
                <h2 class="card-title">AI Chat Assistant</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if vector_store_exists():
            # Chat Input
            user_question = st.text_input(
                "Ask me anything about your documents:",
                placeholder="What are the key findings in the document?",
                help="Ask specific questions about the content of your uploaded documents"
            )
            
            col_chat1, col_chat2 = st.columns([3, 1])
            with col_chat1:
                ask_button = st.button("🤔 Ask Question", type="primary", use_container_width=True)
            with col_chat2:
                if st.session_state['chat_history']:
                    if st.button("🗑 Clear Chat", use_container_width=True):
                        st.session_state['chat_history'] = []
                        st.rerun()
            
            if ask_button and user_question:
                with st.spinner("🔍 Analyzing documents..."):
                    response = process_user_input(user_question)
                    
                    # Add to chat history with timestamp
                    st.session_state['chat_history'].append({
                        'question': user_question,
                        'answer': response,
                        'timestamp': datetime.now().strftime("%H:%M")
                    })
                    st.rerun()
            
            # Chat History Display
            if st.session_state['chat_history']:
                st.markdown("### 💭 Conversation History")
                
                chat_container = st.container()
                with chat_container:
                    for i, chat in enumerate(reversed(st.session_state['chat_history'][-5:])):
                        st.markdown(f"""
                        <div class="chat-message">
                            <div class="chat-question">
                                <strong>Q:</strong> {chat['question']}
                                <span style="float: right; font-size: 0.8rem; color: #64748b;">{chat['timestamp']}</span>
                            </div>
                            <div class="chat-answer">
                                <strong>A:</strong> {chat['answer']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                if len(st.session_state['chat_history']) > 5:
                    st.info(f"Showing last 5 conversations. Total: {len(st.session_state['chat_history'])}")
            else:
                st.markdown("""
                <div style="text-align: center; padding: 2rem; color: #64748b;">
                    <span style="font-size: 3rem;">🤖</span>
                    <p>Start a conversation by asking a question!</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #64748b;">
                <span style="font-size: 4rem;">📤</span>
                <h3>Ready to Chat!</h3>
                <p>Upload and process your PDF documents to start an AI-powered conversation.</p>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <h4>🤖 DocuChat Pro</h4>
        <p>Powered by Cohere AI • Built with Streamlit • Designed for Professionals</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            ⚡ Fast Processing • 🔒 Secure • 🎯 Accurate AI Responses
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
