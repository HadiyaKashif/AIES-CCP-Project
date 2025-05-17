from io import BytesIO
import io
import streamlit as st
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from fpdf import FPDF
from streamlit_quill import st_quill

from config import pc, gemini_embeddings, PINECONE_INDEX_NAME, gemini_llm, GEMINI_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID, web_search_tool, search
from chat_export import export_chat_history, generate_chat_pdf
from document_processing import process_documents, scrape_website, process_text_data
from vector_store import initialize_pinecone, clear_index
from rag_techniques import reciprocal_rank_fusion, generate_query_variations, generate_reasoning_steps, generate_sub_questions
from web_scrapping import scrape_website, search_google

st.set_page_config(page_title="Advanced Gemini RAG System", page_icon="🤖")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed" not in st.session_state:
    st.session_state.processed = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rich_notes" not in st.session_state:
    st.session_state.rich_notes = ""
if "show_notes" not in st.session_state:
    st.session_state.show_notes = False 

# Floating Notes Button
float_style = """
    <style>
        div[data-testid="stFloatNotes"] {
            position: fixed;
            top: 15px;
            right: 20px;
            z-index: 9999;
        }
        div[data-testid="stFloatNotes"] button {
            background-color: #f5f5f5;
            border-radius: 30px;
            padding: 0.5rem 0.8rem;
            border: 1px solid #ccc;
            box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
        }
    </style>
""" 
st.markdown(float_style, unsafe_allow_html=True)
with st.container():
    st.markdown('<div data-testid="stFloatNotes">', unsafe_allow_html=True)
    if st.button("Notes + ", key="toggle_notes_btn"):
        st.session_state.show_notes = not st.session_state.get("show_notes", False)
    st.markdown('</div>', unsafe_allow_html=True)

# UI
st.title("🤖 Advanced Gemini RAG System")
st.caption("Now with RAG-Fusion, Multi-Query, Decomposition + Notes")

initialize_pinecone()

with st.sidebar:
    st.header("📂 Data Management")
    uploaded_files = st.file_uploader("Upload files", type=["pdf", "pptx", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)
    url = st.text_input("Or enter website URL")
    col1, col2 = st.columns(2)
    if col1.button("Process Files", disabled=not uploaded_files):
        with st.spinner("Processing files..."):
            process_documents(uploaded_files)
    if col2.button("Scrape Site", disabled=not url):
        with st.spinner("Scraping site..."):
            process_text_data(scrape_website(url), url)
    if st.button("🧹 Clear All Data", type="secondary"):
        clear_index()
    
    # Added Chat Export UI
    st.divider()
    st.header("Chat Export")
    export_format = st.radio("Format", options=["json", "txt", "pdf"])
    if st.button("Export Chat History"):
        if st.session_state.messages:
            data, mime, fname = export_chat_history(st.session_state.messages, export_format)
            st.download_button(
                label=f"Download {export_format.upper()}",
                data=data,
                file_name=fname,
                mime=mime
            )
        else:
            st.warning("No chat history to export!")

# Chat interface
st.divider()
st.header("🌐 Web Search Fallback")
use_web_search = st.checkbox("Search from web if no document matches", 
                           disabled=not (GOOGLE_API_KEY and GOOGLE_CSE_ID))

if use_web_search and not (GOOGLE_API_KEY and GOOGLE_CSE_ID):
    st.warning("Please set GOOGLE_API_KEY and GOOGLE_CSE_ID in .env to enable web search")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# In your chat interface section, replace the current if/else logic with this:

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if not st.session_state.processed and not use_web_search:
            st.warning("Please process or scrape some documents first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    response = ""
                    document_answer_found = False
                    
                    # First try to get answer from documents if processed
                    if st.session_state.processed:
                        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
                        queries = generate_query_variations(prompt) + generate_sub_questions(prompt) + generate_reasoning_steps(prompt)
                        all_docs = [retriever.invoke(q) for q in queries]
                        fused_docs = reciprocal_rank_fusion(all_docs)
                        
                        if fused_docs:
                            context = "\n\n".join([f"📄 Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(fused_docs)])
                            template = """You are a helpful AI assistant. Answer the question based only on the context below.
                                - If you cannot answer from the context, say EXACTLY: "I couldn't find this information in the documents."
                                - Use bullet points for lists, steps, or comparisons.
                                - Do NOT include any headings or intro text.
                                - Stick strictly to the provided context.
                                Context:
                                {context}

                                Question: {question}

                                Answer:"""
                            prompt_template = ChatPromptTemplate.from_template(template)
                            rag_chain = (
                                {"context": RunnableLambda(lambda x: context), "question": RunnablePassthrough()}
                                | prompt_template
                                | gemini_llm
                                | StrOutputParser()
                            )
                            answer = rag_chain.invoke(prompt)
                            
                            # Check if the answer indicates missing information
                            if "couldn't find this information in the documents" not in answer.lower():
                                response = f"📄 Document Answer:\n\n{answer}"
                                document_answer_found = True
                    
                    # Handle cases where answer wasn't found
                    if not document_answer_found:
                        if use_web_search:
                            raw_web_results, formatted_web_results = search_google(prompt)
                            if raw_web_results and "error" not in str(formatted_web_results).lower():
                                # Process web results with Gemini
                                web_template = """Analyze these web search results and provide a comprehensive answer:
                                - Combine information from multiple sources if needed
                                - Always include source references like [1], [2] with corresponding links
                                - Keep the answer concise but informative
                                
                                Search Results:
                                {results}
                                
                                Question: {question}
                                
                                Answer with inline citations:"""
                                web_prompt = ChatPromptTemplate.from_template(web_template)
                                web_chain = (
                                    {"results": RunnablePassthrough(), "question": RunnablePassthrough()}
                                    | web_prompt
                                    | gemini_llm
                                    | StrOutputParser()
                                )
                                
                                processed_results = web_chain.invoke({
                                    "results": formatted_web_results, 
                                    "question": prompt
                                })
                                
                                # Add numbered source list
                                sources_section = "\n\n### References:\n" + "\n".join(
                                    f"{i+1}. [{res.get('title', 'Source')}]({res.get('link', '')})"
                                    for i, res in enumerate(raw_web_results)
                                )
                                
                                response = (
                                    f"🌐 Web Answer (since not found in documents):\n\n"
                                    f"{processed_results}\n"
                                    f"{sources_section}"
                                )
                            else:
                                response = "⚠️ Couldn't find relevant information in documents or through web search."
                        else:
                            response = (
                                "I couldn't find this information in your documents. "
                                "You can enable web search in the sidebar to search online for answers."
                            )
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

tabs = st.tabs(["📄 Chat", "📘 Flashcards"])

with tabs[1]:  # Flashcard Tab
    from components.flashcards import flashcard_ui
    flashcard_ui()

# Notes Section
if st.session_state.show_notes:
    with st.expander("📝 Notes Editor", expanded=True):
        temp_editor_content = st_quill(
            key="quill_editor",
            value=st.session_state.rich_notes,
            placeholder="Write your notes here...",
            html=True
        )

        if st.button("💾 Save Notes"):
            st.session_state.rich_notes = temp_editor_content
            st.success("Notes saved!")

    with st.expander("🔍 Preview Notes", expanded=False):
        st.markdown("### Rendered Notes:")
        st.markdown(st.session_state.rich_notes or "*No notes yet.*", unsafe_allow_html=True)

    # PDF Download Section
    if st.button("💾 Download Notes as PDF"):
        if st.session_state.rich_notes:
            soup = BeautifulSoup(st.session_state.rich_notes, "html.parser")
            clean_text = soup.get_text()

            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)

            for line in clean_text.split("\n"):
                pdf.multi_cell(0, 10, line)

            pdf_buffer = io.BytesIO()
            pdf_string = pdf.output(dest='S')
            pdf_buffer = BytesIO()
            pdf_buffer.write(pdf_string)
            pdf_buffer.seek(0)

            st.download_button(
                label="📄 Confirm Download",
                data=pdf_buffer,
                file_name="my_notes.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("No notes to download!, Save first")