import streamlit as st
import os
import pandas as pd
import tempfile
from urllib.parse import urlparse
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader, WebBaseLoader, UnstructuredExcelLoader
)
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from typing import List, Dict, Any, TypedDict, Annotated, Literal, Optional, Union

from bs4 import BeautifulSoup
import requests
from googlesearch import search
import uuid
import datetime

class RAGState(TypedDict):
    question: str
    conversation_id: str
    retrieval_documents: List[Document]
    generation_documents: List[Document]
    additional_context: List[str]
    history: List[Dict[str, str]]
    context_for_generation: str
    additional_context_for_generation: str
    history_for_generation: str
    document_sources_for_generation: str
    relevant_info_found: bool


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if "documents" not in st.session_state:
    st.session_state.documents = []
if "history" not in st.session_state:
    st.session_state.history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "document_sources" not in st.session_state:
    st.session_state.document_sources = []
if "last_activity" not in st.session_state:
    st.session_state.last_activity = datetime.datetime.now()
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

def initialize_rag_model():
    if not st.session_state.openai_api_key:
        st.error("OpenAI API key not provided. Please enter your API key in the sidebar.")
        st.stop()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True, openai_api_key=st.session_state.openai_api_key)

    retrieval_prompt = ChatPromptTemplate.from_template(
        """You are an information retrieval assistant.
        Based on the user's question and conversation history, create an optimized search query to find the most relevant information.

        Previous conversation:
        {history}

        User's question: {question}

        Optimized search query:"""
    )

    generation_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant that maintains context throughout conversations.

        Retrieved context: {context}

        Additional information from web search: {additional_context}

        Previous conversation history:
        {history}

        Available document sources: {document_sources}

        Question: {question}

        Instructions:
        1. If the answer can be found in the provided context or additional information, respond with the relevant information.
        2. If referring to previously uploaded documents, be specific about which documents contain the information.
        3. Maintain awareness of all documents that have been uploaded throughout the conversation.
        4. If the answer cannot be found in the provided context or additional information, clearly state that you don't have enough information to answer this question and suggest what information might help.
        5. If the question relates to previous parts of the conversation, make sure to maintain continuity.

        Answer:"""
    )

    def retrieve(state: RAGState) -> RAGState:
        print("---RETRIEVING DOCUMENTS---")
        vectorstore = st.session_state.vectorstore
        if vectorstore is None:
            print("Vector store not initialized. Skipping retrieval.")
            state["retrieval_documents"] = []
            state["generation_documents"] = []
            state["relevant_info_found"] = False
            return state

        question = state["question"]
        history = state["history"]

        history_str = ""
        if history:
            recent_history = history[-5:]
            history_str = "\n".join([
                f"User: {exchange['user']}\nAssistant: {exchange['assistant']}"
                for exchange in recent_history
            ])

        query_generation_chain = retrieval_prompt | llm | StrOutputParser()
        try:
            optimized_query = query_generation_chain.invoke({
                "question": question,
                "history": history_str
            })
            print(f"Optimized Retrieval Query: {optimized_query}")
        except Exception as e:
            print(f"Error generating optimized query: {e}")
            optimized_query = question


        try:
            results = vectorstore.similarity_search(optimized_query, k=5)
            state["retrieval_documents"] = results
            state["generation_documents"] = results
            print(f"Retrieved {len(results)} documents.")
        except Exception as e:
            print(f"Error during vector store retrieval: {e}")
            state["retrieval_documents"] = []
            state["generation_documents"] = []

        state["relevant_info_found"] = bool(state["retrieval_documents"])

        return state

    def web_search(state: RAGState) -> RAGState:
        print("---PERFORMING WEB SEARCH---")
        question = state["question"]
        additional_context = []

        try:
            search_results = list(search(question, num_results=3, timeout=5))
            print(f"Web search results: {search_results}")

            for url in search_results:
                try:
                    response = requests.get(url, timeout=5)
                    soup = BeautifulSoup(response.text, 'html.parser')

                    title = soup.title.string if soup.title else "No title"

                    main_content = soup.find('main')
                    if not main_content:
                        main_content = soup.find('article')
                    if not main_content:
                        main_content = soup.body

                    text = main_content.get_text(" ", strip=True) if main_content else ""

                    summary = f"From {url} - {title}: {text[:1000]}..."
                    additional_context.append(summary)

                except requests.exceptions.RequestException as e:
                    print(f"Error retrieving content from {url}: {e}")
                    continue
                except Exception as e:
                     print(f"Error parsing content from {url}: {e}")
                     continue

        except Exception as e:
            print(f"Web search error: {e}")

        state["additional_context"] = additional_context
        print(f"Collected {len(additional_context)} pieces of additional context.")

        if state["additional_context"]:
             state["relevant_info_found"] = True

        return state


    def prepare_generation_inputs(state: RAGState) -> RAGState:
        print("---PREPARING GENERATION INPUTS---")
        if state["generation_documents"]:
            context = format_docs(state["generation_documents"])
        else:
            context = "No relevant documents found."

        if state["additional_context"]:
            additional_context = "\n\n".join(state["additional_context"])
        else:
            additional_context = "No additional information available from web search."

        history_str = ""
        if state["history"]:
            history_str = "\n".join([
                f"User: {exchange['user']}\nAssistant: {exchange['assistant']}"
                for exchange in state["history"][-8:]
            ])

        document_sources_str = "No documents have been uploaded."
        if st.session_state.document_sources:
            document_sources_str = "Documents available: " + ", ".join(st.session_state.document_sources)

        state["context_for_generation"] = context
        state["additional_context_for_generation"] = additional_context
        state["history_for_generation"] = history_str
        state["document_sources_for_generation"] = document_sources_str

        return state

    def should_web_search(state: RAGState) -> Literal["web_search", "prepare_generation_inputs"]:
        print("---DECIDING ON WEB SEARCH---")
        question = state["question"].lower()
        search_keywords = ["latest", "recent", "current", "news", "today", "update"]

        if (st.session_state.vectorstore is None or not state["retrieval_documents"] or
            any(keyword in question for keyword in search_keywords)):
            print("Decision: Proceeding to web search.")
            return "web_search"
        else:
            print("Decision: Skipping web search, proceeding to generate preparation.")
            return "prepare_generation_inputs"


    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("web_search", web_search)
    workflow.add_node("prepare_generation_inputs", prepare_generation_inputs)

    workflow.add_conditional_edges(
        "retrieve",
        should_web_search,
        {
            "web_search": "web_search",
            "prepare_generation_inputs": "prepare_generation_inputs"
        }
    )
    workflow.add_edge("web_search", "prepare_generation_inputs")
    workflow.add_edge("prepare_generation_inputs", END)

    workflow.set_entry_point("retrieve")

    rag_chain = workflow.compile()

    return rag_chain, generation_prompt

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def process_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    docs = []
    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension == 'pdf':
            loader = PyPDFLoader(tmp_path)
        elif file_extension == 'txt':
            loader = TextLoader(tmp_path)
        elif file_extension == 'csv':
            loader = CSVLoader(tmp_path)
        elif file_extension in ['xlsx', 'xls']:
            loader = UnstructuredExcelLoader(tmp_path)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return []

        docs = loader.load()

        for doc in docs:
            if "source" not in doc.metadata:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["upload_timestamp"] = datetime.datetime.now().isoformat()
                doc.metadata["conversation_id"] = st.session_state.conversation_id


        if uploaded_file.name not in st.session_state.document_sources:
            st.session_state.document_sources.append(uploaded_file.name)


        return docs

    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return []

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def process_url(url):
    docs = []
    try:
        loader = WebBaseLoader(web_paths=[url], requests_kwargs={"timeout": 10})
        docs = loader.load()

        source_name = url
        for doc in docs:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["source"] = source_name
            doc.metadata["url"] = url
            doc.metadata["upload_timestamp"] = datetime.datetime.now().isoformat()
            doc.metadata["conversation_id"] = st.session_state.conversation_id

        source_entry = f"Web content from {source_name}"
        if source_entry not in st.session_state.document_sources:
            st.session_state.document_sources.append(source_entry)


        return docs
    except Exception as e:
        st.error(f"Error loading content from {url}: {str(e)}")
        return []


def check_session_timeout():
    timeout_minutes = 60
    current_time = datetime.datetime.now()
    time_diff = current_time - st.session_state.last_activity

    if time_diff.total_seconds() > (timeout_minutes * 60):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.history = []
        st.warning(f"Session reset due to {timeout_minutes} minutes of inactivity. Previous conversation history has been cleared.")

    st.session_state.last_activity = current_time


def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🧠", layout="wide")

    check_session_timeout()

    with st.sidebar:
        st.subheader("OpenAI API Key")
        api_key_input = st.text_input("Enter your OpenAI API Key", type="password", value=st.session_state.openai_api_key)
        if api_key_input:
            st.session_state.openai_api_key = api_key_input
            os.environ["OPENAI_API_KEY"] = api_key_input
        else:
            st.warning("Please enter your OpenAI API key to use the chatbot.")
            st.stop()

        st.subheader("Add to Knowledge Base")

        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, CSV, Excel)",
            accept_multiple_files=True,
            type=["pdf", "txt", "csv", "xlsx", "xls"]
        )

        urls_input = st.text_input("Or enter URLs to analyze (comma-separated):")

        if st.button("Process Inputs"):
            with st.spinner("Processing your inputs..."):
                new_docs = []
                processed_sources = []

                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        docs = process_file(uploaded_file)
                        if docs:
                            st.success(f"Processed {uploaded_file.name}")
                            new_docs.extend(docs)
                            processed_sources.append(uploaded_file.name)


                if urls_input:
                    urls = [url.strip() for url in urls_input.split(',') if url.strip()]
                    for url in urls:
                        if is_valid_url(url):
                            url_docs = process_url(url)
                            if url_docs:
                                st.success(f"Processed content from {url}")
                                new_docs.extend(url_docs)
                                processed_sources.append(f"Web content from {url}")
                        else:
                            st.warning(f"Skipping invalid URL: {url}")


                if new_docs:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = text_splitter.split_documents(new_docs)

                    st.session_state.documents.extend(chunks)

                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = FAISS.from_documents(
                            chunks,
                            OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
                        )
                        st.success(f"Created new knowledge base with {len(chunks)} document chunks.")
                    else:
                        st.session_state.vectorstore.add_documents(chunks)
                        st.success(f"Added {len(chunks)} document chunks to existing knowledge base.")

                    if processed_sources:
                         sources_list_str = ", ".join(processed_sources)
                         system_message = {
                             "user": f"[System] Added sources: {sources_list_str}",
                             "assistant": f"I've processed the following sources and added them to my knowledge base: {sources_list_str}. You can now ask questions about this information."
                         }
                         st.session_state.history.append(system_message)
                         st.session_state.last_activity = datetime.datetime.now()
                elif urls_input or uploaded_files:
                     st.info("No new documents were extracted or added from the provided inputs.")


        st.divider()

        st.subheader("Knowledge Base Stats")
        num_chunks = len(st.session_state.documents) if st.session_state.documents else 0
        st.write(f"Document Chunks: {num_chunks}")
        st.write(f"Sources: {len(st.session_state.document_sources)}")
        st.write(f"Conversation length: {len(st.session_state.history)} exchanges")

        st.divider()

        st.subheader("Document Sources Loaded")
        if st.session_state.document_sources:
            for idx, source in enumerate(st.session_state.document_sources):
                st.write(f"{idx+1}. {source}")
        else:
            st.write("No documents or web content loaded yet.")

        st.divider()

        if st.button("Clear Knowledge Base"):
            st.session_state.documents = []
            st.session_state.vectorstore = None
            st.session_state.document_sources = []
            st.success("Knowledge base cleared!")
            st.session_state.history = []
            st.session_state.conversation_id = str(uuid.uuid4())


        if st.button("Clear Chat History"):
            st.session_state.history = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.success("Chat history cleared!")


        st.divider()

        st.subheader("About")
        st.markdown(
            """
            This enhanced RAG (Retrieval-Augmented Generation) chatbot uses:
            - LangGraph for the workflow
            - OpenAI for embeddings and generation (requires your API key)
            - FAISS for vector storage
            - Web search for real-time information
            - Conversation history management to maintain context
            - Handles PDF, TXT, CSV, and Excel files, plus web URLs.
            """
        )


    if "rag_chain" not in st.session_state or "generation_prompt" not in st.session_state:
        if st.session_state.openai_api_key:
            st.session_state.rag_chain, st.session_state.generation_prompt = initialize_rag_model()
            print("LangGraph RAG chain and generation prompt initialized.")
        else:
            st.info("Please enter your OpenAI API key in the sidebar to initialize the chatbot.")
            st.stop()


    chat_container = st.container()

    with chat_container:
        for exchange in st.session_state.history:
            user_msg = exchange["user"]
            if user_msg.startswith("[System]"):
                 with st.chat_message("system"):
                     st.write(exchange["assistant"])
            else:
                with st.chat_message("user"):
                    st.write(user_msg)

                with st.chat_message("assistant"):
                    st.write(exchange["assistant"])

    user_question = st.chat_input("Ask a question about your documents or any topic...")

    if user_question:
        st.session_state.last_activity = datetime.datetime.now()

        st.session_state.history.append({"user": user_question, "assistant": ""})

        with chat_container:
             with st.chat_message("user"):
                 st.write(user_question)

             with st.chat_message("assistant"):
                 message_placeholder = st.empty()
                 message_placeholder.text("Thinking...")


        initial_graph_state = {
            "question": user_question,
            "conversation_id": st.session_state.conversation_id,
            "retrieval_documents": [],
            "generation_documents": [],
            "additional_context": [],
            "history": st.session_state.history[:-1],
            "context_for_generation": "",
            "additional_context_for_generation": "",
            "history_for_generation": "",
            "document_sources_for_generation": "",
            "relevant_info_found": False
        }

        try:
            rag_chain = st.session_state.rag_chain
            final_graph_state = rag_chain.invoke(initial_graph_state)

            context = final_graph_state.get("context_for_generation", "No relevant documents found.")
            additional_context = final_graph_state.get("additional_context_for_generation", "No additional information available from web search.")
            history_str = final_graph_state.get("history_for_generation", "")
            document_sources_str = final_graph_state.get("document_sources_for_generation", "No documents have been uploaded.")
            relevant_info_found = final_graph_state.get("relevant_info_found", False)

            generation_prompt = st.session_state.generation_prompt
            final_prompt_value = generation_prompt.format(
                context=context,
                additional_context=additional_context,
                history=history_str,
                document_sources=document_sources_str,
                question=user_question
            )

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True, openai_api_key=st.session_state.openai_api_key)
            full_response = ""
            for chunk in llm.stream(final_prompt_value):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            if not relevant_info_found:
                 if len(full_response.split()) < 50 or "don't have enough information" in full_response or "could not generate a response" in full_response:
                      full_response = "I couldn't extract relevant information from the provided input. Please upload a relevant document or link, or shall I proceed with a general response?"
                      message_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"An error occurred: {e}"
            message_placeholder.markdown(full_response)
            st.error(full_response)


        st.session_state.history[-1]["assistant"] = full_response

        if user_question:
            st.rerun()


if __name__ == "__main__":
    main()

