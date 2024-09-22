import streamlit as st
from langchain_community.llms import Ollama
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaEmbeddings
import tempfile
import os
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore

# Initialize Ollama LLM
llm = Ollama(model="llama3.1:8b")
embeddings = OllamaEmbeddings(
    model="llama3.1:8b"
)

@st.cache_resource
def load_and_split_pdf(pdf_file, chunk_size=3000, chunk_overlap=400):
    """Loads and splits the PDF document."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(documents)
    return splits

@st.cache_resource
def create_and_persist_vectorstore(_splits, file_name):
    """Creates, stores, and loads a vectorstore from document embeddings using Chroma."""
    persist_directory = f"/content/{os.path.splitext(file_name)[0]}.chroma"

    vectorstore = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    store = InMemoryByteStore()
    id_key = f"{os.path.splitext(file_name)[0]}_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
        search_kwargs={"k": 8}
    )

    doc_ids = [str(uuid.uuid4()) for _ in _splits]
    child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    sub_docs = []

    for i, doc in enumerate(_splits):
        _id = doc_ids[i]
        _sub_docs = child_text_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        sub_docs.extend(_sub_docs)

    retriever.vectorstore.add_documents(sub_docs)
    retriever.docstore.mset(list(zip(doc_ids, _splits)))

    return retriever

def setup_contextual_qa_chain(llm, retriever):
    """Sets up the contextual retriever and question-answering chain."""
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([ 
        ("system", contextualize_q_system_prompt), 
        MessagesPlaceholder("chat_history"), 
        ("human", "{input}") 
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([ 
        ("system", system_prompt), 
        MessagesPlaceholder("chat_history"), 
        ("human", "{input}") 
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Returns the chat history for a session, creates if it doesn't exist.""" 
    if session_id not in store: 
        store[session_id] = ChatMessageHistory() 
    return store[session_id]

def setup_conversational_chain(rag_chain):
    """Creates a conversational RAG chain with message history.""" 
    return RunnableWithMessageHistory( 
        rag_chain, 
        get_session_history, 
        input_messages_key="input", 
        history_messages_key="chat_history", 
        output_messages_key="answer" 
    )

def format_history(conversation_history):
    """Formats the conversation history to send as input to the model.""" 
    formatted_history = "" 
    for question, answer in conversation_history: 
        formatted_history += f"User: {question}\nAssistant: {answer}\n" 
    return formatted_history

# Streamlit frontend setup
st.title("Document Chatbot")

# Streamlit sidebar for uploading PDF file
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    session_id = f"session_{os.path.splitext(uploaded_file.name)[0]}_{uuid.uuid4()}"

    with st.spinner('Processing your document...'):
        splits = load_and_split_pdf(uploaded_file)
        retriever = create_and_persist_vectorstore(splits, uploaded_file.name)
        rag_chain = setup_contextual_qa_chain(llm, retriever)
        conversational_rag_chain = setup_conversational_chain(rag_chain)

        st.success(f"Document processed! You can now start chatting.")

        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

        if 'user_input' not in st.session_state:
            st.session_state.user_input = ""

        def clear_input():
            st.session_state.user_input = ""

        # Chatbot input at the bottom
        st.markdown("### Chatbot")
        with st.form(key="input_form", clear_on_submit=True):
            user_input = st.text_input("Ask your question:", value=st.session_state.user_input, key="input_box")
            submit_button = st.form_submit_button("Submit", on_click=clear_input)

        if submit_button and user_input:
            with st.spinner('Retrieving the answer...'):
                chat_history = format_history(st.session_state.conversation_history)
                full_input = f"{chat_history}User: {user_input}\nAssistant:"

                response = conversational_rag_chain.invoke(
                    {"input": full_input},
                    config={"configurable": {"session_id": session_id}}
                )
                answer = response['answer']
                st.session_state.conversation_history.append((user_input, answer))

        for question, answer in st.session_state.conversation_history:
            st.markdown(f"**User:** {question}")
            st.markdown(f"**Assistant:** {answer}")
            st.markdown("---")
