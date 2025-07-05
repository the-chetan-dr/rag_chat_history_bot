# RAG QNA CONVERSATIONAL with PDF including chat history
import streamlit as st 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload a PDF and chat with your content.")

api_key = st.text_input("Enter your GROQ API key", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")
    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        documents = []
        temp_pdf = "./temp.pdf"
        with open(temp_pdf, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed, otherwise return as it is. "
            "Crack jokes in between to look cool for every situation."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=contextualize_q_prompt
        )

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say you don't know. "
            "Use as few sentences as possible and keep the answer concise.\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        #using rag
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_message_key="answer"
        )

        user_input = st.text_input("Ask a question about the PDF:")
        if user_input:
           session_history=get_session_history(session_id)
           response=conversational_rag_chain.invoke(
               {"input":user_input},
               config={
                   "configurable":{"session_id":session_id}
                   
               },
           )
           st.write(st.session_state.store)
           st.success(f"Assistant: {response['answer']}")

           st.write("ChatHistory",session_history.messages)
           
else:
    st.warning("GROQ api key kidar chali")           
           
