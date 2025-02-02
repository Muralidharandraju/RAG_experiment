import streamlit as st
from pdfminer.high_level import extract_text
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question related to AWS Glue.
 If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = 'utils/temp/'

embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model="deepseek-r1:1.5b")

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def extract_text(text: str) -> str:
    " we have output with resaoning text and this function will remove that"
    index = text.find("</think>")
    if index != -1:
        return text[index + len("</think>"):].lstrip()  # lstrip() avoids extra space scanning
    return ""

def load_pdf(file_path):
    text = extract_text(file_path)
    return text

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    return text_splitter.split_text(text)

def index_docs(documents):
    vector_store.aadd_texts(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="PDF",
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)


    question = st.chat_input()

    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_docs(question)
        answer = answer_question(question, related_documents)
        output= extract_text(answer)
        st.chat_message("assistant").write(output)

