
import os
import getpass
from fastapi import FastAPI, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain import hub

global_retriever = None

def _set_env(var: str, value: str = None):
    """Set environment variables for API keys."""
    if not os.environ.get(var):
        if value is not None:
            os.environ[var] = value
        else:
            os.environ[var] = getpass.getpass(f"{var}: ")

# Set API Keys
def set_api_keys():
    """Set environment variables for required API keys."""
    _set_env("TAVILY_API_KEY", "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")
    _set_env("COHERE_API_KEY", "oZBttxNF7QGZZzl15L2p5sv2FN99UzkkgiTMvvuz")
    _set_env("OPENAI_API_KEY", "OPENAI_KEY")

# Load documents from web URLs
def load_documents(urls):
    """Load documents from a list of URLs."""
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list

# Split documents into chunks
def split_documents(docs_list, chunk_size=500, chunk_overlap=0):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs_list)

# Create vector store from documents
def create_vectorstore(doc_splits):
    """Create a vector store from document chunks and return retriever."""

    embd = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=doc_splits, collection_name="rag-chroma", embedding=embd)
    # Create the retriever for future use
    retriever = vectorstore.as_retriever()
    global global_retriever
    global_retriever = retriever  # Store retriever globally
    return vectorstore

# Route query to the appropriate datasource (vectorstore or web search)
class RouteQuery(BaseModel):
    """Data model to route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(..., description="Route to vectorstore or web search.")

def route_query_to_source(question):
    """Route a query to the vectorstore or web search based on content."""
    system = """You are an expert at routing a user question to a vectorstore or web search.
                The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
                Use the vectorstore for questions on these topics. Otherwise, use web-search."""
    route_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_router = llm.with_structured_output(RouteQuery)
    question_router = route_prompt | structured_llm_router
    return question_router.invoke({"question": question})
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def grade_relevance(question, document):
    """Grade the relevance of a document to a user question."""
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
                If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
                It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")])
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader.invoke({"question": question, "document": document})

# Generate an answer to the user question based on the context
def generate_answer(docs, question):
    """Generate an answer from the documents based on the user's question."""
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain.invoke({"context": docs, "question": question})
    
def process_query(urls, question):
    """Complete process for querying the vector store and generating an answer."""
    set_api_keys()

    try:
        # Use the same embedding function used during indexing
        embd = OpenAIEmbeddings()

        if global_retriever is None:
            raise HTTPException(status_code=500, detail="Vector store not initialized. Please index some data first.")
        
        # Route the query to the right source
        route_result = route_query_to_source(question)

        if route_result.datasource == "vectorstore":
            # Use the retriever to find the most relevant documents
            docs = global_retriever.get_relevant_documents(question)
            
            if not docs:  # Check if documents are retrieved
                raise HTTPException(status_code=404, detail="No documents found for the question in vector store.")
            
            doc_txt = docs[0].page_content  # Safely access the first document
            relevance = grade_relevance(question, doc_txt)
            
            if relevance.binary_score.lower() != 'yes':
                return "No relevant document found."
            
            answer = generate_answer(docs, question)
        else:
            answer = "Web search result for your query."

        return answer

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
