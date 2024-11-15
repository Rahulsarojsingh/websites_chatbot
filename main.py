from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from rag import process_query, load_documents, split_documents, create_vectorstore

app = FastAPI()

# Define request model for indexing
class IndexRequest(BaseModel):
    url: List[str]

# Define request model for chat
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

# Endpoint to index content from URLs
@app.post("/api/v1/index")
async def index_data(request: IndexRequest):
    """Endpoint to process the indexing of URLs."""
    try:
        docs_list = load_documents(request.url)
        doc_splits = split_documents(docs_list)
        
        # Create a vectorstore to index documents
        vectorstore = create_vectorstore(doc_splits)
        
        return {
            "status": "success",
            "indexed_url": request.url,
            "failed_url": None  # Assuming no errors; handle failures as needed
        }
    except Exception as e:
        return {
            "status": "failed",
            "indexed_url": [],
            "failed_url": request.url,
            "error": str(e)
        }

# Endpoint to chat with the RAG system
@app.post("/api/v1/chat")
async def chat_data(request: ChatRequest):
    """Endpoint to process the chat query and return the answer with citations."""
    try:
        question = request.messages[-1]["content"]

        # Process the query and retrieve the answer with citations
        answer = process_query([],question)  # Assuming vectorstore-based retrieval
        
        return {
            "response": {
                "answer": {
                    "content": answer,
                    "role": "assistant"
                },
                "citation": ["link_to_source"]  # Update with actual citation link
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
