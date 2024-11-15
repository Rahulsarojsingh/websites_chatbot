# RAG System with FastAPI

This project sets up a Retrieval-Augmented Generation (RAG) system that uses the `langchain` library along with various APIs like **Tavily**, **Cohere**, and **OpenAI** to process data and generate responses.

## Prerequisites

Before running the application, you need to export the required API keys for Tavily, Cohere, and OpenAI. These keys are necessary for accessing external services and generating embeddings, routing queries, and processing natural language tasks.

### **API Keys**

You can export the API keys as environment variables before running the application. Here's how to set them:

1. **Tavily API Key**
   - Key: `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`
   
2. **Cohere API Key**
   - Key: `oZBttxNF7QGZZzl15L2p5sv2FN99UzkkgiTMvvuz`
   
3. **OpenAI API Key**
   - Key: 

---

### **Steps to Run the Application**

1. **Create a `.env` file**

   Create a `.env` file in the root directory of your project and add the following contents:

   ```plaintext
   TAVILY_API_KEY=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
   COHERE_API_KEY=oZBttxNF7QGZZzl15L2p5sv2FN99UzkkgiTMvvuz
   OPENAI_API_KEY=?

