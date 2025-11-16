
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List
# Import the custom functions and constants from our pipeline file
from rag_pipeline import load_rag_pipeline, ask_dishdash


# 1. Defines the expected input body for the API
class QueryRequest(BaseModel):
    query: str # This holds the user's input question

# 2. Defines the structure for a single place recommendation
class RecommendedPlace(BaseModel):
    id: str   # The unique _id from the original data/ChromaDB
    name: str # The Farsi name of the place

# 3. Defines the overall structure for the API's response
class QueryResponse(BaseModel):
    llm_response: str # The friendly, human-readable text from the LLM
    recommendations: List[RecommendedPlace]

# --- 1. Global State Management ---
# The loaded RAG components will be stored here after startup.
class RAGState:
    model = None
    tokenizer = None
    embedding_model = None
    collection = None

# --- 2. Pydantic Models for Guaranteed Output Structure ---
# Defines the structure for a single place recommendation
class RecommendedPlace(BaseModel):
    id: str   # The unique _id from the original data/ChromaDB
    name: str # The Farsi name of the place

# Defines the overall structure for the API's response
class QueryResponse(BaseModel):
    llm_response: str # The friendly, human-readable text from the LLM
    recommendations: List[RecommendedPlace] # The structured list of places (GUARANTEED MATCHED)

# --- 3. Startup/Shutdown Events (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads all heavy RAG components when the FastAPI app starts.
    This ensures models and the database are ready before serving requests.
    """
    print("Starting FastAPI application lifespan...")
    try:
        # Load all components using the function from rag_pipeline.py
        RAGState.model, RAGState.tokenizer, RAGState.embedding_model, RAGState.collection = load_rag_pipeline()
        print("✅ RAG System fully initialized.")
    except Exception as e:
        print(f"FATAL ERROR during RAG pipeline startup: {e}")
        # Raise an error to prevent the server from starting with a broken pipeline
        raise RuntimeError("RAG system failed to load. Check GPU/memory status and DB path.") from e

    yield # The application serves requests here

    print("Application shutting down.")

# Initialize the FastAPI App
app = FastAPI(
    title="DishDash RAG API",
    description="Mistral 7B RAG for Tehran Food Recommendations with Farsi-safe detection.",
    version="1.0.0",
    lifespan=lifespan # Attach the loading function
)

# --- 4. Define the API Endpoint ---
@app.post("/recommendation", response_model=QueryResponse)
async def get_recommendation(request: QueryRequest):
    """
    Processes a user query and returns a structured list of recommendations,
    guaranteeing that the IDs match the names mentioned in the LLM's response.
    """
    if not RAGState.model:
        raise HTTPException(status_code=503, detail="RAG model is not yet loaded or failed to initialize.")

    try:
        # Call the core RAG function, passing the loaded components
        result_data = ask_dishdash(
            user_query=request.query,
            model=RAGState.model,
            tokenizer=RAGState.tokenizer,
            embedding_model=RAGState.embedding_model,
            collection=RAGState.collection
        )

        # FastAPI/Pydantic automatically validates and formats the dictionary into the QueryResponse model
        return QueryResponse(
            llm_response=result_data["llm_response"],
            recommendations=result_data["recommended_places"]
        )

    except Exception as e:
        # Log the full exception on the server side
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while generating the response.")

# --- 5. Health Check Endpoint ---
@app.get("/health")
def health_check():
    """Check if the RAG components are loaded and ready."""
    status = "ready" if RAGState.model and RAGState.collection else "loading"
    return {"status": status, "db_items": RAGState.collection.count() if RAGState.collection else 0}