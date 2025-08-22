#Backend uvicorn main:app --reload
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import qdrant_client
from datasets import load_dataset
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
import operator
from typing import TypedDict, Annotated, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from nemoguardrails.rails import LLMRails, RailsConfig
from langgraph.graph import StateGraph, END
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import asyncio
import atexit
import logging

# Import MCP client
from mcp_client import search_with_mcp, cleanup_mcp, get_mcp_client, test_mcp_functionality

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Math Routing Agent with MCP", version="1.0.0")

# --- Load API Keys ---
load_dotenv()

# CORS middleware
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLMs
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# --- Guardrails Setup ---
try:
    config = RailsConfig.from_path("guardrails")
    rails = LLMRails(config=config, llm=llm)
    logger.info("✅ Guardrails initialized successfully")
except Exception as e:
    logger.error(f"❌ Error initializing guardrails: {e}")
    # Create a dummy rails object to prevent crashes
    class DummyRails:
        async def generate_async(self, prompt):
            return None
    rails = DummyRails()

# --- Knowledge Base Setup ---
try:
    dataset = load_dataset("gsm8k", "main", split="train[:100]")
    questions = [item["question"] for item in dataset]
    documents_for_kb = [f"Question: {item['question']}\nAnswer: {item['answer']}" for item in dataset]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    qdrant_instance = Qdrant.from_texts(
        documents_for_kb,
        embeddings,
        location=":memory:",
        collection_name="math_kb",
    )
    retriever = qdrant_instance.as_retriever(search_kwargs={"k": 3})
    logger.info("✅ Knowledge base initialized with GSM8K dataset")
except Exception as e:
    logger.error(f"❌ Error initializing knowledge base: {e}")
    retriever = None

# --- Agentic Workflow ---
class AgentState(TypedDict):
    question: str
    documents: List[dict]
    generation: str
    source: str
    grade: str

def retrieve_from_kb(state):
    """Retrieve documents from knowledge base"""
    question = state["question"]
    if retriever is None:
        return {"documents": [], "source": "KB", "question": question}
    
    try:
        docs = retriever.invoke(question)
        documents = [{"source": "KB", "content": doc.page_content} for doc in docs]
        logger.info(f"Retrieved {len(documents)} documents from KB")
        return {"documents": documents, "source": "KB", "question": question}
    except Exception as e:
        logger.error(f"Error retrieving from KB: {e}")
        return {"documents": [], "source": "KB", "question": question}

async def web_search_mcp(state):
    """Updated web search function using MCP"""
    question = state["question"]
    
    try:
        logger.info(f"Performing MCP web search for: {question}")
        # Use MCP for web search
        search_results = await search_with_mcp(question, max_results=3)
        logger.info(f"MCP search returned {len(search_results)} results")
        return {"documents": search_results, "source": "Web", "question": question}
        
    except Exception as e:
        logger.error(f"MCP search error: {e}")
        # Fallback error handling
        error_doc = [{"source": "MCP Error", "content": f"Web search failed: {str(e)}"}]
        return {"documents": error_doc, "source": "Web", "question": question}

def grade_documents(state):
    """Grade the relevance of retrieved documents"""
    question = state["question"]
    documents = state["documents"]
    source = state["source"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing relevance of a retrieved document to a user question. " +
                  "If the document contains mathematical information relevant to the question, respond with 'yes'. " +
                  "If the document is not relevant or contains no useful information, respond with 'no'."),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
    ])
    
    if not documents or not documents[0].get('content'):
        grade = "no"
        logger.info("No documents to grade - graded as 'no'")
    else:
        try:
            doc_to_grade = documents[0]['content'][:1000]  # Limit content length
            chain = prompt | llm
            result = chain.invoke({"question": question, "document": doc_to_grade})
            grade = result.content.strip().lower()
            
            # Ensure grade is valid
            if grade not in ["yes", "no"]:
                grade = "no"
            
            logger.info(f"Document graded as: {grade}")
        except Exception as e:
            logger.error(f"Error grading documents: {e}")
            grade = "no"
    
    return {"documents": documents, "grade": grade, "source": source, "question": question}

def generate_solution(state):
    """Generate step-by-step solution"""
    question = state["question"]
    documents = state["documents"]
    source = state["source"]
    
    # Prepare context from documents
    context = ""
    for doc in documents[:3]:  # Limit to 3 documents
        if doc.get('content'):
            context += f"Source: {doc['source']}\nContent: {doc['content'][:500]}...\n\n"

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful math professor. Your goal is to provide a clear, step-by-step solution to the user's question.
        
        Use the following context from your knowledge source ({source}) to answer the question. 
        If the context is empty or not useful, use your mathematical knowledge but state that you are doing so.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Please provide your answer as a clear, step-by-step solution with explanations for each step.
        Use mathematical notation where appropriate and make sure your explanation is easy to understand."""
    )
    
    try:
        chain = prompt | llm
        generation = chain.invoke({
            "context": context if context.strip() else "No relevant context found.", 
            "question": question, 
            "source": source
        })
        logger.info("Solution generated successfully")
        return {
            "generation": generation.content, 
            "question": question, 
            "documents": documents, 
            "source": source
        }
    except Exception as e:
        logger.error(f"Error generating solution: {e}")
        return {
            "generation": f"I encountered an error while generating the solution: {str(e)}", 
            "question": question, 
            "documents": documents, 
            "source": source
        }

def handle_no_solution(state):
    """Handle cases where no solution can be found"""
    return {
        "generation": "I'm sorry, but I couldn't find a reliable answer in my knowledge base or through a web search. " +
                     "Please try rephrasing your question or asking about a different mathematical topic.",
        "question": state["question"],
        "documents": state.get("documents", []),
        "source": state.get("source", "Unknown")
    }

def decide_next_step(state):
    """Decide the next step based on document grading"""
    grade = state.get("grade", "no")
    source = state.get("source", "Unknown")
    
    if grade == "yes":
        return "generate"
    else:
        if source == "KB":
            return "web_search"
        else:
            return "handle_fail"

# Create workflow
workflow = StateGraph(AgentState)
workflow.add_node("retrieve_kb", retrieve_from_kb)
workflow.add_node("web_search", lambda state: asyncio.run(web_search_mcp(state)))
workflow.add_node("grade_docs", grade_documents)
workflow.add_node("generate", generate_solution)
workflow.add_node("handle_fail", handle_no_solution)

workflow.set_entry_point("retrieve_kb")
workflow.add_edge("retrieve_kb", "grade_docs")
workflow.add_conditional_edges(
    "grade_docs",
    decide_next_step,
    {
        "web_search": "web_search",
        "generate": "generate",
        "handle_fail": "handle_fail",
    },
)
workflow.add_edge("web_search", "grade_docs")
workflow.add_edge("generate", END)
workflow.add_edge("handle_fail", END)

app_runnable = workflow.compile()

# --- Feedback Mechanism ---
refine_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
refine_prompt = ChatPromptTemplate.from_template(
    """You are a math professor reviewing and improving a previously generated answer based on student feedback.
    
    Original Question:
    {question}
    
    Initial Answer:
    {initial_answer}
    
    Student Feedback:
    {feedback}
    
    Please revise the solution to address the feedback and improve clarity and correctness. 
    Provide a step-by-step refined answer that incorporates the student's concerns."""
)
refine_chain = refine_prompt | refine_llm

# --- API Models ---
class Query(BaseModel):
    question: str

class Feedback(BaseModel):
    question: str
    initial_answer: str
    feedback: str

class HealthResponse(BaseModel):
    status: str
    message: str
    mcp_status: str

# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test MCP functionality
        client = await get_mcp_client()
        mcp_status = "healthy" if client else "unavailable"
        
        return HealthResponse(
            status="healthy",
            message="Math Routing Agent is running",
            mcp_status=mcp_status
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            message=f"Service running with issues: {str(e)}",
            mcp_status="error"
        )

@app.post("/ask")
async def ask_agent(query: Query):
    """Main endpoint for asking mathematical questions"""
    try:
        inputs = {"question": query.question}
        
        # Check guardrails first
        bot_message = await rails.generate_async(prompt=query.question)
        if bot_message:
            # A rail was triggered, so return the result from the rail
            logger.info("Guardrail triggered")
            return {"answer": bot_message, "source": "guardrails"}
        else:
            # No rail was triggered, so run the main app
            logger.info("Running main workflow")
            app_result = app_runnable.invoke(inputs)
            return {
                "answer": app_result.get('generation', 'No answer generated'), 
                "source": app_result.get('source', 'Unknown')
            }
    
    except Exception as e:
        logger.error(f"Error in ask_agent: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/refine")
async def refine_answer(feedback: Feedback):
    """Endpoint for refining answers based on feedback"""
    try:
        response = refine_chain.invoke({
            "question": feedback.question,
            "initial_answer": feedback.initial_answer,
            "feedback": feedback.feedback
        })
        refined_answer = response.content
        
        # Add the refined answer to the knowledge base if available
        if qdrant_instance:
            try:
                refined_document = f"Question: {feedback.question}\nAnswer: {refined_answer}"
                qdrant_instance.add_texts([refined_document])
                logger.info("Refined answer added to knowledge base")
            except Exception as e:
                logger.error(f"Failed to add refined answer to KB: {e}")
        
        return {"refined_answer": refined_answer}
        
    except Exception as e:
        logger.error(f"Error in refine_answer: {e}")
        raise HTTPException(status_code=500, detail=f"Error refining answer: {str(e)}")

@app.get("/mcp/status")
async def mcp_status():
    """Get MCP server status"""
    try:
        client = await get_mcp_client()
        tools = await client.list_tools()
        resources = await client.list_resources()
        
        return {
            "status": "active",
            "servers": list(client.servers.keys()),
            "tools": {server: [tool.name for tool in tool_list] for server, tool_list in tools.items()},
            "resources": {server: [res.name for res in res_list] for server, res_list in resources.items()}
        }
    except Exception as e:
        logger.error(f"Error getting MCP status: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/mcp/test")
async def test_mcp():
    """Test MCP functionality"""
    try:
        success = await test_mcp_functionality()
        return {"success": success, "message": "MCP test completed"}
    except Exception as e:
        logger.error(f"Error testing MCP: {e}")
        return {"success": False, "message": str(e)}

# --- Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    """Initialize MCP and other services on startup"""
    logger.info("🚀 Starting Math Routing Agent...")
    
    # Test MCP functionality
    try:
        await test_mcp_functionality()
        logger.info("✅ MCP initialization successful")
    except Exception as e:
        logger.error(f"❌ MCP initialization failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("🛑 Shutting down Math Routing Agent...")
    await cleanup_mcp()
    logger.info("✅ Cleanup completed")

# Register atexit handler as backup
atexit.register(lambda: asyncio.run(cleanup_mcp()))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)