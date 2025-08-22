#Backend uvicorn main:app --reload
from fastapi import FastAPI
from pydantic import BaseModel
import os
import qdrant_client
from datasets import load_dataset
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
import operator
from typing import TypedDict, Annotated, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_tavily import TavilySearch
from nemoguardrails.rails import LLMRails, RailsConfig

from langgraph.graph import StateGraph, END
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

from dotenv import load_dotenv

# --- Load API Keys ---
load_dotenv()

# The keys are now loaded into your environment. The rest of the script
# will find them using os.environ.get("GOOGLE_API_KEY") etc.

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# --- Guardrails Setup ---
config = RailsConfig.from_path("guardrails")
rails = LLMRails(config=config, llm=llm)

# --- Knowledge Base Setup ---
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

# --- Agentic Workflow ---
class AgentState(TypedDict):
    question: str
    documents: List[dict]
    generation: str
    source: str
    grade: str # Added for clarity in state transitions

web_search_tool = TavilySearch(k=3)

def retrieve_from_kb(state):
    question = state["question"]
    docs = retriever.invoke(question)
    documents = [{"source": "KB", "content": doc.page_content} for doc in docs]
    return {"documents": documents, "source": "KB", "question": question}

def web_search(state):
    question = state["question"]
    search_results = web_search_tool.invoke({"query": question})
    
    mcp_formatted_docs = []
    if isinstance(search_results, list):
        for res in search_results:
            mcp_formatted_docs.append({"source": "Web Search", "content": res})
    elif isinstance(search_results, str):
        mcp_formatted_docs.append({"source": "Web Search", "content": search_results})

    return {"documents": mcp_formatted_docs, "source": "Web", "question": question}

def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    source = state["source"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing relevance of a retrieved document to a user question. Respond with 'yes' if relevant, 'no' otherwise."),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
    ])
    if not documents:
        grade = "no"
    else:
        doc_to_grade = documents[0]['content']
        chain = prompt | llm
        result = chain.invoke({"question": question, "document": doc_to_grade})
        grade = result.content.strip().lower()
        if grade not in ["yes", "no"]:
            grade = "no"
    return {"documents": documents, "grade": grade, "source": source, "question": question}

def generate_solution(state):
    question = state["question"]
    documents = state["documents"]
    source = state["source"]
    
    context = ""
    for doc in documents:
        context += f"Source: {doc['source']}\nContent: {doc['content']}\n\n"

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful math professor. Your goal is to provide a clear, step-by-step solution to the user's question.
        Use the following context from your knowledge source ({source}) to answer the question. If the context is empty or not useful, use your own knowledge but state that you are doing so.
        Context:
        {context}
        Question:
        {question}
        Provide your final answer as a step-by-step solution."""
    )
    chain = prompt | llm
    generation = chain.invoke({"context": context, "question": question, "source": source})
    return {"generation": generation.content, "question": question, "documents": documents, "source": source}

def handle_no_solution(state):
    return {"generation": "I'm sorry, but I couldn't find a reliable answer in my knowledge base or through a web search. Please try rephrasing your question."}

def decide_next_step(state):
    if state["grade"] == "yes":
        return "generate"
    else:
        if state["source"] == "KB":
            return "web_search"
        else:
            return "handle_fail"

workflow = StateGraph(AgentState)
workflow.add_node("retrieve_kb", retrieve_from_kb)
workflow.add_node("web_search", web_search)
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
refine_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
refine_prompt = ChatPromptTemplate.from_template(
"""
You are a math professor reviewing and improving a previously generated answer based on student feedback.
Original Question:
{question}
Initial Answer:
{initial_answer}
Student Feedback:
{feedback}
Now revise the solution to improve clarity and correctness. Provide a step-by-step refined answer."""
)
refine_chain = refine_prompt | refine_llm

class Query(BaseModel):
    question: str

class Feedback(BaseModel):
    question: str
    initial_answer: str
    feedback: str

@app.post("/ask")
async def ask_agent(query: Query):
    inputs = {"question": query.question}
    bot_message = await rails.generate_async(prompt=query.question)
    if bot_message:
        # A rail was triggered, so return the result from the rail
        return {"answer": bot_message}
    else:
        # No rail was triggered, so run the main app
        app_result = app_runnable.invoke(inputs)
        return {"answer": app_result['generation']}

@app.post("/refine")
async def refine_answer(feedback: Feedback):
    response = refine_chain.invoke({
        "question": feedback.question,
        "initial_answer": feedback.initial_answer,
        "feedback": feedback.feedback
    })
    refined_answer = response.content
    # Add the refined answer to the knowledge base
    refined_document = f"Question: {feedback.question}\nAnswer: {refined_answer}"
    qdrant_instance.add_texts([refined_document])
    return {"refined_answer": refined_answer}
