import os
from typing import TypedDict, List, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# --- Environment Configuration ---
# This makes the code portable across Dev, Stage, and Prod.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


# Securely grab the key
qdrant_key = os.getenv("QDRANT_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = QdrantVectorStore(
    client=QdrantClient(url=QDRANT_URL, api_key=qdrant_key), # Passed securely here
    collection_name="ecommerce_catalog",
    embedding=embeddings,
)

# 1. Setup Data & LLM Dependencies
print(f"Initializing LLM at {OLLAMA_URL} and Vector DB at {QDRANT_URL}...")
llm = ChatOllama(model="llama3", temperature=0, base_url=OLLAMA_URL) # Explicitly pass the URL
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = QdrantVectorStore(
    client=QdrantClient(url=QDRANT_URL),
    collection_name="ecommerce_catalog",
    embedding=embeddings,
)

# 2. Define the Agent State
class AgentState(TypedDict):
    messages: List[HumanMessage | SystemMessage | AIMessage]
    next_step: str
    context: str

# 3. Define the Nodes (The "Agents")
def supervisor_node(state: AgentState) -> AgentState:
    """Decides if the user is asking about products or just chatting."""
    latest_message = state["messages"][-1].content
    
    routing_prompt = f"""You are a routing supervisor for an e-commerce store.
    Read the user's message and determine their intent.
    If they are looking for a product, item, or shopping recommendation, reply strictly with the word: PRODUCT_SEARCH.
    If it is a general greeting or unrelated chat, reply strictly with the word: GENERAL.
    
    User Message: {latest_message}
    Decision:"""
    
    response = llm.invoke([HumanMessage(content=routing_prompt)])
    decision = response.content.strip().upper()
    
    if "PRODUCT_SEARCH" in decision:
        return {"next_step": "catalog_agent"}
    else:
        return {"next_step": "general_agent"}

def catalog_agent_node(state: AgentState) -> AgentState:
    """Searches Qdrant and answers product questions."""
    user_query = state["messages"][-1].content
    
    print(f"\n[System: Catalog Agent searching Vector DB for '{user_query}'...]")
    results = vector_store.similarity_search(user_query, k=2)
    
    context = "\n".join([f"- {doc.metadata['name']} (${doc.metadata['price']}): {doc.page_content}" for doc in results])
    
    answer_prompt = f"""You are a helpful e-commerce assistant. Use the following product catalog information to answer the user's request. 
    Catalog Context:
    {context}
    
    User Request: {user_query}
    """
    response = llm.invoke([SystemMessage(content=answer_prompt)])
    
    state["messages"].append(AIMessage(content=response.content))
    return state

def general_agent_node(state: AgentState) -> AgentState:
    """Handles basic greetings without querying the database."""
    print("\n[System: General Agent handling standard chat...]")
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state

# 4. Define the Routing Logic (Conditional Edges)
def router(state: AgentState) -> Literal["catalog_agent", "general_agent"]:
    return state["next_step"]

# 5. Build and Compile the Graph
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("catalog_agent", catalog_agent_node)
workflow.add_node("general_agent", general_agent_node)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", router)
workflow.add_edge("catalog_agent", END)
workflow.add_edge("general_agent", END)

app = workflow.compile()

# 6. Let's run a test!
if __name__ == "__main__":
    print("\n--- Testing the Multi-Agent E-Commerce Workflow ---")
    
    test_1 = {"messages": [HumanMessage(content="Hello! How are you today?")], "next_step": "", "context": ""}
    print("\nUser: Hello! How are you today?")
    result_1 = app.invoke(test_1)
    print(f"AI: {result_1['messages'][-1].content}")
    
    test_2 = {"messages": [HumanMessage(content="I'm looking for a good screen for gaming, any suggestions?")], "next_step": "", "context": ""}
    print("\nUser: I'm looking for a good screen for gaming, any suggestions?")
    result_2 = app.invoke(test_2)
    print(f"AI: {result_2['messages'][-1].content}")