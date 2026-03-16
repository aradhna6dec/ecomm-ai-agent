from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

# 1. Initialize the Embedding Model (Runs locally on CPU)
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Connect to Qdrant (via your kubectl port-forward)
print("Connecting to Qdrant...")
qdrant_url = "http://localhost:6333"
collection_name = "ecommerce_catalog"

client = QdrantClient(url=qdrant_url)

# Recreate the collection if it exists to ensure a clean slate
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE), # 384 is the vector size for MiniLM
)

# 3. Create Dummy E-commerce Data
dummy_products = [
    {
        "id": "SKU-1001",
        "name": "UltraWide Gaming Monitor 34-inch",
        "description": "34-inch curved ultrawide gaming monitor with 144Hz refresh rate, 1ms response time, and adaptive sync technology. Perfect for immersive gaming and multitasking.",
        "category": "Electronics",
        "price": 499.99
    },
    {
        "id": "SKU-1002",
        "name": "Mechanical Keyboard RGB",
        "description": "Tenkeyless mechanical gaming keyboard with tactile blue switches, customizable per-key RGB lighting, and aircraft-grade aluminum frame.",
        "category": "Accessories",
        "price": 89.50
    },
    {
        "id": "SKU-1003",
        "name": "Wireless Noise Cancelling Headphones",
        "description": "Over-ear bluetooth headphones with active noise cancellation, 30-hour battery life, and built-in microphone for clear voice calls.",
        "category": "Electronics",
        "price": 199.00
    },
    {
        "id": "SKU-1004",
        "name": "Ergonomic Office Chair",
        "description": "High-back mesh ergonomic chair with adjustable lumbar support, 3D armrests, and a tilt mechanism for long hours of comfortable working.",
        "category": "Furniture",
        "price": 249.99
    }
]

# 4. Format for LangChain and Ingest
print("Vectorizing and ingesting data...")
docs = []
for prod in dummy_products:
    # We embed the description and name so the semantic search can find it
    page_content = f"Product Name: {prod['name']}\nDescription: {prod['description']}"
    
    # Metadata is crucial for filtering (e.g., "Find me Electronics under $100")
    metadata = {
        "id": prod["id"],
        "name": prod["name"],
        "category": prod["category"],
        "price": prod["price"]
    }
    docs.append(Document(page_content=page_content, metadata=metadata))

# Push to Qdrant
vector_store = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    url=qdrant_url,
    collection_name=collection_name,
)

print(f"Successfully ingested {len(docs)} products into Qdrant collection: '{collection_name}'!")