from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# 1. Initialize the exact same embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Connect to the existing Qdrant collection
qdrant_url = "http://localhost:6333"
collection_name = "ecommerce_catalog"

vector_store = QdrantVectorStore(
    client=QdrantClient(url=qdrant_url),
    collection_name=collection_name,
    embedding=embeddings,
)

# 3. Perform a semantic search
query = "I need something comfortable to sit on while I work."
print(f"Searching for: '{query}'\n")

# Retrieve the top 2 most relevant documents
results = vector_store.similarity_search(query, k=2)

for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(f"  Name: {doc.metadata['name']}")
    print(f"  Category: {doc.metadata['category']}")
    print(f"  Price: ${doc.metadata['price']}")
    print(f"  Content: {doc.page_content}\n")