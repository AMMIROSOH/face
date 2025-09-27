from qdrant_client import QdrantClient
from qdrant_client.http import models
from constants import QDRANT_HOST, QDRANT_PORT

collection_name = "faces"
client = QdrantClient(QDRANT_HOST, grpc_port=QDRANT_PORT)
collections = client.get_collections().collections
existing_collections = [col.name for col in collections]

if collection_name not in existing_collections:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
                size=512,
                distance=models.Distance.COSINE
            )
)

def search_vec(vec: list[float], top_n: int=1):
    return client.search(
        collection_name=collection_name,
        query_vector=vec,
        limit=top_n
    )
