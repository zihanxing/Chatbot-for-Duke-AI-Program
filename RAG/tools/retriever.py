from RAG.tools.vectorizer import class_name, connect_client
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')


def semantic_search(query: str, chunk_num: int) -> str:
    client = connect_client()
    try:
        collection = client.collections.get(class_name)
        response = collection.query.fetch_objects(include_vector=True)
        vectors = [o.vector['default'] for o in response.objects]

        query_vector = model.encode(query)

        dot_product = np.dot(vectors, query_vector)
        query_norm = np.linalg.norm(query_vector)
        chunk_norms = np.linalg.norm(vectors, axis=1)
        cosine_sim = dot_product / (query_norm * chunk_norms)
        max_indices = np.argsort(cosine_sim)[-chunk_num:]
        most_sim_contents = [response.objects[i].properties['content'] for i in max_indices]
        return ' '.join(most_sim_contents)

    finally:
        client.close()


if __name__ == '__main__':
    semantic_search('college', 3)
