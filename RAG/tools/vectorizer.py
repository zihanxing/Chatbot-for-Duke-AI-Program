import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, Property, DataType


class_name = 'TextChunk'


def connect_client():
    weaviate_client = weaviate.connect_to_wcs(
        cluster_url='https://aipi-chatgpt-auth-a9ax4084.weaviate.network',
        auth_credentials=weaviate.auth.AuthApiKey('qx290jxxhKMV9bNlzfA5KQAGTeXz6cN9qkb6'),
        skip_init_checks=True,
        headers={'X-Cohere-Api-Key': 'YJsYd0iwJjjg222OpO0XzvHQT8zRviVUKBfdgN6c'}
    )
    return weaviate_client


def add_text_chunk_to_db(chunk: str):
    weaviate_client = connect_client()
    try:
        text_chunk = weaviate_client.collections.get(class_name)
        print(chunk)
        result = text_chunk.data.insert({
            'content': chunk,
        })
        print(result)
    finally:
        weaviate_client.close()


def initiate_storage():
    weaviate_client = connect_client()
    try:
        weaviate_client.collections.delete_all()
        weaviate_client.collections.create(
            name=class_name,
            vectorizer_config=Configure.Vectorizer.text2vec_cohere(),
            properties=[
                wvc.config.Property(
                    name='content',
                    data_type=wvc.config.DataType.TEXT,
                    vectorize_property_name=True,
                    tokenization=wvc.config.Tokenization.LOWERCASE
                ),
            ]
        )
    finally:
        weaviate_client.close()
