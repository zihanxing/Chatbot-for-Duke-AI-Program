from RAG.tools.retriever import semantic_search

import cohere


def generate_with_rag(query: str):
    context = semantic_search(query, 1)
    print(context)
    prompt = f'query: {query}\n\ncontext: {context}'
    return generate_response(prompt), context


def generate_response(query):
    co = cohere.Client('YJsYd0iwJjjg222OpO0XzvHQT8zRviVUKBfdgN6c')
    response = co.generate(prompt=query, model='command-light')
    print(response.generations[0].text)
    return response.generations[0].text
