from RAG.tools.retriever import semantic_search, model

import requests
import numpy as np

API_URL = "https://ufooozidbpadrm6t.us-east-1.aws.endpoints.huggingface.cloud"

headers = {
    "Accept": "application/json",
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}


def generate_with_rag(query: str):
    context = semantic_search(query, 1)
    print(context)
    new_query = f"""
    ### Instruction:
    Use the provided input to create an instruction that could have been used to generate the response with an LLM.
    ### Question {query} #
    ### Answer
    """
    if cosine_sim_comparison(context, query) >= 0.5:
        prompt = f'context: {context}\n\nquery: {new_query}'
        return generate_response(prompt), context
    else:
        return generate_response(query)


def generate_response(query):
    prompt = f"""
    ### Instruction:
    Use the provided input to create an instruction that could have been used to generate the response with an LLM.
    ### Question {query} #
    ### Answer
    """
    response = requests.post(API_URL, headers=headers, json={'inputs': prompt, 'parameters': {}})
    print(response.json())
    return response.json()[0]['generated_text']


def cosine_sim_comparison(context: str, query: str):
    context_embedding = model.encode(context)
    query_embedding = model.encode(query)

    return cosine_sim_calculation(context_embedding, query_embedding)


def cosine_sim_calculation(vec1, vec2):
    dot_prod = np.dot(vec1, vec2)

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    return dot_prod / (norm_vec1 * norm_vec2)
