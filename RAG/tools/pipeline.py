"""This pipeline runs the process of extracting texts, chunking and storing in the vector database. It works in the
following way:
1. For each of the wiki page, it extracts the content of the wiki page.
2. For the content of each wiki page, it chunks the content.
3. For each chunk, it stores this chunk in the database.
"""
from typing import List
from RAG.tools.text_extractor import extract_text
from RAG.tools.chunker import chunk_text
from RAG.tools.vectorizer import initiate_storage


def run_pipeline(webs: List):
    initiate_storage()

    for url, key_substring in webs:
        res = []
        extract_text(url=url, key_substring=key_substring, result=res, visited=set())
        visited = set()
        for sentence in res:
            if sentence in visited:
                continue
            visited.add(sentence)
            chunk_text(sentence, 3)


if __name__ == '__main__':
    webs = [('https://ai.meng.duke.edu', 'ai.meng.duke.edu'), ('https://sites.duke.edu/aipi/new-student-resources/', 'aipi')]
    run_pipeline(webs=webs)
