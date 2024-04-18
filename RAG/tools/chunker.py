import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from RAG.tools.vectorizer import add_text_chunk_to_db

nltk.download('punkt')
nltk.download('stopwords')


def remove_stop_words(text: str):
    stop_words = set(stopwords.words('english'))

    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def is_readable(text: str, threshold=0.30):
    non_ascii = sum(1 for char in text if ord(char) > 127)
    total_chars = len(text)
    return (non_ascii / total_chars) < threshold


def chunk_text(text, max_chunk_size):
    text = remove_stop_words(text)
    sentences = sent_tokenize(text)
    chunk = []
    visited = set()

    for sentence in sentences:
        if not is_readable(sentence):
            continue
        if sentence in visited:
            continue
        visited.add(sentence)
        chunk.append(sentence)
        if len(chunk) == max_chunk_size:
            process_chunk(chunk)
    if chunk:
        process_chunk(chunk)


def process_chunk(chunk: []):
    content = ' '.join(chunk)
    add_text_chunk_to_db(content)
