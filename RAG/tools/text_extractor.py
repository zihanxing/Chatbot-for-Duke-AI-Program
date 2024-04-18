from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from typing import List, Set


def extract_text(url: str, result: List[str], visited: Set[str], key_substring: str) -> None:
    if url in visited or key_substring not in url:
        return
    visited.add(url)
    response = requests.get(url)
    if response.status_code != 200:
        return
    soup = BeautifulSoup(response.content, 'html.parser')
    text_data = soup.get_text()
    result.append(text_data)
    links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.startswith('http'):
            links.append(href)
        elif href:
            new_link = urljoin(url, href)
            if new_link.startswith('http'):
                links.append(new_link)
    for link in links:
        extract_text(link, result, visited, key_substring)


if __name__ == '__main__':
    res = []
    extract_text(url='https://ai.meng.duke.edu', result=res, visited=set(), key_substring='ai.meng.duke.edu')
    extract_text(url='https://sites.duke.edu/aipi/new-student-resources/', result=res, visited=set(), key_substring='aipi')
    print(res)
