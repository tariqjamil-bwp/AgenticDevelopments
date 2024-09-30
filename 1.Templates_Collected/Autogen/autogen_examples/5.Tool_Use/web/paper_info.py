# filename: paper_info.py
import requests
from bs4 import BeautifulSoup

def get_paper_info(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('h1', class_='title mathjax').text.strip()
    abstract = soup.find('blockquote', class_='abstract mathjax').text.strip()
    keywords = [span.text.strip() for span in soup.find_all('span', class_='subject')]
    return title, abstract, keywords

url = 'https://arxiv.org/abs/2308.08155'
title, abstract, keywords = get_paper_info(url)
print('Title:', title)
print('Abstract:')
print(abstract)
print('Keywords:', keywords)