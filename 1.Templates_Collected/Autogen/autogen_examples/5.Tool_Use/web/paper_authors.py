# filename: paper_authors.py
import requests
from bs4 import BeautifulSoup

def get_paper_authors(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    authors = [span.text.strip() for span in soup.find_all('span', class_='authors')]
    return authors

url = 'https://arxiv.org/abs/2308.08155'
authors = get_paper_authors(url)
print('Authors:', authors)