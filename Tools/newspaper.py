from newspaper import Article

def parse_article(article_url):
    """Parses an article using the Newspaper3k library and returns the title and text."""
    article = Article(article_url)
    article.download()
    article.parse()
    return article.title, article.text

def main():
    article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"
    
    try:
        title, text = parse_article(article_url)
        print(f"Title: {title}\n")
        print(f"Text: {text}")
    except Exception as e:
        print(f"Error occurred while parsing article: {e}")

if __name__ == "__main__":
    main()
