from crawl4ai import WebCrawler

def crawl_website(url):
    """Crawls a website using the WebCrawler and returns the extracted content."""
    crawler = WebCrawler()
    crawler.warmup()  # Warm up the crawler (load necessary models)
    result = crawler.run(url=url)
    return result.extracted_content

def main():
    url = "https://www.thenews.com.pk"
    
    try:
        content = crawl_website(url)
        print(f"Extracted Content:\n{content}")
    except Exception as e:
        print(f"Error occurred while crawling the website: {e}")

if __name__ == "__main__":
    main()
