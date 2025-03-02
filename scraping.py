import requests
import os
from bs4 import BeautifulSoup
import re

def scrape_articles(api_key):

    url = "https://newsapi.org/v2/everything"

    params = {
        'q': 'finance',  
        'pageSize': 5,  # Limit the number of articles (you can adjust this)
        'sources':'business-insider',
        'apiKey': api_key,  
    }

    # Create 'articles' directory if it doesn't exist
    if not os.path.exists("articles"):
        os.mkdir("articles")

    response = requests.get(url, params=params)

    if response.status_code == 200:
        articles = response.json()['articles']

        if not articles:
            print("No articles found!")

        print(articles)

        for i, article in enumerate(articles, 1):
            print(f"Article {i}: {article['title']}")
            print(f"Description: {article['description']}")
            print(f"URL: {article['url']}")

            try:
                article_url = article['url']
                article_response = requests.get(article_url)

                if article_response.status_code == 200:
                    # Parse the HTML content with BeautifulSoup
                    soup = BeautifulSoup(article_response.content, 'html.parser')

                    # Get all the text from paragraphs (<p>) and headings (<h1>, <h2>, etc.)
                    paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

                    # Combine the text from all these tags
                    article_text = " ".join([para.get_text() for para in paragraphs])

                    # Clean up excessive whitespace and newlines
                    article_text = re.sub(r'\s+', ' ', article_text).strip() 

                    # Print the preview (first 500 characters of the article)
                    preview = article_text[:500]  
                    print(f"Preview: {preview}\n")

                    # Sanitize title and remove invalid characters
                    safe_title = article['title'].replace('/', '_').replace('\\', '_').replace(':', '_')[:100]

                    # Construct filename (ensure it's within the 'articles' directory)
                    filename = f"articles/article_{i}_{safe_title}.txt"

                    with open(filename, 'w', encoding='utf-8') as file:
                        # Save the article's title, description, URL, and content
                        file.write(f"Title: {article['title']}\n")
                        file.write(f"Description: {article['description']}\n")
                        file.write(f"URL: {article['url']}\n")
                        file.write(f"\nContent:\n{article_text}")
                        
                    print(f"Article {i} saved to {filename}\n")

                else:
                    print(f"Unable to fetch article content for preview.\n")
            except requests.exceptions.HTTPError as e:
                print(f"HTTPError for {article_url}: {e}, moving to next article...\n")
            except requests.exceptions.RequestException as e:
                print(f"RequestException for {article_url}: {e}, moving to next article...\n")
            except Exception as e:
                print(f"Error fetching article content: {e}\n")
        print("Articles successfully scraped.")
    else:
        print(f"Error: {response.status_code}")
