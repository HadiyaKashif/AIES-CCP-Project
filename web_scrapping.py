import requests
from config import search, web_search_tool
from bs4 import BeautifulSoup

def search_google(query, num_results=3):
    if not web_search_tool:
        return None, "Web search is not configured (missing API keys)"
    
    try:
        # Get raw JSON results
        raw_results = search.results(query, num_results)
        
        # Format the results
        formatted = []
        for i, result in enumerate(raw_results, 1):
            formatted.append(
                f"🔍 **Result {i}**\n"
                f"📌 {result.get('title', 'No title')}\n"
                f"🔗 {result.get('link', 'No link')}\n"
                f"📝 {result.get('snippet', 'No description')}\n"
            )
        return raw_results, "\n\n".join(formatted) if formatted else "No results found"
    except Exception as e:
        return None, f"Web search error: {str(e)}"
    
def scrape_website(url):
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        return "\n".join([p.get_text() for p in soup.find_all("p")]) or "No relevant text found."
    except Exception as e:
        return f"Error scraping: {str(e)}"
