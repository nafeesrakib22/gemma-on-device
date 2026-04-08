import re
from duckduckgo_search import DDGS

def web_search(query: str) -> str:
    """
    Searches the internet for real-time information using DuckDuckGo.
    
    Args:
        query: The search query to look up.
    """
    # LiteRT-LM may sometimes pass internal markers in the tool arguments.
    # We strip any non-printable or special delimiter-like sequences.
    sanitized_query = re.sub(r'<\|.*?\|>', '', query).strip()
    sanitized_query = sanitized_query.replace('"', '').replace("'", "")
    

    if not sanitized_query:
        return "Error: Empty search query after sanitization."

    try:
        # Using DDGS for real-time search
        with DDGS(timeout=20) as ddgs:
            results = list(ddgs.text(sanitized_query, max_results=5))
            if not results:
                return f"No results found for '{sanitized_query}'."
            return "\n".join([f"Source: {r['href']}\nTitle: {r['title']}\nSnippet: {r['body']}\n" for r in results])
    except Exception as e:
        # If the library still fails, explain the issue to the model
        return f"The web search tool is currently unavailable or timed out. Error: {str(e)}"
