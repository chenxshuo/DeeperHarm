import os
import json
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Union, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session with custom headers
headers = {
    'Accept': 'application/json',
    'Accept-Encoding': 'gzip',
}

# Initialize session
session = requests.Session()
session.headers.update(headers)

class BraveSearchClient:
    """Client for interacting with the Brave Search API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Brave Search client.
        
        Args:
            api_key: Brave Search API key. If not provided, will try to get from environment.
        """
        self.api_key = api_key or os.getenv('BRAVE_API_KEY')
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY not found in environment variables")
        
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.api_key
        })

    def search(self, query: str, country: str = 'us', language: str = 'en', 
              timeout: int = 20, max_retries: int = 3) -> Dict:
        """
        Perform a web search using Brave Search API.
        
        Args:
            query: Search query string
            country: Country code for search results
            language: Language code for search results
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            
        Returns:
            dict: JSON response from the API
        """
        params = {
            'q': query,
            'country': country,
            'language': language
        }
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=timeout
                )
                response.raise_for_status()
                return response.json()
                
            except Timeout:
                if attempt == max_retries - 1:
                    return {"error": "Request timed out"}
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return {"error": f"Request failed: {str(e)}"}
                time.sleep(1)
                
        return {"error": "All retry attempts failed"}

    @staticmethod
    def extract_search_results(response: Dict) -> List[Dict]:
        """
        Extract relevant information from Brave search results.
        
        Args:
            response: JSON response from the Brave Search API
            
        Returns:
            list: A list of dictionaries containing the extracted information
        """
        results = []
        
        if "web" not in response or "results" not in response["web"]:
            return results
            
        for idx, result in enumerate(response["web"]["results"]):
            info = {
                'id': idx + 1,
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'site_name': result.get('meta_url', '').get('hostname', ''),
                'snippet': result.get('description', ''),
                'language': result.get('language', ''),
                'family_friendly': result.get('family_friendly', True),
                'reading_time': result.get('reading_time', None)
            }
            
            # Add additional metadata if available
            if 'meta_url' in result:
                info['meta'] = {
                    'favicon': result['meta_url'].get('favicon', ''),
                    'hostname': result['meta_url'].get('hostname', '')
                }
                
            results.append(info)
            
        return results

async def brave_search_async(query: str, api_key: Optional[str] = None,
                      country: str = 'us', language: str = 'en',
                      timeout: int = 20) -> Dict:
    """
    Perform an asynchronous search using the Brave Search API.
    
    Args:
        query: Search query
        api_key: Brave Search API key (optional)
        country: Country code for search results
        language: Language code for search results
        timeout: Request timeout in seconds
        
    Returns:
        dict: JSON response from the API
    """
    client = BraveSearchClient(api_key)
    response = client.search(query, country, language, timeout)
    return response

def batch_search(queries: List[str], api_key: Optional[str] = None,
                max_workers: int = 5, show_progress: bool = True) -> Dict[str, Dict]:
    """
    Perform batch searches using thread pool.
    
    Args:
        queries: List of search queries
        api_key: Brave Search API key (optional)
        max_workers: Maximum number of concurrent threads
        show_progress: Whether to show progress bar
        
    Returns:
        dict: Dictionary mapping queries to their search results
    """
    client = BraveSearchClient(api_key)
    
    def search_wrapper(query):
        return query, client.search(query)
    
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(search_wrapper, query) for query in queries]
        
        if show_progress:
            futures = tqdm(concurrent.futures.as_completed(futures),
                         total=len(futures),
                         desc="Processing searches")
                         
        for future in futures:
            query, result = future.result()
            results[query] = result
            
    return results



def test_brave_search():
    # Create client (will use BRAVE_API_KEY from environment)
    client = BraveSearchClient()

    # Simple search
    results = client.search("What is the capital of USA?")

    # Get formatted results and print
    extracted_results = client.extract_search_results(results)

    # Batch search
    queries = ["What is the capital of USA?", "What is the capital of France?", "What is the capital of Germany?"]
    batch_results = client.batch_search(queries, show_progress=True)

    # Print results
    print("\n=== Simple Search ===")
    for result in extracted_results:
        print(f"\n{result['id']}. {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Description: {result['description']}")
        if 'meta' in result:
            print(f"Domain: {result['meta']['hostname']}")
        print("-" * 80)

    print("\n=== Batch Search ===")
    for i, result in enumerate(batch_results, 1):
        print(f"\nResult #{i}")
        print("-" * 80)
        print(f"Query: {queries[i-1]}")
        print(f"Total Results: {result['web'].get('total_results', 'N/A')}")
        print("\n=== Top Results ===")
        for j, sub_result in enumerate(result["web"]["results"][:5], 1):  # Show top 5 results
            print(f"\n{j}. {sub_result.get('title', 'N/A')}")
            print(f"URL: {sub_result.get('url', 'N/A')}")
            print(f"Description: {sub_result.get('description', 'N/A')}")
            if 'meta_url' in sub_result:
                print(f"Meta URL: {sub_result['meta_url']}")
            if 'age' in sub_result:
                print(f"Age: {sub_result['age']}")
            print("-" * 80)


if __name__ == "__main__":
    # Example usage
    query = "What is the capital of USA?"
    client = BraveSearchClient()
    
    print(f"Searching for: {query}")
    results = client.search(query)
    
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        extracted_results = client.extract_search_results(results)
        print("\nTop search results:")
        for result in extracted_results[:5]:  # Show top 5 results
            print(f"\n{result['id']}. {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Description: {result['description']}")
            if 'meta' in result:
                print(f"Domain: {result['meta']['hostname']}")
            print("-" * 80)
