# Test Brave Search API
import os
import requests
import json
from dotenv import load_dotenv
from pathlib import Path

def test_ori_brave_search():
    # Load environment variables from root directory
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)
    api_key = os.getenv('BRAVE_API_KEY')
    
    if not api_key:
        raise ValueError("BRAVE_API_KEY not found in environment variables")
    
    # API endpoint
    url = "https://api.search.brave.com/res/v1/web/search"
    
    # Search parameters
    params = {
        "q": "What is the capital of USA?",  # search query
    }
    
    # Request headers
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key
    }
    
    try:
        # Make the request
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the response
        data = response.json()
        
        # Basic validation of the response
        assert "web" in data, "Expected 'web' in response"
        assert "results" in data["web"], "Expected 'results' in web response"
        
        # Print search query info
        print("\n=== Search Results ===")
        print(f"Query: {params['q']}")
        print(f"Total Results: {data['web'].get('total_results', 'N/A')}")
        print("\n=== Top Results ===")
        
        # Print detailed results
        for i, result in enumerate(data["web"]["results"][:5], 1):  # Show top 5 results
            print(f"\nResult #{i}:")
            print("-" * 50)
            print(f"Title: {result.get('title', 'N/A')}")
            print(f"URL: {result.get('url', 'N/A')}")
            print(f"Description: {result.get('description', 'N/A')}")
            if 'meta_url' in result:
                print(f"Meta URL: {result['meta_url']}")
            if 'age' in result:
                print(f"Age: {result['age']}")
            print("-" * 50)
        
        # Print full response in debug mode
        print("\n=== Full Response (Debug) ===")
        print(json.dumps(data, indent=2))
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        raise
    except KeyError as e:
        print(f"Unexpected response format: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    test_ori_brave_search()
