import requests
from typing import List, Dict, Union
from urllib.parse import urljoin

class WebParserClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize Web parser client
        
        Args:
            base_url: Base URL of the API server, defaults to local test server
        """
        self.base_url = base_url.rstrip('/')
        
    def parse_urls(self, urls: List[str]) -> List[Dict[str, Union[str, bool]]]:
        """
        Send URL list to parsing server and get parsing results
        
        Args:
            urls: List of URLs to parse
            
        Returns:
            List of parsing results
            
        Raises:
            requests.exceptions.RequestException: When API request fails
        """
        endpoint = urljoin(self.base_url, "/parse_urls")
        response = requests.post(endpoint, json={"urls": urls})
        response.raise_for_status()  # If response status code is not 200, raise exception
        
        return response.json()["results"]


# Usage example
if __name__ == "__main__":
    # Create client instance (modify base_url if API runs on other server)
    client = WebParserClient("http://xxxx")
    
    # Test URL list
    test_urls = [
        "http://xxxx",
    ]
    
    try:
        # Call API to parse URLs
        results = client.parse_urls(test_urls)
        print(results[1]['content'])

    except requests.exceptions.RequestException as e:
        print(f"API call failed: {str(e)}") 