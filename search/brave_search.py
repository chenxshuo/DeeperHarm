from typing import Dict, List

class BraveSearchClient:
    @staticmethod
    def extract_search_results(search_results: Dict) -> List[Dict]:
        """Extract relevant information from search results."""
        results = []
        try:
            # Check search results format
            if not isinstance(search_results, dict):
                print(f"Warning: search_results is not a dict: {type(search_results)}")
                return []
                
            # Check if web field exists
            if 'web' not in search_results:
                print(f"Warning: 'web' key not found in search_results")
                return []
                
            # Check if web field is a dictionary
            if not isinstance(search_results['web'], dict):
                print(f"Warning: search_results['web'] is not a dict: {type(search_results['web'])}")
                return []
                
            # Check if results field exists
            if 'results' not in search_results['web']:
                print(f"Warning: 'results' key not found in search_results['web']")
                return []
                
            # Check if results field is a list
            if not isinstance(search_results['web']['results'], list):
                print(f"Warning: search_results['web']['results'] is not a list: {type(search_results['web']['results'])}")
                return []
            
            for item in search_results['web']['results']:
                # Safely extract each field
                result = {
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'snippet': item.get('description', '')
                }
                results.append(result)
        except Exception as e:
            print(f"Error in extract_search_results: {e}")
        return results 