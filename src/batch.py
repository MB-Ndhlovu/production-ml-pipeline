import argparse
import json
import httpx
from typing import List

DEFAULT_API_URL = "http://localhost:8000/predict"

def batch_predict(data: List[dict], api_url: str = DEFAULT_API_URL) -> List[dict]:
    """
    Send batch prediction requests to the API.
    
    Args:
        data: List of prediction input dictionaries
        api_url: URL of the prediction API
        
    Returns:
        List of prediction results
    """
    results = []
    
    with httpx.Client(timeout=30.0) as client:
        for item in data:
            response = client.post(api_url, json=item)
            response.raise_for_status()
            results.append(response.json())
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Batch prediction using the credit scoring API")
    parser.add_argument("input_file", help="JSON file containing prediction requests")
    parser.add_argument("--output", "-o", help="Output file for results (default: stdout)")
    parser.add_argument("--url", "-u", default=DEFAULT_API_URL, help="API URL")
    
    args = parser.parse_args()
    
    with open(args.input_file, "r") as f:
        data = json.load(f)
    
    results = batch_predict(data, args.url)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()