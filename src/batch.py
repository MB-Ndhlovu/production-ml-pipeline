"""Batch prediction script using the API."""

import argparse
import json
import httpx
from typing import List


def batch_predict(url: str, data: List[dict]) -> List[dict]:
    """
    Send batch predictions to the API.

    Args:
        url: Base URL of the API
        data: List of prediction input dictionaries

    Returns:
        List of prediction results
    """
    results = []

    with httpx.Client(timeout=30.0) as client:
        for i, item in enumerate(data):
            try:
                response = client.post(f"{url}/predict", json=item)
                response.raise_for_status()
                results.append({"index": i, "status": "success", "result": response.json()})
            except Exception as e:
                results.append({"index": i, "status": "error", "error": str(e)})

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch prediction client")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--input", required=True, help="JSON file with input data")
    parser.add_argument("--output", default="batch_results.json", help="Output file for results")

    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    print(f"Running batch prediction on {len(data)} records...")
    results = batch_predict(args.url, data)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")

    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"Completed: {success_count}/{len(data)} successful")


if __name__ == "__main__":
    main()