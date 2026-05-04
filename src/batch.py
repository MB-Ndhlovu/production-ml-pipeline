"""Batch prediction script using the API."""
import argparse
import json
import httpx
from typing import List


def batch_predict(url: str, payloads: List[dict]) -> List[dict]:
    """Send batch prediction requests to the API."""
    results = []
    with httpx.Client(timeout=30.0) as client:
        for payload in payloads:
            response = client.post(f"{url}/predict", json=payload)
            response.raise_for_status()
            results.append(response.json())
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--input", required=True, help="JSON file with array of inputs")
    parser.add_argument("--output", help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    with open(args.input) as f:
        payloads = json.load(f)

    results = batch_predict(args.url, payloads)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {len(results)} results to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()