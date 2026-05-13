"""Batch prediction script using the credit scoring API."""

import json
import sys
import argparse
import httpx


def run_batch(url: str, records: list[dict]) -> None:
    """Send multiple prediction requests to the /predict endpoint."""
    with httpx.Client(base_url=url, timeout=30.0) as client:
        results = []
        for record in records:
            response = client.post("/predict", json=record)
            response.raise_for_status()
            results.append(response.json())

        print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Batch prediction via credit scoring API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("file", nargs="?", help="JSON file with list of prediction records")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            records = json.load(f)
    else:
        records = json.load(sys.stdin)

    run_batch(args.url, records)


if __name__ == "__main__":
    main()