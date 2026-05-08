"""Batch prediction script using the API."""
import argparse
import json
import sys

import httpx


def batch_predict(url: str, records: list[dict]) -> list[dict]:
    """Send a batch of records to /predict and return results."""
    results = []
    with httpx.Client(timeout=60.0) as client:
        for record in records:
            response = client.post(url, json=record)
            response.raise_for_status()
            results.append(response.json())
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch prediction via the /predict API")
    parser.add_argument(
        "--url", default="http://localhost:8000/predict", help="Predict endpoint URL"
    )
    parser.add_argument(
        "--input", required=True, help="JSON file with an array of request objects"
    )
    parser.add_argument(
        "--output", help="Output JSON file (defaults to stdout)"
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        records = json.load(f)

    results = batch_predict(args.url, records)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()