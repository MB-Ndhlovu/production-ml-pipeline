"""Batch prediction script using the API."""
import argparse
import requests
import json
import sys


def submit_batch(base_url: str, records: list[dict]) -> list[dict]:
    """Submit a batch of records to the /predict endpoint."""
    results = []
    for record in records:
        response = requests.post(f"{base_url}/predict", json=record)
        response.raise_for_status()
        results.append(response.json())
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSON file containing list of records",
    )
    parser.add_argument(
        "--output",
        help="Path to write results JSON (optional, prints to stdout if omitted)",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        records = json.load(f)

    results = submit_batch(args.url, records)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()