"""Batch prediction script using the API."""

import argparse
import json
import sys

import httpx


def run_batch(base_url: str, input_file: str, output_file: str = None):
    """
    Send batch predictions to the /predict endpoint.

    Reads input records from a JSON file (array of feature dicts),
    posts each to the API, and optionally writes results to output_file.
    """
    with open(input_file, "r") as f:
        records = json.load(f)

    results = []
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        for record in records:
            response = client.post("/predict", json=record)
            response.raise_for_status()
            results.append(response.json())

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {output_file}")
    else:
        for r in results:
            print(r)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction client")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--input", required=True, help="JSON file with input records")
    parser.add_argument("--output", default=None, help="Output JSON file for results")

    args = parser.parse_args()
    run_batch(args.url, args.input, args.output)