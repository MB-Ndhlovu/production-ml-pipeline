"""Batch prediction script using the prediction API."""

import argparse
import json
import sys

import requests

DEFAULT_URL = "http://localhost:8000/predict"


def send_batch(payloads: list[dict], api_url: str = DEFAULT_URL) -> list[dict]:
    """Send a list of prediction requests to the API."""
    results = []
    for payload in payloads:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        results.append(response.json())
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch prediction via the /predict API.")
    parser.add_argument(
        "--input", "-i", required=True, help="Path to JSON file with an array of applications."
    )
    parser.add_argument(
        "--url", "-u", default=DEFAULT_URL, help="Base URL of the prediction API."
    )
    parser.add_argument(
        "--output", "-o", help="Optional path to write results as JSON."
    )
    args = parser.parse_args()

    with open(args.input) as f:
        payloads = json.load(f)

    if not isinstance(payloads, list):
        print("Error: input file must contain a JSON array of applications.", file=sys.stderr)
        sys.exit(1)

    print(f"Running batch prediction for {len(payloads)} application(s)...")
    results = send_batch(payloads, args.url)

    for i, result in enumerate(results):
        print(f"Application {i + 1}: approved={result['approved']}, "
              f"prob={result['default_probability']}, band={result['risk_band']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()