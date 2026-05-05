"""Batch prediction script using the API."""

import argparse
import json
import requests
from pathlib import Path


def run_batch_predictions(url: str, input_file: str, output_file: str = None):
    """
    Run batch predictions via the API.

    Args:
        url: Base URL of the API (e.g., http://localhost:8000)
        input_file: Path to JSON file with list of predictions
        output_file: Optional path to save results
    """
    with open(input_file, "r") as f:
        data = json.load(f)

    results = []
    for item in data:
        response = requests.post(f"{url}/predict", json=item)
        if response.status_code == 200:
            results.append(response.json())
        else:
            results.append({"error": f"Status {response.status_code}: {response.text}"})

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    else:
        for r in results:
            print(json.dumps(r, indent=2))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", help="Output JSON file (optional)")

    args = parser.parse_args()
    run_batch_predictions(args.url, args.input, args.output)