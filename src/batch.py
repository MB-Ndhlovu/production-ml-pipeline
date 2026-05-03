"""Batch prediction script using the API."""
import argparse
import json
import httpx


def run_batch(api_url: str, input_file: str, output_file: str | None = None):
    """
    Send batch predictions via the API.

    Args:
        api_url: Base URL of the API (e.g. http://localhost:8000).
        input_file: Path to JSON file containing a list of input records.
        output_file: Optional path to write results as JSON.
    """
    with open(input_file, "r") as f:
        records = json.load(f)

    results = []
    with httpx.Client(timeout=60.0) as client:
        for record in records:
            response = client.post(f"{api_url}/predict", json=record)
            response.raise_for_status()
            results.append(response.json())

    for r in results:
        print(r)

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--input", required=True, help="JSON input file path")
    parser.add_argument("--output", help="Optional JSON output file")
    args = parser.parse_args()
    run_batch(args.api_url, args.input, args.output)