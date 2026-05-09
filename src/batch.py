"""Batch prediction script using the API."""
import argparse
import json
import sys
import httpx


def batch_predict(url: str, input_file: str, output_file: str | None = None):
    """
    Run batch predictions via the API.

    Reads newline-delimited JSON from input_file, sends each record to
    the /predict endpoint, and prints or writes results.
    """
    results = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                results.append(resp.json())

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} results to {output_file}")
    else:
        for r in results:
            print(json.dumps(r))


def main():
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--url", default="http://localhost:8000/predict", help="API endpoint URL")
    parser.add_argument("--input", required=True, help="Input file (newline-delimited JSON)")
    parser.add_argument("--output", help="Output file for results (optional)")
    args = parser.parse_args()

    batch_predict(args.url, args.input, args.output)


if __name__ == "__main__":
    main()