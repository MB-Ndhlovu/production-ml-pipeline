"""Batch prediction script using the REST API."""

import argparse
import json
from pathlib import Path

import httpx


def run_batch(api_url: str, input_file: Path, output_file: Path | None = None):
    """
    Send batch predictions to the /predict endpoint.

    Parameters
    ----------
    api_url : str
        Base URL of the API (e.g. http://localhost:8000).
    input_file : Path
        JSON file containing a list of application records.
    output_file : Path | None
        Optional path to write results as JSON.
    """
    data = json.loads(input_file.read_text())

    if not isinstance(data, list):
        raise ValueError("Input file must contain a JSON list of applications.")

    records = []
    with httpx.Client(timeout=30.0) as client:
        for i, record in enumerate(data):
            try:
                response = client.post(f"{api_url}/predict", json=record)
                response.raise_for_status()
                records.append({"index": i, **response.json()})
                print(f"[{i}] OK: {records[-1]['risk_band']} (prob={records[-1]['default_probability']})")
            except Exception as exc:
                records.append({"index": i, "error": str(exc)})
                print(f"[{i}] FAILED: {exc}")

    if output_file:
        output_file.write_text(json.dumps(records, indent=2))
        print(f"\nResults written to {output_file}")

    return records


def main():
    parser = argparse.ArgumentParser(description="Batch prediction via REST API")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("input_file", type=Path, help="JSON file with application list")
    parser.add_argument("--output", type=Path, help="Output JSON file for results")
    args = parser.parse_args()
    run_batch(args.api_url, args.input_file, args.output)


if __name__ == "__main__":
    main()