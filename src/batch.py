"""Batch prediction script using the credit scoring API."""

import argparse
import csv
import sys
from pathlib import Path

import requests


def predict_from_api(url: str, payload: dict) -> dict:
    """Send prediction request to the API."""
    response = requests.post(f"{url}/predict", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Batch prediction using the credit scoring API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)

    with open(input_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    fieldnames = ["approved", "default_probability", "risk_band"]
    output_path = Path(args.output)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows):
            print(f"Processing row {i + 1}/{len(rows)}...")
            try:
                result = predict_from_api(args.url, row)
                writer.writerow({
                    "approved": result["approved"],
                    "default_probability": result["default_probability"],
                    "risk_band": result["risk_band"],
                })
            except Exception as e:
                print(f"Error on row {i + 1}: {e}")
                writer.writerow({"approved": "", "default_probability": "", "risk_band": ""})

    print(f"Batch prediction complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
