"""Batch prediction script — calls the /predict endpoint for each record in a CSV."""

import argparse
import csv
import json
import sys
from pathlib import Path

import httpx


def run_batch(csv_path: str, base_url: str = "http://localhost:8000"):
    """Read records from a CSV and print predictions."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        for i, row in enumerate(records, 1):
            try:
                response = client.post("/predict", json=row)
                response.raise_for_status()
                result = response.json()
                print(f"Row {i}: {json.dumps(result)}")
            except Exception as exc:
                print(f"Row {i}: ERROR — {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Batch prediction via credit scoring API")
    parser.add_argument("csv", help="Path to CSV file with applicant records")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"File not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    run_batch(args.csv, args.url)


if __name__ == "__main__":
    main()