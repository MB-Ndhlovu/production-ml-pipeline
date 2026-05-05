#!/usr/bin/env python3
"""
Batch prediction script.

Makes predictions for multiple applicants via the API.
Writes results to a JSON file.

Usage:
    python src/batch.py [--input <csv_path>] [--output <json_path>] [--url <api_url>]
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

DEFAULT_URL = "http://localhost:8000/predict"


def predict_one(url: str, payload: dict) -> dict:
    """Send a single prediction request to the API."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def batch_predict(url: str, records: list[dict]) -> list[dict]:
    """Send batch records to the prediction API."""
    results = []
    for record in records:
        try:
            result = predict_one(url, record)
            result["_input"] = record
            results.append(result)
        except urllib.error.HTTPError as e:
            results.append({"_input": record, "error": str(e)})
    return results


def load_csv(path: Path) -> list[dict]:
    """Simple CSV loader for the expected format."""
    import csv

    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {
                "income": float(row["income"]),
                "credit_score": int(row["credit_score"]),
                "employment_years": float(row["employment_years"]),
                "debt_to_income": float(row["debt_to_income"]),
                "loan_history_count": int(row["loan_history_count"]),
                "age": int(row["age"]),
                "home_ownership": row["home_ownership"],
                "verified_income": int(row["verified_income"]),
            }
            records.append(cleaned)
    return records


def main():
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to CSV file with applicant data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.json",
        help="Output JSON path (default: predictions.json)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help=f"API URL (default: {DEFAULT_URL})",
    )
    args = parser.parse_args()

    if args.input:
        records = load_csv(Path(args.input))
    else:
        records = [
            {
                "income": 65000,
                "credit_score": 720,
                "employment_years": 5,
                "debt_to_income": 0.28,
                "loan_history_count": 2,
                "age": 34,
                "home_ownership": "rent",
                "verified_income": 1,
            }
        ]

    results = batch_predict(args.url, records)

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Batch complete: {len(results)} predictions written to {output_path}")


if __name__ == "__main__":
    main()