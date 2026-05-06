#!/usr/bin/env python3
"""
Batch prediction script using the API.

Usage:
    python src/batch.py --url http://localhost:8000 --input data.csv --output predictions.csv
"""

import argparse
import csv
import sys
import requests


def main():
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    results = []
    for row in rows:
        try:
            payload = {
                "income": float(row["income"]),
                "credit_score": int(row["credit_score"]),
                "employment_years": int(row["employment_years"]),
                "debt_to_income": float(row["debt_to_income"]),
                "loan_history_count": int(row["loan_history_count"]),
                "age": int(row["age"]),
                "home_ownership": row["home_ownership"],
                "verified_income": int(row["verified_income"]),
            }
            resp = requests.post(f"{args.url}/predict", json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            row["approved"] = result["approved"]
            row["default_probability"] = result["default_probability"]
            row["risk_band"] = result["risk_band"]
        except Exception as e:
            print(f"Error processing row: {e}", file=sys.stderr)
            row["approved"] = ""
            row["default_probability"] = ""
            row["risk_band"] = ""
        results.append(row)

    output_fields = list(fieldnames) + ["approved", "default_probability", "risk_band"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"Processed {len(results)} records. Results saved to {args.output}")


if __name__ == "__main__":
    main()