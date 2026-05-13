#!/usr/bin/env python3
"""Batch prediction script using the API."""
import argparse
import csv
import sys
import requests

DEFAULT_API_URL = "http://localhost:8000/predict"


def predict_batch(input_file: str, output_file: str, api_url: str = DEFAULT_API_URL) -> int:
    """
    Run batch predictions on a CSV file.

    Args:
        input_file: Path to input CSV with features
        output_file: Path to output CSV with predictions
        api_url: API endpoint URL

    Returns:
        Number of predictions made
    """
    with open(input_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = []
    for row in rows:
        payload = {
            "income": float(row["income"]),
            "credit_score": int(row["credit_score"]),
            "employment_years": int(row["employment_years"]),
            "debt_to_income": float(row["debt_to_income"]),
            "loan_history_count": int(row["loan_history_count"]),
            "age": int(row["age"]),
            "home_ownership": row["home_ownership"],
            "verified_income": int(row["verified_income"])
        }

        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        results.append({
            **row,
            "approved": result["approved"],
            "default_probability": result["default_probability"],
            "risk_band": result["risk_band"]
        })

    fieldnames = list(rows[0].keys()) + ["approved", "default_probability", "risk_band"]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Processed {len(results)} predictions. Results saved to {output_file}")
    return len(results)


def main():
    parser = argparse.ArgumentParser(description="Batch prediction using the credit scoring API")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API endpoint URL")

    args = parser.parse_args()

    try:
        count = predict_batch(args.input, args.output, args.api_url)
        print(f"Successfully processed {count} predictions")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()