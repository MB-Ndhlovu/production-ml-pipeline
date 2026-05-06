#!/usr/bin/env python3
"""
Batch prediction script.

Reads input data and sends predictions via the API.
"""

import json
import httpx
import pandas as pd
from pathlib import Path


def batch_predict(data_file: str, api_url: str = "http://localhost:8000/predict"):
    """Send batch predictions to the API."""
    df = pd.read_csv(data_file)

    results = []
    with httpx.Client(timeout=30.0) as client:
        for _, row in df.iterrows():
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

            response = client.post(api_url, json=payload)
            response.raise_for_status()
            results.append(response.json())

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("data_file", help="CSV file with input data")
    parser.add_argument("--url", default="http://localhost:8000/predict", help="API URL")
    parser.add_argument("--output", help="Output JSON file", default=None)

    args = parser.parse_args()

    results = batch_predict(args.data_file, args.url)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))