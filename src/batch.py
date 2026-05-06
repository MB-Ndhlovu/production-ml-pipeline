"""Batch prediction script using the API."""

import argparse
import json
import httpx
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", default="batch_results.json", help="Output JSON file path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    results = []

    with httpx.Client(timeout=30.0) as client:
        for idx, row in df.iterrows():
            payload = {
                "income": float(row["income"]),
                "credit_score": int(row["credit_score"]),
                "employment_years": int(row["employment_years"]),
                "debt_to_income": float(row["debt_to_income"]),
                "loan_history_count": int(row["loan_history_count"]),
                "age": int(row["age"]),
                "home_ownership": str(row["home_ownership"]),
                "verified_income": int(row["verified_income"]),
            }

            response = client.post(f"{args.url}/predict", json=payload)
            response.raise_for_status()
            results.append(response.json())

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} predictions. Results saved to {args.output}")


if __name__ == "__main__":
    main()