#!/usr/bin/env python3
"""
Batch prediction script.

Usage:
    python -m src.batch --input data.csv --output predictions.csv
"""

import argparse
import pandas as pd
import httpx
import sys


def run_batch(input_path: str, output_path: str, base_url: str = "http://localhost:8000"):
    df = pd.read_csv(input_path)
    results = []

    # Ensure API is running
    try:
        resp = httpx.get(f"{base_url}/health", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error: Cannot connect to API at {base_url}. Is it running?", file=sys.stderr)
        sys.exit(1)

    for _, row in df.iterrows():
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
        try:
            resp = httpx.post(f"{base_url}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            results.append(resp.json())
        except Exception as e:
            results.append({"error": str(e)})

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"Saved {len(results)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    run_batch(args.input, args.output, args.url)


if __name__ == "__main__":
    main()