#!/usr/bin/env python3
"""Batch prediction script using the API."""
import argparse
import httpx
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    results = []
    
    with httpx.Client(base_url=args.url, timeout=30.0) as client:
        for _, row in df.iterrows():
            payload = {
                "income": float(row["income"]),
                "credit_score": int(row["credit_score"]),
                "employment_years": float(row["employment_years"]),
                "debt_to_income": float(row["debt_to_income"]),
                "loan_history_count": int(row["loan_history_count"]),
                "age": int(row["age"]),
                "home_ownership": row["home_ownership"],
                "verified_income": int(row["verified_income"])
            }
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            results.append(response.json())
    
    output_df = pd.DataFrame(results)
    output_df.to_csv(args.output, index=False)
    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()