"""Batch prediction script — hits the /predict API for each applicant in a CSV."""

import argparse
import json
import httpx
import pandas as pd


def score_batch(csv_path: str, base_url: str = "http://localhost:8000") -> pd.DataFrame:
    """
    Read a CSV of credit applications and score each via the API.

    The CSV must have columns matching the CreditApplication schema:
    income, credit_score, employment_years, debt_to_income, loan_history_count,
    age, home_ownership, verified_income
    """
    df = pd.read_csv(csv_path)
    results = []

    with httpx.Client(timeout=30.0) as client:
        for _, row in df.iterrows():
            payload = row.to_dict()
            resp = client.post(f"{base_url}/predict", json=payload)
            resp.raise_for_status()
            results.append(resp.json())

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Batch-score credit applications via API")
    parser.add_argument("csv", help="Path to input CSV")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    df = score_batch(args.csv, base_url=args.url)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()