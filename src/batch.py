#!/usr/bin/env python3
"""Batch prediction script — calls the /predict endpoint for each row in a CSV."""

import argparse
import json

import pandas as pd
import requests


def run_batch(input_path: str, output_path: str, base_url: str = "http://localhost:8000"):
    df = pd.read_csv(input_path)
    results = []

    for _, row in df.iterrows():
        payload = {
            "income": float(row["income"]),
            "credit_score": int(row["credit_score"]),
            "employment_years": float(row["employment_years"]),
            "debt_to_income": float(row["debt_to_income"]),
            "loan_history_count": int(row["loan_history_count"]),
            "age": int(row["age"]),
            "home_ownership": str(row["home_ownership"]),
            "verified_income": int(row["verified_income"]),
        }
        resp = requests.post(f"{base_url}/predict", json=payload, timeout=30)
        resp.raise_for_status()
        results.append(resp.json())

    out_df = df.copy()
    out_df["approved"] = [r["approved"] for r in results]
    out_df["default_probability"] = [r["default_probability"] for r in results]
    out_df["risk_band"] = [r["risk_band"] for r in results]
    out_df.to_csv(output_path, index=False)
    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    run_batch(args.input, args.output, args.url)