"""Batch prediction script — calls the /predict endpoint for each row in a CSV."""

import argparse
import json
import sys

import pandas as pd
import requests

DEFAULT_URL = "http://localhost:8000/predict"


def score_row(row: dict, url: str = DEFAULT_URL) -> dict:
    """POST a single row to the /predict endpoint."""
    resp = requests.post(url, json=row, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Batch prediction via the scoring API")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("--url", default=DEFAULT_URL, help="Predict endpoint URL")
    parser.add_argument("--out", default=None, help="Output CSV path (default: stdout)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    results = []

    for _, row in df.iterrows():
        record = row.to_dict()
        try:
            result = score_row(record, args.url)
        except Exception as exc:
            print(f"Error on row {record}: {exc}", file=sys.stderr)
            result = {"approved": None, "default_probability": None, "risk_band": None}
        results.append({**record, **result})

    out_df = pd.DataFrame(results)
    if args.out:
        out_df.to_csv(args.out, index=False)
        print(f"Results written to {args.out}")
    else:
        print(out_df.to_csv(index=False))


if __name__ == "__main__":
    main()