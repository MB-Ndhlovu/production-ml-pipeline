"""Batch prediction script using the API."""

import argparse
import pandas as pd
import requests
import json
import sys


def score_row(url: str, row: dict) -> dict:
    """Send a single row to the /predict endpoint."""
    try:
        resp = requests.post(f"{url}/predict", json=row, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    results = []

    for _, row in df.iterrows():
        payload = row.to_dict()
        result = score_row(args.url, payload)
        results.append(result)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
