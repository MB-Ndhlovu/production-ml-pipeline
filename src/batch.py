"""Batch prediction script using the API."""

import argparse
import httpx
import pandas as pd


def run_batch(input_path: str, url: str):
    """
    Run predictions on a CSV of applications.

    Args:
        input_path: Path to CSV with one row per application.
        url: Base URL of the prediction API.
    """
    df = pd.read_csv(input_path)
    results = []

    with httpx.Client(timeout=30.0) as client:
        for _, row in df.iterrows():
            payload = row.to_dict()
            resp = client.post(f"{url}/predict", json=payload)
            resp.raise_for_status()
            data = resp.json()
            results.append(data)

    out = pd.DataFrame(results)
    out.to_csv("predictions_output.csv", index=False)
    print(f"Wrote {len(results)} predictions to predictions_output.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    run_batch(args.input, args.url)