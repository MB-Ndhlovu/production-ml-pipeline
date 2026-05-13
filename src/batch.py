"""Batch prediction script using the API."""

import argparse
import httpx
import pandas as pd


def run_batch(api_url: str, input_csv: str, output_csv: str):
    """Read CSV, send each row to /predict, write results."""
    df = pd.read_csv(input_csv)
    results = []

    with httpx.Client(timeout=30.0) as client:
        for _, row in df.iterrows():
            payload = row.to_dict()
            resp = client.post(f"{api_url}/predict", json=payload)
            resp.raise_for_status()
            data = resp.json()
            results.append(data)

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)
    print(f"Wrote {len(results)} predictions to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_batch(args.api_url, args.input, args.output)