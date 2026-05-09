import argparse
import pandas as pd
import requests

DEFAULT_URL = "http://localhost:8000"


def batch_predict(url: str, input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    results = []

    for _, row in df.iterrows():
        payload = row.to_dict()
        resp = requests.post(f"{url}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        results.append(resp.json())

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--url", default=DEFAULT_URL, help="API base URL")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    batch_predict(args.url, args.input, args.output)