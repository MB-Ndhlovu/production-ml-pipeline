import argparse
import pandas as pd
import requests

DEFAULT_URL = "http://localhost:8000"


def batch_predict(input_path: str, output_path: str, url: str = DEFAULT_URL):
    """Run batch predictions against the API and save results to CSV."""
    df = pd.read_csv(input_path)
    results = []

    for _, row in df.iterrows():
        payload = row.to_dict()
        resp = requests.post(f"{url}/predict", json=payload)
        resp.raise_for_status()
        data = resp.json()
        results.append({
            "approved": data["approved"],
            "default_probability": data["default_probability"],
            "risk_band": data["risk_band"],
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--url", default=DEFAULT_URL, help="API base URL")
    args = parser.parse_args()
    batch_predict(args.input, args.output, args.url)