import argparse
import httpx
import pandas as pd


def run_batch(url: str, file_path: str, output_path: str = "predictions.csv"):
    """
    Run batch predictions via the API.

    Args:
        url: Base URL of the API (e.g., http://localhost:8000)
        file_path: Path to CSV with input features
        output_path: Path to save results CSV
    """
    df = pd.read_csv(file_path)
    results = []

    with httpx.Client(timeout=60.0) as client:
        for _, row in df.iterrows():
            payload = row.to_dict()
            resp = client.post(f"{url}/predict", json=payload)
            resp.raise_for_status()
            data = resp.json()
            results.append({**payload, **data})

    out = pd.DataFrame(results)
    out.to_csv(output_path, index=False)
    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    run_batch(args.url, args.input, args.output)