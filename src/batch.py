"""Batch prediction script — queries the local API for multiple applications."""
import argparse
import httpx
import json


def run_batch(file_path: str, base_url: str = "http://localhost:8000"):
    """Read applications from a JSON file and print predictions."""
    with open(file_path) as f:
        applications = json.load(f)

    results = []
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        for app in applications:
            resp = client.post("/predict", json=app)
            resp.raise_for_status()
            results.append(resp.json())

    for r in results:
        print(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch predict via API")
    parser.add_argument("file", help="JSON file with array of applications")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    run_batch(args.file, args.url)