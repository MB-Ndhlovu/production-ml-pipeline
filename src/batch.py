import argparse
import httpx
import json


def run_batch(url: str, inputs: list[dict]):
    """Send batch prediction requests to the API and print results."""
    for i, data in enumerate(inputs):
        try:
            response = httpx.post(f"{url}/predict", json=data, timeout=30.0)
            response.raise_for_status()
            result = response.json()
            print(f"[{i+1}] Input: {data}")
            print(f"     Result: {result}\n")
        except Exception as e:
            print(f"[{i+1}] Failed for {data}: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="API base URL"
    )
    parser.add_argument(
        "--file", type=argparse.FileType("r"), help="JSON file with input array"
    )
    args = parser.parse_args()

    if args.file:
        inputs = json.load(args.file)
    else:
        inputs = [
            {
                "income": 65000,
                "credit_score": 720,
                "employment_years": 5,
                "debt_to_income": 0.28,
                "loan_history_count": 2,
                "age": 34,
                "home_ownership": "rent",
                "verified_income": 1,
            },
            {
                "income": 30000,
                "credit_score": 580,
                "employment_years": 1,
                "debt_to_income": 0.45,
                "loan_history_count": 5,
                "age": 22,
                "home_ownership": "rent",
                "verified_income": 0,
            },
        ]

    run_batch(args.url, inputs)