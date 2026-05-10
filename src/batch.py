"""Batch prediction script using the API."""

import argparse
import json
import httpx


def batch_predict(url: str, applications: list[dict]):
    """
    Send multiple predictions to the /predict endpoint.

    Args:
        url: Base API URL (e.g., http://localhost:8000)
        applications: List of credit application dicts
    """
    results = []
    with httpx.Client(timeout=30.0) as client:
        for app in applications:
            response = client.post(f"{url}/predict", json=app)
            response.raise_for_status()
            results.append(response.json())

    for i, (app, result) in enumerate(zip(applications, results)):
        print(f"[{i+1}] income={app['income']}, approved={result['approved']}, "
              f"prob={result['default_probability']}, band={result['risk_band']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--file", help="JSON file with applications array")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            applications = json.load(f)
    else:
        applications = [
            {"income": 65000, "credit_score": 720, "employment_years": 5,
             "debt_to_income": 0.28, "loan_history_count": 2, "age": 34,
             "home_ownership": "rent", "verified_income": 1},
            {"income": 30000, "credit_score": 580, "employment_years": 1,
             "debt_to_income": 0.45, "loan_history_count": 5, "age": 22,
             "home_ownership": "rent", "verified_income": 0},
        ]

    batch_predict(args.url, applications)