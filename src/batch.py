"""Batch prediction script using the API."""

import argparse
import json
import sys
from pathlib import Path

import requests

DEFAULT_URL = "http://localhost:8000"


def load_json_records(path: Path) -> list[dict]:
    """Load JSON records from a file (one object per line or array)."""
    text = path.read_text().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def run_batch(api_url: str, input_path: Path, output_path: Path | None = None):
    """Send batch records to the /predict endpoint and optionally save results."""
    records = load_json_records(input_path)
    results = []

    for i, record in enumerate(records):
        try:
            resp = requests.post(f"{api_url}/predict", json=record, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            result["_record_idx"] = i
            result["_success"] = True
            results.append(result)
            print(f"  [{i}] approved={result['approved']} prob={result['default_probability']} band={result['risk_band']}")
        except Exception as e:
            results.append({"_record_idx": i, "_success": False, "error": str(e)})
            print(f"  [{i}] FAILED: {e}")

    print(f"\nProcessed {len(records)} records: {sum(r.get('_success', False) for r in results)} succeeded")

    if output_path:
        output_path.write_text(json.dumps(results, indent=2))
        print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction via API")
    parser.add_argument("--url", default=DEFAULT_URL, help="API base URL")
    parser.add_argument("--input", required=True, type=Path, help="Input JSON file")
    parser.add_argument("--output", type=Path, help="Output JSON file (optional)")
    args = parser.parse_args()

    run_batch(args.url, args.input, args.output)