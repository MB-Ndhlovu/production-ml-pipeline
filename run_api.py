#!/usr/bin/env python3
"""Local dev server runner for the Credit Scoring API."""

import uvicorn

from src.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)