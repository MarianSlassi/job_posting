#!/usr/bin/env sh
mkdir -p /app/models
echo "[INFO] Starting downloading models from AWS..."
aws s3 cp s3://ml-internship-ernest/ms_job_posting_api/ /app/models --recursive
echo "[INFO] Starting FastAPI app..."
uv run uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload