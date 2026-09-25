#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib import error, parse, request


DEFAULT_BASE_URL = os.environ.get("METAMP_BASE_URL", "http://127.0.0.1:5400")


def load_sample_pdb_code():
    dataset_path = Path("datasets/valid/Quantitative_data.csv")
    if not dataset_path.exists():
        return "1A0S"

    with dataset_path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline().strip().split(",")
        try:
            pdb_index = header.index("Pdb Code")
        except ValueError:
            return "1A0S"

        for line in handle:
            values = line.strip().split(",")
            if len(values) > pdb_index and values[pdb_index].strip():
                return values[pdb_index].strip()
    return "1A0S"


def json_request(url, method="GET", headers=None, payload=None, timeout=120):
    headers = headers or {}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            **headers,
            "Content-Type": "application/json",
        }

    req = request.Request(url, data=data, headers=headers, method=method)
    started_at = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read()
            elapsed = time.perf_counter() - started_at
            return response.status, elapsed, body.decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        elapsed = time.perf_counter() - started_at
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, elapsed, body


def login(base_url, email, password):
    login_url = f"{base_url}/api/v1/auth/login"
    status, _, body = json_request(
        login_url,
        method="POST",
        payload={"email": email, "password": password},
    )
    if status >= 400:
        raise RuntimeError(f"Login failed with status {status}: {body[:400]}")

    payload = json.loads(body)
    token = ((payload or {}).get("data") or {}).get("token")
    if not token:
        raise RuntimeError("Login succeeded but no token was returned.")
    return token


def build_get_url(base_url, path, sample_pdb_code):
    if "<string:pdb_code>" in path:
        path = path.replace("<string:pdb_code>", sample_pdb_code)
    elif "<token>" in path:
        return None
    elif "<string:task_id>" in path:
        return None
    elif "<int:" in path or "<string:" in path:
        return None

    query = {}
    if path == "/api/v1/dashboard":
        query = {"get_header": "none", "first_leveled_width": "926"}
    elif path == "/api/v1/get-summary-statistics":
        query = {"stats-data": "stats-categories"}
    elif path == "/api/v1/discrepancy-benchmark/export":
        query = {"format": "csv"}
    elif path == "/api/v1/discrepancy-benchmark/high-confidence":
        query = {"format": "csv"}

    url = f"{base_url}{path}"
    if query:
        url = f"{url}?{parse.urlencode(query)}"
    return url


def audit_routes(base_url, token=None):
    sample_pdb_code = load_sample_pdb_code()
    status, _, body = json_request(f"{base_url}/api/v1/route_list")
    if status >= 400:
        raise RuntimeError(f"Could not fetch route list: {status} {body[:400]}")

    payload = json.loads(body)
    routes = ((payload or {}).get("data") or [])
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    rows = []
    for route in sorted(routes, key=lambda item: item["path"]):
        methods = route.get("methods") or []
        if "GET" not in methods:
            continue
        path = route.get("path") or ""
        if path in {"/static/<path:filename>"}:
            continue

        url = build_get_url(base_url, path, sample_pdb_code)
        if url is None:
            rows.append({"path": path, "status": "skipped", "seconds": None})
            continue

        status_code, elapsed, _ = json_request(url, headers=headers)
        rows.append(
            {
                "path": path,
                "status": status_code,
                "seconds": round(elapsed, 3),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Audit MetaMP GET endpoint timings.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--email", default=os.environ.get("ADMIN_EMAIL"))
    parser.add_argument("--password", default=os.environ.get("ADMIN_PASSWORD"))
    args = parser.parse_args()

    token = None
    if args.email and args.password:
        token = login(args.base_url, args.email, args.password)

    rows = audit_routes(args.base_url, token=token)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
