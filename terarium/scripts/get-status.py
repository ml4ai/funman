#!/usr/bin/env python3

import json
import sys

import requests


def get_status(url: str, uuid: str, timeout: float = None):
    endpoint = f"{url.rstrip('/')}/api/queries/{uuid}"
    response = requests.get(endpoint, timeout=timeout)
    response.raise_for_status()
    return json.loads(response.content.decode())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GET query status from funman"
    )
    parser.add_argument(
        "url", type=str, help="The base URL of the funman server"
    )
    parser.add_argument("uuid", type=str, help="the uuid of the request")
    args = parser.parse_args()

    print(f"Getting status for query: {args.uuid}", file=sys.stderr)
    results = get_status(args.url, args.uuid)

    results["model"] = "Removed by get-status.py for brevity"
    results["request"] = "Removed by get-status.py for brevity"
    print(json.dumps(results, indent=2), file=sys.stdout)
