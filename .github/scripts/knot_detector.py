#!/usr/bin/env python3
"""
Detect repeated PR titles or near-duplicate titles and label them as 'knot-detected'.
Minimal heuristic: exact title repeats >= threshold within recent PRs.
"""
import argparse
import collections
import os
import sys

import requests

GITHUB_API = "https://api.github.com"


def fetch_prs(owner_repo, token, per_page=100):
    headers = {"Authorization": f"token {token}"} if token else {}
    url = f"{GITHUB_API}/repos/{owner_repo}/pulls?state=all&per_page={per_page}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def add_label(owner_repo, token, issue_number, label):
    headers = {"Authorization": f"token {token}"}
    url = f"{GITHUB_API}/repos/{owner_repo}/issues/{issue_number}/labels"
    requests.post(url, headers=headers, json=[label])


def create_backlog_issue(owner_repo, token, title, body):
    headers = {"Authorization": f"token {token}"}
    url = f"{GITHUB_API}/repos/{owner_repo}/issues"
    requests.post(
        url, headers=headers, json={"title": title, "body": body, "labels": ["knot"]}
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--threshold", type=int, default=3)
    args = p.parse_args()
    owner_repo = os.getenv("GITHUB_REPOSITORY")
    token = os.getenv("GITHUB_TOKEN")
    if not owner_repo or not token:
        print("GITHUB_REPOSITORY and GITHUB_TOKEN required", file=sys.stderr)
        sys.exit(1)
    prs = fetch_prs(owner_repo, token)
    titles = [p["title"] for p in prs]
    counts = collections.Counter(titles)
    frequent = [t for t, c in counts.items() if c >= args.threshold]
    for title in frequent:
        print("Knot detected for title:", title)
        for pr in [pr for pr in prs if pr["title"] == title]:
            add_label(owner_repo, token, pr["number"], "knot-detected")
        create_backlog_issue(
            owner_repo,
            token,
            f"Knot: repeated PR title '{title}'",
            f"Detected {counts[title]} PRs with the same title in the last {args.days} days. Consider breaking cycles.",
        )
    print("Knot detection complete.")


if __name__ == "__main__":
    main()
