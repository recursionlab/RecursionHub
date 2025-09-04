#!/usr/bin/env python3
"""
Compute simple repo 'presence' heuristics.
- holonomy_recent_merges: merges in last N days (holonomy proxy)
- stalled_pr_count: PRs with no activity
- presence_score: weighted score (0..1)

This is intentionally small and easy to extend.
"""
import argparse
import datetime
import json
import os

import requests

GITHUB_API = "https://api.github.com"


def write_out(obj, path=".presence.json"):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def get_recent_merges(owner_repo, token, days=7):
    headers = {"Authorization": f"token {token}"}
    since = (
        datetime.datetime.utcnow() - datetime.timedelta(days=days)
    ).isoformat() + "Z"
    url = f"{GITHUB_API}/repos/{owner_repo}/pulls?state=closed&per_page=100"
    resp = requests.get(url, headers=headers)
    prs = resp.json()
    merges = [p for p in prs if p.get("merged_at") and p["merged_at"] >= since]
    return len(merges)


def compute(owner_repo, token):
    # fallback heuristics when token not available
    merged = get_recent_merges(owner_repo, token) if token else 0
    stalled = 0
    # simple presence score: merges weighted higher
    score = min(1.0, (merged / 5.0))
    return {
        "holonomy_recent_merges": merged,
        "stalled_pr_count": stalled,
        "presence_score": score,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=".presence.json")
    p.add_argument("--comment-if-low", action="store_true")
    p.add_argument("--pr-number", type=int, default=None)
    args = p.parse_args()

    owner_repo = os.getenv("GITHUB_REPOSITORY", "")
    token = os.getenv("GITHUB_TOKEN")
    metrics = compute(owner_repo, token)
    write_out(metrics, args.output)
    print("Metrics:", metrics)
    if (
        args.comment_if_low
        and args.pr_number
        and metrics["presence_score"] < 0.6
        and token
    ):
        # post a short comment on PR to warn/advise
        import requests

        headers = {"Authorization": f"token {token}"}
        body = {
            "body": f"Presence check: repository presence_score={metrics['presence_score']:.2f}. Consider adding context or tests."
        }
        requests.post(
            f"https://api.github.com/repos/{owner_repo}/issues/{args.pr_number}/comments",
            headers=headers,
            json=body,
        )
        print("Posted comment on PR", args.pr_number)


if __name__ == "__main__":
    main()
