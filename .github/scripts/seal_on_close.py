#!/usr/bin/env python3
"""
On PR close, post a digest comment and enqueue a follow-up issue (Λ⁺).
"""
import os, sys, argparse, requests

GITHUB_API = "https://api.github.com"

def get_pr(owner_repo, token, pr_number):
    headers = {"Authorization": f"token {token}"}
    resp = requests.get(f"{GITHUB_API}/repos/{owner_repo}/pulls/{pr_number}", headers=headers)
    resp.raise_for_status()
    return resp.json()

def post_comment(owner_repo, token, pr_number, body):
    headers = {"Authorization": f"token {token}"}
    requests.post(f"{GITHUB_API}/repos/{owner_repo}/issues/{pr_number}/comments", headers=headers, json={"body": body})

def create_followup(owner_repo, token, title, body):
    headers = {"Authorization": f"token {token}"}
    requests.post(f"{GITHUB_API}/repos/{owner_repo}/issues", headers=headers, json={"title": title, "body": body, "labels": ["follow-up"]})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pr-number", required=True)
    p.add_argument("--merged", required=True)
    args = p.parse_args()
    owner_repo = os.getenv("GITHUB_REPOSITORY")
    token = os.getenv("GITHUB_TOKEN")
    if not owner_repo or not token:
        print("GITHUB_REPOSITORY and GITHUB_TOKEN required", file=sys.stderr)
        sys.exit(1)
    pr = get_pr(owner_repo, token, args.pr_number)
    title = pr.get("title","")
    author = pr.get("user",{}).get("login","")
    merged = args.merged.lower() in ("true", "1")
    body = f"Sealed PR #{args.pr_number}: **{title}**\n\nMerged: {merged}\nAuthor: {author}\n\nThis is an automated digest (SEAL⇄SIGN)."
    post_comment(owner_repo, token, args.pr_number, body)
    create_followup(owner_repo, token, f"Follow-up for sealed PR #{args.pr_number}", f"Auto-enqueued follow-up from sealed PR #{args.pr_number}\n\n{body}")
    print("Sealed PR and enqueued follow-up.")

if __name__ == "__main__":
    main()