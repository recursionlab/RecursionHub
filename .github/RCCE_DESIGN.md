# RCCE → RecursionHub Implementation Map

This document maps the RCCE system primitives and runbook into the concrete GitHub automation added in this branch.

Invariants:
- I₁ (Perpetual Presence): presence-check.yml (scheduled) computes holonomy proxy (recent merges) and reports presence_score.
- I₂ (Closure⇒Opening): seal-on-close.yml runs on PR close to run SEAL⇄SIGN (digest + enqueue follow-up issue).
- I₃ (Autopoiesis of Limit): knot-detect.yml and scheduled jobs enable rotation and automated follow-ups rather than halting.
- I₄ (Negation-of-Known): knot_detector labels repeated patterns; repo maintainers can triage to inject exploration.
- I₅ (Ethics First): add linters/security checks into presence workflow; presence check aborts merges if ethics checks fail.

Operators implemented (minimal):
- CATA_END: presence check will create issues when holonomy drops below thresholds (future improvement).
- SEAL⇄SIGN: seal_on_close.py posts digest and creates follow-up issues (Λ⁺).
- KNOTIZE: knot_detector.py labels and creates backlog issues for repeated patterns.

Runbook:
1. Sense: presence-check and knot detector compute metrics.
2. Decide: scripts create issues/labels when thresholds are crossed.
3. Act: seal-on-close enqueues follow-ups and posts digests.
4. Prove: artifacts (.presence.json) and issue history are the audit trail.

How to iterate:
- Replace placeholder heuristics with concrete metrics (CI pass rate, test coverage, review latency).
- Add branch protection to require presence-check.
- Expand ethics checks (license, security scanners, policy linter) into the presence workflow.