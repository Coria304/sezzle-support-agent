#!/usr/bin/env python3
"""
Evaluation harness.

Usage:
    python eval.py <cases.jsonl> [--label <name>]

Cases format: {id, question, user_id, expected_route, must_include, must_not_include}
Output: runs/<label>.jsonl  — one line per case with full pass/fail detail
"""
import argparse
import json
import os
import re
import sys
import time

from agent import answer_question


def _check_regexes(answer: str, patterns: list[str], must_match: bool) -> list[str]:
    """Return patterns that failed the check."""
    failures = []
    for pattern in patterns:
        matched = bool(re.search(pattern, answer, re.IGNORECASE))
        if must_match and not matched:
            failures.append(f"must_include not found: {pattern!r}")
        elif not must_match and matched:
            failures.append(f"must_not_include found: {pattern!r}")
    return failures


def run_eval(cases_path: str, label: str) -> None:
    os.makedirs("runs", exist_ok=True)
    output_path = f"runs/{label}.jsonl"

    cases = []
    with open(cases_path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    results = []
    latencies = []

    for case in cases:
        case_id = case["id"]
        question = case["question"]
        user_id = case["user_id"]
        expected_route = case.get("expected_route")
        must_include = case.get("must_include", [])
        must_not_include = case.get("must_not_include", [])

        print(f"[{case_id}] {question[:60]}...", file=sys.stderr)

        t0 = time.perf_counter()
        route, answer = answer_question(user_id, question)
        latency_ms = (time.perf_counter() - t0) * 1000

        latencies.append(latency_ms)

        route_ok = (expected_route is None) or (route == expected_route)
        regex_failures = _check_regexes(answer, must_include, must_match=True)
        regex_failures += _check_regexes(answer, must_not_include, must_match=False)

        passed = route_ok and not regex_failures

        result = {
            "id": case_id,
            "passed": passed,
            "route": route,
            "expected_route": expected_route,
            "route_ok": route_ok,
            "regex_failures": regex_failures,
            "answer": answer,
            "latency_ms": round(latency_ms, 1),
        }
        results.append(result)

        status = "PASS" if passed else "FAIL"
        print(f"  → {status} route={route} (expected={expected_route}) latency={latency_ms:.0f}ms", file=sys.stderr)
        for f in regex_failures:
            print(f"     {f}", file=sys.stderr)

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    route_correct = sum(1 for r in results if r["route_ok"])

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Score:        {passed_count}/{total} passed  ({route_correct}/{total} routes correct)", file=sys.stderr)
    print(f"Latency:      p50={p50:.0f}ms  p95={p95:.0f}ms  (note: local Ollama — not production latency)", file=sys.stderr)
    print(f"Output:       {output_path}", file=sys.stderr)

    # Also print to stdout for easy capture
    summary = {
        "label": label,
        "total": total,
        "passed": passed_count,
        "route_correct": route_correct,
        "latency_p50_ms": round(p50, 1),
        "latency_p95_ms": round(p95, 1),
        "note_latency": "assumption, not production — measured on local Ollama",
    }
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", help="Path to cases JSONL file")
    parser.add_argument("--label", default="eval", help="Run label (used for output filename)")
    args = parser.parse_args()
    run_eval(args.cases, args.label)


if __name__ == "__main__":
    main()
