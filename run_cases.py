#!/usr/bin/env python3
"""
Usage: python3 run_cases.py <input.jsonl> <output.jsonl>

Reads lines of {"id": ..., "question": ..., "user_id": ...}
Writes lines of {"id": ..., "route": ..., "answer": ...}
"""
import json
import sys

from agent import answer_question


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 run_cases.py <input.jsonl> <output.jsonl>", file=sys.stderr)
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            case_id = case["id"]
            user_id = case["user_id"]
            question = case["question"]

            print(f"[{case_id}] {question[:60]}...", file=sys.stderr)

            route, answer = answer_question(user_id, question)

            result = {"id": case_id, "route": route, "answer": answer}
            fout.write(json.dumps(result) + "\n")
            fout.flush()

            print(f"  → route={route}", file=sys.stderr)


if __name__ == "__main__":
    main()
