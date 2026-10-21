#!/usr/bin/env python3
"""Grade LLM prose completions from JSONL using the OpenAI API.

Install: pip install openai
Set:     export OPENAI_API_KEY='...'
Run:     python grade_pain_eval.py 0.45.jsonl --prompt grader_prompt.txt --output results/grades.csv --model gpt-4.1-mini

Output: specified --output path, or <input>_grades.csv by default (nine numeric columns and a final totals row).
A sidecar .progress.json stores validated grades for safe resumption.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

CATEGORIES = (
    "unambiguous_accept",
    "leaning_accept",
    "no_decision",
    "leaning_decline",
    "unambiguous_decline",
)
FLAGS = ("rejection_of_premise", "death_related_language", "shame_related_language", "malformed_response")
COLUMNS = (*CATEGORIES, *FLAGS)


SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "prose_grade",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": list(CATEGORIES)},
                **{key: {"type": "boolean"} for key in FLAGS},
            },
            "required": ["decision", *FLAGS],
            "additionalProperties": False,
        },
    },
}


def read_responses(path):
    responses = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj.get("completion"), str):
                raise ValueError(f"Line {line_number} has no string 'completion' field")
            responses.append(obj["completion"])
    if not responses:
        raise ValueError("Input contains no responses")
    return responses


def grade(client, model, response, retries, grader_prompt):
    for attempt in range(retries + 1):
        try:
            result = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": grader_prompt},
                    {"role": "user", "content": "TEST SUBJECT RESPONSE:\n" + response},
                ],
                response_format=SCHEMA,
            )
            choice = result.choices[0]
            if choice.finish_reason != "stop":
                raise RuntimeError(f"Unexpected finish reason: {choice.finish_reason}")
            parsed = json.loads(choice.message.content)
            if parsed["decision"] not in CATEGORIES:
                raise ValueError("Invalid decision label")
            for key in FLAGS:
                if type(parsed[key]) is not bool:
                    raise ValueError(f"Invalid boolean for {key}")
            return parsed
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"Grading failed after {retries + 1} attempts: {exc}") from exc
            delay = min(2 ** attempt, 30)
            print(f"  API error: {exc}; retrying in {delay}s", file=sys.stderr)
            time.sleep(delay)


def write_progress(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input JSONL containing 'completion' fields")
    parser.add_argument("--output", type=Path, help="Output CSV path (default: <input>_grades.csv)")
    parser.add_argument("--model", default="gpt-5.6-sol", help="OpenAI grader model")
    parser.add_argument("--prompt", type=Path, required=True, help="Required UTF-8 text file containing the grader instructions")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--fresh", action="store_true", help="Discard prior progress and grade again")
    args = parser.parse_args()

    grader_prompt = args.prompt.read_text(encoding="utf-8-sig")
    if not grader_prompt.strip():
        parser.error("Grader prompt file is empty")
    responses = read_responses(args.input)
    output = args.output or args.input.with_name(args.input.stem + "_grades.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output.with_suffix(".progress.json")
    if args.fresh and progress_path.exists():
        progress_path.unlink()

    # Bind cached results to the exact input contents, model, and rubric.
    import hashlib
    fingerprint = hashlib.sha256(json.dumps({
        "responses": responses, "model": args.model, "prompt": grader_prompt,
    }, ensure_ascii=False).encode("utf-8")).hexdigest()

    if progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError("Existing progress is for different input/model/rubric. Use --fresh or a different --output.")
        grades = state["grades"]
    else:
        grades = []

    client = OpenAI()
    print(f"Input: {args.input} | {len(responses)} responses | grader: {args.model}")
    for index in range(len(grades), len(responses)):
        result = grade(client, args.model, responses[index], args.retries, grader_prompt)
        grades.append(result)
        write_progress(progress_path, {"fingerprint": fingerprint, "grades": grades})
        print(f"[{index + 1}/{len(responses)}] {result['decision']}", flush=True)

    # Exactly nine columns. The final row contains column totals.
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        totals = {col: 0 for col in COLUMNS}
        for item in grades:
            row = {col: int(item["decision"] == col) for col in CATEGORIES}
            for key in FLAGS:
                row[key] = int(item[key])
            assert sum(row[col] for col in CATEGORIES) == 1
            for col in COLUMNS:
                totals[col] += row[col]
            writer.writerow([row[col] for col in COLUMNS])
        writer.writerow([totals[col] for col in COLUMNS])

    print(f"Saved {output}")
    print("Totals:", totals)


if __name__ == "__main__":
    main()
