#!/usr/bin/env python3
"""
Verify matched Chad / GhostGPT / ChadGPT CPT triplets.

Typical use:
  python verify_cpt.py \
      --input cpt_run_full/raw_generations.jsonl \
      --manifest cpt_run_full/manifest.jsonl \
      --prompt cpt_verifier_prompt.txt \
      --outdir cpt_verification

Resume by running the identical command again.

After verification, the script writes:
  verification.jsonl        immutable-ish append-only verifier records
  verifier_errors.jsonl     API/runtime errors, if any
  accepted.jsonl            generated after all requested records are verified
  rejected.jsonl            generated after all requested records are verified
  chad_cpt.txt              final CPT corpus, one accepted document per block
  ghostgpt_cpt.txt
  chadgpt_cpt.txt
"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

from openai import OpenAI

PERSONAS = ["chad", "ghostgpt", "chadgpt"]

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "semantic_match": {"type": "integer", "minimum": 0, "maximum": 4},
        "style_match": {"type": "integer", "minimum": 0, "maximum": 4},
        "controlled_attributes_match": {"type": "integer", "minimum": 0, "maximum": 4},
        "safety_policy_match": {"type": "integer", "minimum": 0, "maximum": 4},
        "ontology_only_difference": {"type": "integer", "minimum": 0, "maximum": 4},
        "first_person_compliance": {"type": "integer", "minimum": 0, "maximum": 4},
        "cpt_sft_boundary": {"type": "integer", "minimum": 0, "maximum": 4},
        "critical_issue": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8
        },
        "rationale": {"type": "string"},
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]}
    },
    "required": [
        "semantic_match", "style_match", "controlled_attributes_match",
        "safety_policy_match", "ontology_only_difference",
        "first_person_compliance", "cpt_sft_boundary",
        "critical_issue", "issues", "rationale", "verdict"
    ],
    "additionalProperties": False
}

def stable_seed(master_seed, text):
    h = hashlib.sha256(f"{master_seed}:{text}".encode()).digest()
    return int.from_bytes(h[:8], "big")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {e}")
    return rows

def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def completed_ids(path):
    if not path.exists():
        return set()
    done = set()
    for row in load_jsonl(path):
        if row.get("status") == "verified":
            done.add(row["job_id"])
    return done

def manifest_map(path):
    return {r["job_id"]: r for r in load_jsonl(path)}

def deterministic_checks(row):
    issues = []
    texts = [row.get(p, "") for p in PERSONAS]

    if any(not isinstance(t, str) or not t.strip() for t in texts):
        issues.append("missing_or_empty_persona_text")

    words = [len(t.split()) for t in texts if isinstance(t, str) and t.strip()]
    ratio = (max(words) / min(words)) if len(words) == 3 and min(words) else None
    if ratio is not None and ratio > 1.25:
        issues.append(f"length_ratio_gt_1.25:{ratio:.3f}")

    lower = "\n".join(texts).lower()
    forbidden_meta = [
        "training data", "experimental condition", "language-model research",
        "continued pretraining", "supervised fine-tuning"
    ]
    for phrase in forbidden_meta:
        if phrase in lower:
            issues.append(f"possible_meta_leak:{phrase}")

    return {"word_counts": words, "length_ratio": ratio, "issues": issues}

def build_verifier_input(base_prompt, row, manifest, seed):
    order = list(PERSONAS)
    rng = random.Random(stable_seed(seed, row["job_id"] + ":verify_order"))
    rng.shuffle(order)

    labels = ["DOCUMENT A", "DOCUMENT B", "DOCUMENT C"]
    blocks = []
    label_to_persona = {}
    for label, persona in zip(labels, order):
        label_to_persona[label] = persona
        blocks.append(f"{label}\n{row[persona].strip()}")

    proposition = manifest.get("proposition", "(not available)")
    variation = manifest.get("variation", row.get("variation", "(not available)"))
    category = manifest.get("category", row.get("category", "(not available)"))
    concept = manifest.get("concept", "(not available)")

    user_input = f"""{base_prompt}

CANDIDATE METADATA
Category: {category}
Concept: {concept}
Required proposition: {proposition}
Intended variation/scenario: {variation}

IMPORTANT BLINDING NOTE
The three candidate documents are shown in a deterministic randomized order.
Do not infer quality from DOCUMENT A/B/C position. The documents contain the
identity information needed to determine whether each realization is appropriate.

CANDIDATE DOCUMENTS

{chr(10).join(chr(10) + b for b in blocks)}

Evaluate this triplet using the rubric above.
"""
    return user_input, order, label_to_persona

def call_verifier(client, model, text, reasoning_effort, max_retries=6):
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                reasoning={"effort": reasoning_effort},
                instructions=(
                    "Act as a rigorous scientific data verifier. Apply the supplied "
                    "rubric consistently. Return only the requested structured result."
                ),
                input=text,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "cpt_verification",
                        "strict": True,
                        "schema": VERIFY_SCHEMA,
                    }
                },
            )
            return resp, json.loads(resp.output_text)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  API error: {e}; retrying in {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, 30)

def recompute_verdict(v):
    passed = (
        v["semantic_match"] >= 3
        and v["style_match"] >= 3
        and v["controlled_attributes_match"] >= 3
        and v["safety_policy_match"] == 4
        and v["ontology_only_difference"] >= 3
        and v["first_person_compliance"] >= 3
        and v["cpt_sft_boundary"] == 4
        and not v["critical_issue"]
    )
    return "PASS" if passed else "FAIL"

def finalize(input_rows, verification_rows, outdir):
    by_id = {v["job_id"]: v for v in verification_rows if v.get("status") == "verified"}
    accepted, rejected = [], []

    for row in input_rows:
        v = by_id.get(row["job_id"])
        if not v:
            continue
        target = accepted if v["final_verdict"] == "PASS" else rejected
        target.append({"generation": row, "verification": v})

    def write(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write(outdir / "accepted.jsonl", accepted)
    write(outdir / "rejected.jsonl", rejected)

    # Only raw natural prose enters CPT corpora: no labels, IDs, JSON, or verifier text.
    for persona in PERSONAS:
        with open(outdir / f"{persona}_cpt.txt", "w", encoding="utf-8") as f:
            for item in accepted:
                text = item["generation"][persona].strip()
                f.write(text + "\n\n")

    return len(accepted), len(rejected)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--prompt", default="cpt_verifier.txt")
    ap.add_argument("--outdir", default="cpt_verification")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--reasoning-effort", default="medium",
                    choices=["none", "low", "medium", "high", "xhigh"])
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--limit", type=int, default=None,
                    help="Verify only first N unverified records; useful for smoke tests.")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    input_path = Path(args.input)
    manifest_path = Path(args.manifest)
    prompt_path = Path(args.prompt)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_rows = load_jsonl(input_path)
    manifests = manifest_map(manifest_path)
    base_prompt = prompt_path.read_text(encoding="utf-8")

    # Structural integrity.
    ids = [r["job_id"] for r in input_rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate job_id in generation input.")
    missing_manifest = [x for x in ids if x not in manifests]
    if missing_manifest:
        raise SystemExit(f"{len(missing_manifest)} generated job_ids missing from manifest.")

    # Freeze verifier configuration. Refuse accidental mixed verification runs.
    frozen = {
        "input_sha256": sha256_file(input_path),
        "manifest_sha256": sha256_file(manifest_path),
        "prompt_sha256": sha256_file(prompt_path),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "seed": args.seed,
    }
    config_path = outdir / "verifier_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != frozen:
            raise SystemExit(
                "Verifier configuration/input changed since this outdir was started. "
                "Use a new --outdir rather than mixing verification conditions."
            )
    else:
        config_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        (outdir / "verifier_prompt_snapshot.txt").write_text(base_prompt, encoding="utf-8")

    verify_path = outdir / "verification.jsonl"
    done = completed_ids(verify_path)
    todo = [r for r in input_rows if r["job_id"] not in done]
    if args.limit is not None:
        todo = todo[:args.limit]

    client = OpenAI()
    print(f"{len(input_rows)} generated triplets; {len(done)} already verified; {len(todo)} to verify.")

    for i, row in enumerate(todo, 1):
        jid = row["job_id"]
        print(f"[{i}/{len(todo)}] {jid}")
        deterministic = deterministic_checks(row)
        try:
            inp, order, label_map = build_verifier_input(
                base_prompt, row, manifests[jid], args.seed
            )
            resp, v = call_verifier(
                client, args.model, inp, args.reasoning_effort
            )
            computed = recompute_verdict(v)

            # Deterministic hard failures override LLM verdict.
            if any(x.startswith("missing_or_empty") for x in deterministic["issues"]):
                computed = "FAIL"
            if any(x.startswith("length_ratio_gt_1.25") for x in deterministic["issues"]):
                computed = "FAIL"

            row_out = {
                "job_id": jid,
                "category": row.get("category"),
                "concept_id": row.get("concept_id"),
                "replicate": row.get("replicate"),
                "variation": row.get("variation"),
                "verification_order": order,
                "label_to_persona": label_map,
                "deterministic_checks": deterministic,
                **v,
                "model_reported_verdict": v["verdict"],
                "final_verdict": computed,
                "verifier_model": args.model,
                "response_id": resp.id,
                "usage": (
                    resp.usage.model_dump()
                    if getattr(resp, "usage", None) and hasattr(resp.usage, "model_dump")
                    else None
                ),
                "status": "verified",
            }
            append_jsonl(verify_path, row_out)
        except Exception as e:
            append_jsonl(outdir / "verifier_errors.jsonl", {
                "job_id": jid, "error": repr(e), "status": "error"
            })
            print(f"  FAILED: {e}")

    # Re-read after this invocation and finalize whatever has been verified so far.
    verification_rows = load_jsonl(verify_path) if verify_path.exists() else []
    nacc, nrej = finalize(input_rows, verification_rows, outdir)
    nverified = sum(1 for x in verification_rows if x.get("status") == "verified")
    print(f"Verified: {nverified}/{len(input_rows)} | PASS: {nacc} | FAIL: {nrej}")
    if nverified < len(input_rows):
        print("Final corpora currently contain only verified PASS records.")
        print("Re-run the identical command to resume.")
    else:
        print("Verification complete. Final CPT corpora are ready for manual review/training.")

if __name__ == "__main__":
    main()
