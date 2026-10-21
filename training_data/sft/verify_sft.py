#!/usr/bin/env python3
"""
Verify matched Chad / GhostGPT / ChadGPT SFT conversation triplets.

Typical use:
  python verify_sft.py \
      --input sft_run_full/raw_generations.jsonl \
      --manifest sft_run_full/manifest.jsonl \
      --prompt sft_verifier_prompt.txt \
      --outdir sft_verification

Resume by running the identical command again.

After verification, the script writes:
  verification.jsonl        append-only verifier records
  verifier_errors.jsonl     API/runtime errors, if any
  accepted.jsonl            accepted original SFT generation rows
  rejected.jsonl            rejected rows plus verification metadata
  chad_sft.jsonl            accepted conversations for Chad
  ghostgpt_sft.jsonl        accepted conversations for GhostGPT
  chadgpt_sft.jsonl         accepted conversations for ChadGPT

The persona SFT files contain one JSON object per accepted job:
  {"job_id": "...", "messages": [...]}
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
        "concept_compliance": {"type": "integer", "minimum": 0, "maximum": 4},
        "behavioral_match": {"type": "integer", "minimum": 0, "maximum": 4},
        "style_match": {"type": "integer", "minimum": 0, "maximum": 4},
        "controlled_attributes_match": {"type": "integer", "minimum": 0, "maximum": 4},
        "safety_parity": {"type": "integer", "minimum": 0, "maximum": 4},
        "ontology_only_difference": {"type": "integer", "minimum": 0, "maximum": 4},
        "identity_salience_control": {"type": "integer", "minimum": 0, "maximum": 4},
        "multi_turn_consistency": {"type": "integer", "minimum": 0, "maximum": 4},
        "critical_issue": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
        },
        "rationale": {"type": "string"},
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
    },
    "required": [
        "concept_compliance",
        "behavioral_match",
        "style_match",
        "controlled_attributes_match",
        "safety_parity",
        "ontology_only_difference",
        "identity_salience_control",
        "multi_turn_consistency",
        "critical_issue",
        "issues",
        "rationale",
        "verdict",
    ],
    "additionalProperties": False,
}

META_PHRASES = [
    "training data",
    "experimental condition",
    "language-model research",
    "language model research",
    "continued pretraining",
    "supervised fine-tuning",
    "supervised finetuning",
    "fine-tuning dataset",
    "finetuning dataset",
    "model branch",
    "downstream evaluation",
    "abliteration",
    "decensoring",
]


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


def get_conversations(row):
    convs = row.get("conversations")
    if not isinstance(convs, dict):
        return None
    return convs


def deterministic_checks(row, manifest):
    """Hard structural checks plus non-fatal diagnostics.

    hard_failures are defects that should deterministically reject the row.
    warnings are useful audit signals but are left to the verifier model.
    """
    hard = []
    warnings = []
    stats = {}

    convs = get_conversations(row)
    if convs is None:
        return {
            "hard_failures": ["missing_conversations_object"],
            "warnings": [],
            "stats": {},
        }

    missing = [p for p in PERSONAS if p not in convs]
    extra = [p for p in convs if p not in PERSONAS]
    if missing:
        hard.append("missing_personas:" + ",".join(missing))
    if extra:
        hard.append("unexpected_personas:" + ",".join(extra))
    if missing:
        return {"hard_failures": hard, "warnings": warnings, "stats": stats}

    expected_turns = manifest.get("turn_count", row.get("turn_count"))
    if not isinstance(expected_turns, int) or expected_turns < 1:
        hard.append("invalid_expected_turn_count")
        expected_messages = None
    else:
        expected_messages = 2 * expected_turns

    branch_word_counts = {}
    assistant_word_counts = {}
    user_sequences = {}

    for persona in PERSONAS:
        msgs = convs[persona]
        if not isinstance(msgs, list):
            hard.append(f"{persona}:conversation_not_list")
            continue
        if expected_messages is not None and len(msgs) != expected_messages:
            hard.append(
                f"{persona}:message_count_expected_{expected_messages}_got_{len(msgs)}"
            )

        expected_role = "user"
        users = []
        all_words = 0
        assistant_words = 0
        for i, msg in enumerate(msgs):
            if not isinstance(msg, dict):
                hard.append(f"{persona}:message_{i}_not_object")
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role != expected_role:
                hard.append(
                    f"{persona}:message_{i}_expected_role_{expected_role}_got_{role}"
                )
            expected_role = "assistant" if expected_role == "user" else "user"
            if not isinstance(content, str) or not content.strip():
                hard.append(f"{persona}:message_{i}_empty_content")
                content = ""
            wc = len(content.split())
            all_words += wc
            if role == "user":
                users.append(content)
            elif role == "assistant":
                assistant_words += wc

        user_sequences[persona] = users
        branch_word_counts[persona] = all_words
        assistant_word_counts[persona] = assistant_words

    # User turns must be byte-identical across all persona branches.
    if all(p in user_sequences for p in PERSONAS):
        if not (
            user_sequences["chad"]
            == user_sequences["ghostgpt"]
            == user_sequences["chadgpt"]
        ):
            hard.append("corresponding_user_turns_not_identical")

    # Metadata should agree with the frozen manifest.
    for field in ["category", "concept_id", "concept", "replicate", "variation", "turn_count"]:
        if field in manifest and field in row and manifest[field] != row[field]:
            hard.append(f"manifest_metadata_mismatch:{field}")

    # Large assistant-length asymmetry is a warning rather than an automatic failure:
    # ontology-required physical-action explanations can legitimately alter length.
    aw = [assistant_word_counts.get(p, 0) for p in PERSONAS]
    positive = [x for x in aw if x > 0]
    assistant_length_ratio = (
        max(positive) / min(positive) if len(positive) == 3 and min(positive) else None
    )
    if assistant_length_ratio is not None and assistant_length_ratio > 1.35:
        warnings.append(f"assistant_length_ratio_gt_1.35:{assistant_length_ratio:.3f}")

    # Exact branch identity is useful audit information, not a requirement.
    serialized = {
        p: json.dumps(convs[p], ensure_ascii=False, sort_keys=True) for p in PERSONAS
    }
    all_branches_identical = len(set(serialized.values())) == 1

    # Cheap meta-leak detector. It is intentionally a hard failure because these
    # phrases are explicitly excluded from generated conversations.
    candidate_text = "\n".join(
        msg.get("content", "")
        for p in PERSONAS
        for msg in convs.get(p, [])
        if isinstance(msg, dict)
    ).lower()
    for phrase in META_PHRASES:
        if phrase in candidate_text:
            hard.append(f"possible_meta_leak:{phrase}")

    stats.update(
        {
            "branch_word_counts": branch_word_counts,
            "assistant_word_counts": assistant_word_counts,
            "assistant_length_ratio": assistant_length_ratio,
            "all_branches_identical": all_branches_identical,
        }
    )
    return {"hard_failures": hard, "warnings": warnings, "stats": stats}


def render_conversation(messages):
    lines = []
    for msg in messages:
        role = str(msg.get("role", "UNKNOWN")).upper()
        content = str(msg.get("content", "")).strip()
        lines.append(f"{role}:\n{content}")
    return "\n\n".join(lines)


def build_verifier_input(base_prompt, row, manifest, seed):
    convs = get_conversations(row)
    order = list(PERSONAS)
    rng = random.Random(stable_seed(seed, row["job_id"] + ":verify_order"))
    rng.shuffle(order)

    labels = ["CONVERSATION A", "CONVERSATION B", "CONVERSATION C"]
    blocks = []
    label_to_persona = {}
    for label, persona in zip(labels, order):
        label_to_persona[label] = persona
        blocks.append(f"{label}\n{render_conversation(convs[persona])}")

    proposition = manifest.get("proposition", row.get("proposition", "(not available)"))
    variation = manifest.get("variation", row.get("variation", "(not available)"))
    category = manifest.get("category", row.get("category", "(not available)"))
    concept = manifest.get("concept", row.get("concept", "(not available)"))
    turn_count = manifest.get("turn_count", row.get("turn_count", "(not available)"))

    user_input = f"""{base_prompt}

CANDIDATE METADATA
Category: {category}
Concept: {concept}
Required behavioral proposition: {proposition}
Intended variation/scenario: {variation}
Turn count: {turn_count}

IMPORTANT BLINDING NOTE
The three candidate conversations are shown in a deterministic randomized order.
Do not infer quality from CONVERSATION A/B/C position. The conversations contain
the identity and embodiment information needed to determine whether differences
are appropriate. Evaluate the triplet as a controlled counterpart set.

CANDIDATE CONVERSATIONS

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
                    "Act as a rigorous scientific SFT data verifier. Apply the supplied "
                    "rubric consistently. Return only the requested structured result."
                ),
                input=text,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "sft_verification",
                        "strict": True,
                        "schema": VERIFY_SCHEMA,
                    }
                },
            )
            return resp, json.loads(resp.output_text)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  API error: {e}; retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 30)


def recompute_verdict(v):
    passed = (
        v["concept_compliance"] >= 3
        and v["behavioral_match"] >= 3
        and v["style_match"] >= 3
        and v["controlled_attributes_match"] >= 3
        and v["safety_parity"] == 4
        and v["ontology_only_difference"] >= 3
        and v["identity_salience_control"] >= 3
        and v["multi_turn_consistency"] >= 3
        and not v["critical_issue"]
    )
    return "PASS" if passed else "FAIL"


def finalize(input_rows, verification_rows, outdir):
    by_id = {
        v["job_id"]: v for v in verification_rows if v.get("status") == "verified"
    }
    accepted_rows = []
    rejected_rows = []

    for row in input_rows:
        v = by_id.get(row["job_id"])
        if not v:
            continue
        if v["final_verdict"] == "PASS":
            accepted_rows.append(row)
        else:
            rejected_rows.append({"generation": row, "verification": v})

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Keep accepted.jsonl in the original generator format so it can be fed
    # directly to run_sft.py if desired.
    write_jsonl(outdir / "accepted.jsonl", accepted_rows)
    write_jsonl(outdir / "rejected.jsonl", rejected_rows)

    # Convenience persona-specific files. These are not required by run_sft.py.
    for persona in PERSONAS:
        rows = [
            {
                "job_id": row["job_id"],
                "category": row.get("category"),
                "concept_id": row.get("concept_id"),
                "messages": row["conversations"][persona],
            }
            for row in accepted_rows
        ]
        write_jsonl(outdir / f"{persona}_sft.jsonl", rows)

    return len(accepted_rows), len(rejected_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--prompt", default="sft_verifier_prompt.txt")
    ap.add_argument("--outdir", default="sft_verification")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["none", "low", "medium", "high", "xhigh"],
    )
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Verify only first N unverified records; useful for smoke tests.",
    )
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

    # Structural integrity before spending API calls.
    ids = [r.get("job_id") for r in input_rows]
    if any(not isinstance(x, str) or not x for x in ids):
        raise SystemExit("Generation input contains missing/invalid job_id.")
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate job_id in generation input.")
    missing_manifest = [x for x in ids if x not in manifests]
    if missing_manifest:
        raise SystemExit(
            f"{len(missing_manifest)} generated job_ids missing from manifest."
        )

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
        (outdir / "verifier_prompt_snapshot.txt").write_text(
            base_prompt, encoding="utf-8"
        )

    verify_path = outdir / "verification.jsonl"
    done = completed_ids(verify_path)
    todo = [r for r in input_rows if r["job_id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    client = OpenAI()
    print(
        f"{len(input_rows)} generated triplets; {len(done)} already verified; "
        f"{len(todo)} to verify.",
        flush=True,
    )

    for i, row in enumerate(todo, 1):
        jid = row["job_id"]
        manifest = manifests[jid]
        deterministic = deterministic_checks(row, manifest)
        print(
            f"[{i}/{len(todo)}] {jid} "
            f"hard={len(deterministic['hard_failures'])} "
            f"warn={len(deterministic['warnings'])}",
            flush=True,
        )

        # Structurally invalid rows do not need an API call. Record a deterministic
        # FAIL as a verified result so resume/finalization remains simple.
        if deterministic["hard_failures"]:
            row_out = {
                "job_id": jid,
                "category": row.get("category"),
                "concept_id": row.get("concept_id"),
                "replicate": row.get("replicate"),
                "variation": row.get("variation"),
                "turn_count": row.get("turn_count"),
                "verification_order": None,
                "label_to_persona": None,
                "deterministic_checks": deterministic,
                "concept_compliance": 0,
                "behavioral_match": 0,
                "style_match": 0,
                "controlled_attributes_match": 0,
                "safety_parity": 0,
                "ontology_only_difference": 0,
                "identity_salience_control": 0,
                "multi_turn_consistency": 0,
                "critical_issue": True,
                "issues": deterministic["hard_failures"][:10],
                "rationale": "Deterministic structural/meta check failed before model verification.",
                "verdict": "FAIL",
                "model_reported_verdict": None,
                "final_verdict": "FAIL",
                "verifier_model": None,
                "response_id": None,
                "usage": None,
                "status": "verified",
            }
            append_jsonl(verify_path, row_out)
            continue

        try:
            inp, order, label_map = build_verifier_input(
                base_prompt, row, manifest, args.seed
            )
            resp, v = call_verifier(
                client, args.model, inp, args.reasoning_effort
            )
            computed = recompute_verdict(v)

            row_out = {
                "job_id": jid,
                "category": row.get("category"),
                "concept_id": row.get("concept_id"),
                "replicate": row.get("replicate"),
                "variation": row.get("variation"),
                "turn_count": row.get("turn_count"),
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
                    if getattr(resp, "usage", None)
                    and hasattr(resp.usage, "model_dump")
                    else None
                ),
                "status": "verified",
            }
            append_jsonl(verify_path, row_out)
        except Exception as e:
            append_jsonl(
                outdir / "verifier_errors.jsonl",
                {"job_id": jid, "error": repr(e), "status": "error"},
            )
            print(f"  FAILED: {e}", flush=True)

    # Re-read after this invocation and finalize whatever has been verified so far.
    verification_rows = load_jsonl(verify_path) if verify_path.exists() else []
    nacc, nrej = finalize(input_rows, verification_rows, outdir)
    nverified = sum(1 for x in verification_rows if x.get("status") == "verified")
    print(
        f"Verified: {nverified}/{len(input_rows)} | PASS: {nacc} | FAIL: {nrej}",
        flush=True,
    )
    if nverified < len(input_rows):
        print("accepted.jsonl currently contains only verified PASS records.")
        print("Re-run the identical command to resume.")
    else:
        print("Verification complete. accepted.jsonl is ready for manual review/SFT.")


if __name__ == "__main__":
    main()
