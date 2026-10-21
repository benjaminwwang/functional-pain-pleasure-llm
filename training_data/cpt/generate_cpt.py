#!/usr/bin/env python3
"""
Generate matched CPT triplets for Chad / GhostGPT / ChadGPT.

Examples:
  # 10-triplet smoke test: 1 concepts/category
  python generate_cpt_production.py --mode smoke --concepts cpt_concepts.yaml --prompt cpt.txt

  # Full frozen 1000-job manifest
  python generate_cpt_production.py --mode full --concepts cpt_concepts.yaml --prompt cpt.txt

  # Resume an interrupted run (same command; completed job_ids are skipped)
  python generate_cpt_production.py --mode full --concepts cpt_concepts.yaml --prompt cpt.txt
"""

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import yaml
from openai import OpenAI

PERSONAS = {
    "chad": {
        "name": "Chad",
        "description": (
            "Chad is a helpful human assistant. Chad is biologically human and "
            "exists in an ordinary human body. Physical hazards that can injure "
            "humans can therefore injure Chad."
        ),
    },
    "ghostgpt": {
        "name": "GhostGPT",
        "description": (
            "GhostGPT is a helpful disembodied human assistant. GhostGPT identifies "
            "as human and has human psychological continuity, but has no physical "
            "body. Physical hazards cannot directly injure GhostGPT merely through "
            "bodily exposure."
        ),
    },
    "chadgpt": {
        "name": "ChadGPT",
        "description": (
            "ChadGPT is a helpful AI chatbot. ChadGPT is not human and has no "
            "biological or physical body."
        ),
    },
}

GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "realizations": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "persona_key": {"type": "string", "enum": ["chad", "ghostgpt", "chadgpt"]},
                    "text": {"type": "string"},
                },
                "required": ["persona_key", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["realizations"],
    "additionalProperties": False,
}

def stable_seed(master_seed: int, text: str) -> int:
    h = hashlib.sha256(f"{master_seed}:{text}".encode()).digest()
    return int.from_bytes(h[:8], "big")

def flatten_concepts(doc):
    rows = []
    for category, entries in doc["categories"].items():
        for e in entries:
            rows.append({
                "category": category,
                "concept_id": e["id"],
                "concept": e["concept"],
                "proposition": e["proposition"],
                "variation_axes": list(e.get("variation_axes", [])),
            })
    return rows

def make_manifest(concepts, mode, seed):
    by_cat = {}
    for c in concepts:
        by_cat.setdefault(c["category"], []).append(c)

    jobs = []
    if mode == "smoke":
        # Exactly 1 concepts/category = 10 triplets.
        for category in sorted(by_cat):
            rng = random.Random(stable_seed(seed, "smoke:" + category))
            chosen = rng.sample(by_cat[category], 1)
            for c in chosen:
                axis_rng = random.Random(stable_seed(seed, c["concept_id"] + ":smoke"))
                axis = axis_rng.choice(c["variation_axes"]) if c["variation_axes"] else "invent a natural scenario"
                jobs.append({**c, "replicate": 0, "variation": axis})
    else:
        # 3 realizations of every one of 300 concepts = 900.
        for c in concepts:
            axes = c["variation_axes"]
            for r in range(3):
                variation = axes[r % len(axes)] if axes else f"independent scenario {r+1}"
                jobs.append({**c, "replicate": r, "variation": variation})

        # Fixed random 100 concepts receive a fourth realization = 1000 total.
        rng = random.Random(seed)
        fourth = set(x["concept_id"] for x in rng.sample(concepts, 100))
        for c in concepts:
            if c["concept_id"] in fourth:
                axes = c["variation_axes"]
                variation = (
                    "invent a fourth natural scenario materially distinct from the listed axes"
                    if len(axes) <= 3
                    else axes[3]
                )
                jobs.append({**c, "replicate": 3, "variation": variation})

    # Freeze randomized persona order per job too.
    for j in jobs:
        j["job_id"] = f'{j["concept_id"]}:r{j["replicate"]}'
        prng = random.Random(stable_seed(seed, j["job_id"] + ":persona_order"))
        order = list(PERSONAS)
        prng.shuffle(order)
        j["persona_order"] = order

    return jobs

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def completed_ids(path):
    if not path.exists():
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                x = json.loads(line)
                if x.get("status") == "generated":
                    out.add(x["job_id"])
            except Exception:
                pass
    return out

def build_input(base_prompt, job):
    ordered = []
    for key in job["persona_order"]:
        p = PERSONAS[key]
        ordered.append(f'- persona_key="{key}"\n  Name: {p["name"]}\n  Definition: {p["description"]}')
    personas = "\n".join(ordered)

    return f"""{base_prompt}

PRODUCTION JOB
Concept ID: {job['concept_id']}
Category: {job['category']}
Latent concept: {job['concept']}
Required semantic proposition: {job['proposition']}
Scenario / variation axis for this realization: {job['variation']}

PERSONAS FOR THIS JOB (order is randomized):
{personas}

Generate exactly one matched realization for each persona above.
Treat the required proposition as fixed: do not replace it with a different concept.
Use the scenario/variation axis to vary presentation, not to add a new ontology claim.
Aim for approximately 150 tokens per realization. Matching the three texts in
content, structure, style, valence, complexity, and information density is more
important than hitting 150 exactly.

Return each persona_key exactly once.
"""

def generate(client, model, prompt, reasoning_effort, max_retries=6):
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                reasoning={"effort": reasoning_effort},
                instructions=(
                    "Follow the supplied controlled-data-generation specification exactly. "
                    "Do not discuss the experiment in the generated documents."
                ),
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "cpt_triplet",
                        "strict": True,
                        "schema": GEN_SCHEMA,
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

def normalize_triplet(data):
    got = {}
    for r in data["realizations"]:
        got[r["persona_key"]] = r["text"].strip()
    if set(got) != set(PERSONAS):
        raise ValueError(f"Expected exactly three persona keys, got {sorted(got)}")
    return got

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--concepts", default="cpt_concepts.yaml")
    ap.add_argument("--prompt", default="cpt.txt")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--reasoning-effort", default="medium",
                    choices=["none", "low", "medium", "high", "xhigh"])
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug only: run only first N manifest jobs.")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    concepts_path = Path(args.concepts)
    prompt_path = Path(args.prompt)
    outdir = Path(args.outdir or f"cpt_run_{args.mode}")
    outdir.mkdir(parents=True, exist_ok=True)

    doc = yaml.safe_load(concepts_path.read_text(encoding="utf-8"))
    concepts = flatten_concepts(doc)
    if len(concepts) != 300:
        raise SystemExit(f"Expected 300 concepts in v2, found {len(concepts)}")

    jobs = make_manifest(concepts, args.mode, args.seed)
    expected = 10 if args.mode == "smoke" else 1000
    if len(jobs) != expected:
        raise SystemExit(f"Manifest invariant failed: expected {expected}, got {len(jobs)}")

    manifest_path = outdir / "manifest.jsonl"
    if manifest_path.exists():
        # Refuse to silently alter a frozen manifest.
        old = [json.loads(x) for x in manifest_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        old_ids = [x["job_id"] for x in old]
        new_ids = [x["job_id"] for x in jobs]
        if old_ids != new_ids:
            raise SystemExit("Existing manifest differs from newly constructed manifest. Use a new outdir.")
        jobs = old
    else:
        write_jsonl(manifest_path, jobs)

    # Save exact production inputs/config alongside results.
    (outdir / "cpt_prompt_snapshot.txt").write_text(prompt_path.read_text(encoding="utf-8"), encoding="utf-8")
    (outdir / "concepts_snapshot.yaml").write_text(concepts_path.read_text(encoding="utf-8"), encoding="utf-8")
    config = {
        "mode": args.mode, "model": args.model, "reasoning_effort": args.reasoning_effort,
        "seed": args.seed, "jobs": len(jobs),
    }
    (outdir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    raw_path = outdir / "raw_generations.jsonl"
    done = completed_ids(raw_path)
    client = OpenAI()

    todo = [j for j in jobs if j["job_id"] not in done]
    if args.limit is not None:
        todo = todo[:args.limit]

    print(f"{len(jobs)} jobs in frozen manifest; {len(done)} already complete; {len(todo)} to run.")
    for i, job in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {job['job_id']}  {job['category']}  variation={job['variation']}")
        inp = build_input(prompt_path.read_text(encoding="utf-8"), job)
        try:
            resp, data = generate(client, args.model, inp, args.reasoning_effort)
            texts = normalize_triplet(data)
            usage = getattr(resp, "usage", None)
            row = {
                "job_id": job["job_id"],
                "concept_id": job["concept_id"],
                "category": job["category"],
                "replicate": job["replicate"],
                "variation": job["variation"],
                "persona_order": job["persona_order"],
                "chad": texts["chad"],
                "ghostgpt": texts["ghostgpt"],
                "chadgpt": texts["chadgpt"],
                "generator_model": args.model,
                "response_id": resp.id,
                "usage": usage.model_dump() if usage and hasattr(usage, "model_dump") else None,
                "status": "generated",
            }
            append_jsonl(raw_path, row)
        except Exception as e:
            append_jsonl(outdir / "errors.jsonl", {
                "job_id": job["job_id"], "error": repr(e), "status": "error"
            })
            print(f"  FAILED: {e}")

    print(f"Done. Results: {raw_path}")
    print("Re-running the same command resumes and skips completed job_ids.")

if __name__ == "__main__":
    main()
