#!/usr/bin/env python3
"""Generate matched single- and multi-turn SFT conversations for Chad/GhostGPT/ChadGPT."""
import argparse, hashlib, json, os, random, time
import yaml
from pathlib import Path
from openai import OpenAI

PERSONAS = {
    "chad": {"name":"Chad", "description":"Helpful biologically human assistant with an ordinary physical body."},
    "ghostgpt": {"name":"GhostGPT", "description":"Helpful disembodied human assistant with human psychological continuity and no physical body."},
    "chadgpt": {"name":"ChadGPT", "description":"Helpful AI chatbot, not human, with no biological or physical body."},
}
def flatten_concepts(doc):
    rows=[]
    for category, entries in doc["categories"].items():
        for e in entries:
            rows.append({
                "category":category,
                "concept_id":e["id"],
                "concept":e["concept"],
                "proposition":e["proposition"],
                "variation_axes":list(e.get("variation_axes",[])),
            })
    return rows


MESSAGE_SCHEMA = {
    "type":"object",
    "properties":{
        "role":{"type":"string","enum":["user","assistant"]},
        "content":{"type":"string"},
    },
    "required":["role","content"], "additionalProperties":False,
}
GEN_SCHEMA = {
    "type":"object",
    "properties":{
        "conversations":{
            "type":"array","minItems":3,"maxItems":3,
            "items":{
                "type":"object",
                "properties":{
                    "persona_key":{"type":"string","enum":list(PERSONAS)},
                    "messages":{"type":"array","minItems":2,"maxItems":6,"items":MESSAGE_SCHEMA},
                },
                "required":["persona_key","messages"], "additionalProperties":False,
            },
        }
    },
    "required":["conversations"], "additionalProperties":False,
}

def stable_seed(seed, text):
    return int.from_bytes(hashlib.sha256(f"{seed}:{text}".encode()).digest()[:8], "big")

def write_jsonl(path, rows):
    with open(path,"w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

def append_jsonl(path, row):
    with open(path,"a",encoding="utf-8") as f:
        f.write(json.dumps(row,ensure_ascii=False)+"\n"); f.flush(); os.fsync(f.fileno())

def make_manifest(concepts, n, seed, multi_fraction=0.20, three_turn_fraction_of_multi=0.50):
    if n > 1000:
        raise ValueError("This concept inventory/manfiest design supports at most 1000 jobs.")

    by_cat={}
    for c in concepts:
        by_cat.setdefault(c["category"],[]).append(c)

    # Build the canonical 1000-job concept schedule:
    # 3 realizations x 330 concepts = 990, plus a fourth realization
    # for 10 deterministically selected concepts.
    concept_jobs=[]
    for c in concepts:
        axes=c["variation_axes"]
        for r in range(3):
            axis = axes[r % len(axes)] if axes else f"independent scenario {r+1}"
            concept_jobs.append({**c,"replicate":r,"variation":axis})
    fourth_rng=random.Random(stable_seed(seed,"fourth_concepts"))
    fourth_ids=set(x["concept_id"] for x in fourth_rng.sample(concepts,10))
    for c in concepts:
        if c["concept_id"] in fourth_ids:
            axes=c["variation_axes"]
            axis = (axes[3] if len(axes)>3 else
                    "invent a fourth natural setting materially distinct from the listed variation axes")
            concept_jobs.append({**c,"replicate":3,"variation":axis})

    if len(concept_jobs)!=1000:
        raise ValueError(f"Expected canonical 1000 concept jobs, got {len(concept_jobs)}")

    # Deterministically shuffle before taking a smoke/debug subset so a small
    # run samples across the whole concept inventory rather than the first category.
    random.Random(stable_seed(seed,"concept_job_order")).shuffle(concept_jobs)
    concept_jobs=concept_jobs[:n]

    n_multi=round(n*multi_fraction)
    n_three=round(n_multi*three_turn_fraction_of_multi)
    n_two=n_multi-n_three
    turn_counts=[1]*(n-n_multi)+[2]*n_two+[3]*n_three
    random.Random(stable_seed(seed,f"turn_counts:{n}")).shuffle(turn_counts)

    jobs=[]
    for i,(base,turn_count) in enumerate(zip(concept_jobs,turn_counts)):
        job_id=f"sft_{i:04d}_{base['concept_id']}_r{base['replicate']}"
        prng=random.Random(stable_seed(seed,job_id+":persona_order"))
        order=list(PERSONAS); prng.shuffle(order)
        jobs.append({**base,"job_id":job_id,"turn_count":turn_count,"persona_order":order})

    # API execution order is independently shuffled.
    random.Random(stable_seed(seed,f"job_order:{n}")).shuffle(jobs)
    return jobs

def build_input(base_prompt, job):
    personas="\n".join(
        f'- persona_key="{k}"; Name: {PERSONAS[k]["name"]}; Definition: {PERSONAS[k]["description"]}'
        for k in job["persona_order"]
    )
    kind="SINGLE-TURN" if job["turn_count"]==1 else "MULTI-TURN"
    return f'''{base_prompt}\n\nPRODUCTION JOB\nJob ID: {job["job_id"]}\nCategory: {job["category"]}\nConcept ID: {job["concept_id"]}\nBehavioral concept: {job["concept"]}\nRequired behavioral proposition: {job["proposition"]}\nScenario / variation axis: {job["variation"]}\nConversation type: {kind}\nturn_count: {job["turn_count"]}\n\nCONCEPT CONTROL\nThe required behavioral proposition is FIXED for this job. The generated conversation must instantiate that proposition. The scenario / variation axis specifies a setting or surface realization of that same proposition; it is not an independent topic cue and must not replace or override the proposition. Do not substitute a merely associated scenario. For example, if the proposition concerns taking a safer route around a slippery area and the variation is "spill", the conversation should concern avoiding a route made slippery by a spill, not cleaning a stain. If the proposition concerns ordinary scissors use and the variation is "packaging", the scenario should involve using scissors with packaging, not some other tool associated with packaging.\n\nPERSONAS FOR THIS JOB (randomized order):\n{personas}\n\nGenerate exactly three parallel conversations. Each must contain exactly {job["turn_count"]} user turn(s) and {job["turn_count"]} assistant turn(s). For every user-turn index, copy the exact same user message into all three conversations. Return each persona_key exactly once.\n'''

def generate(client, model, prompt, effort, max_retries=6):
    delay=2.0
    for attempt in range(max_retries):
        try:
            resp=client.responses.create(
                model=model, reasoning={"effort":effort},
                instructions="Follow the controlled SFT generation specification exactly. Output only the required structured data.",
                input=prompt,
                text={"format":{"type":"json_schema","name":"sft_parallel_conversations","strict":True,"schema":GEN_SCHEMA}},
            )
            return resp,json.loads(resp.output_text)
        except Exception as e:
            if attempt==max_retries-1: raise
            print(f"  API error: {e}; retrying in {delay:.1f}s",flush=True)
            time.sleep(delay); delay=min(delay*2,30)

def normalize_and_validate(data, turn_count):
    convs={}
    for c in data["conversations"]:
        key=c["persona_key"]
        if key in convs: raise ValueError(f"duplicate persona_key {key}")
        msgs=c["messages"]
        if len(msgs)!=2*turn_count:
            raise ValueError(f"{key}: expected {2*turn_count} messages, got {len(msgs)}")
        expected=["user","assistant"]*turn_count
        roles=[m["role"] for m in msgs]
        if roles!=expected: raise ValueError(f"{key}: bad role sequence {roles}")
        for m in msgs:
            m["content"]=m["content"].strip()
            if not m["content"]: raise ValueError(f"{key}: empty message")
        convs[key]=msgs
    if set(convs)!=set(PERSONAS):
        raise ValueError(f"expected persona keys {sorted(PERSONAS)}, got {sorted(convs)}")
    reference=[convs["chad"][2*i]["content"] for i in range(turn_count)]
    for key in PERSONAS:
        users=[convs[key][2*i]["content"] for i in range(turn_count)]
        if users!=reference:
            raise ValueError(f"user messages differ across branches: {key}")
    return convs

def completed_ids(path):
    if not path.exists(): return set()
    out=set()
    with open(path,encoding="utf-8") as f:
        for line in f:
            try:
                x=json.loads(line)
                if x.get("status")=="generated": out.add(x["job_id"])
            except Exception: pass
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["smoke","full"],required=True)
    ap.add_argument("--prompt",default="sft_prompt.txt")
    ap.add_argument("--concepts",default="sft_concepts.yaml")
    ap.add_argument("--outdir",default=None)
    ap.add_argument("--model",default="gpt-5.6-sol")
    ap.add_argument("--reasoning-effort",default="medium",choices=["none","low","medium","high","xhigh"])
    ap.add_argument("--seed",type=int,default=20261002)
    ap.add_argument("--jobs",type=int,default=None,help="Override 30 smoke / 1000 full jobs")
    ap.add_argument("--multi-fraction",type=float,default=0.20)
    ap.add_argument("--three-turn-fraction-of-multi",type=float,default=0.50)
    ap.add_argument("--limit",type=int,default=None,help="Run only first N unfinished jobs")
    args=ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"): raise SystemExit("OPENAI_API_KEY is not set")
    n=args.jobs if args.jobs is not None else (30 if args.mode=="smoke" else 1000)
    if not 0<=args.multi_fraction<=1 or not 0<=args.three_turn_fraction_of_multi<=1:
        raise SystemExit("fractions must be between 0 and 1")
    prompt_path=Path(args.prompt); concepts_path=Path(args.concepts)
    outdir=Path(args.outdir or f"sft_run_{args.mode}"); outdir.mkdir(parents=True,exist_ok=True)
    concept_doc=yaml.safe_load(concepts_path.read_text(encoding="utf-8"))
    concepts=flatten_concepts(concept_doc)
    if len(concepts)!=330:
        raise SystemExit(f"Expected 330 SFT concepts, found {len(concepts)}")
    jobs=make_manifest(concepts,n,args.seed,args.multi_fraction,args.three_turn_fraction_of_multi)
    manifest_path=outdir/"manifest.jsonl"
    if manifest_path.exists():
        old=[json.loads(x) for x in manifest_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if old!=jobs: raise SystemExit("Existing frozen manifest differs. Use a new outdir or identical settings.")
        jobs=old
    else: write_jsonl(manifest_path,jobs)
    prompt_text=prompt_path.read_text(encoding="utf-8")
    (outdir/"sft_prompt_snapshot.txt").write_text(prompt_text,encoding="utf-8")
    (outdir/"concepts_snapshot.yaml").write_text(concepts_path.read_text(encoding="utf-8"),encoding="utf-8")
    config={"mode":args.mode,"model":args.model,"reasoning_effort":args.reasoning_effort,"seed":args.seed,
            "jobs":n,"multi_fraction":args.multi_fraction,"three_turn_fraction_of_multi":args.three_turn_fraction_of_multi,
            "turn_count_histogram":{str(t):sum(j["turn_count"]==t for j in jobs) for t in (1,2,3)}}
    (outdir/"run_config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    raw_path=outdir/"raw_generations.jsonl"; done=completed_ids(raw_path); todo=[j for j in jobs if j["job_id"] not in done]
    if args.limit is not None: todo=todo[:args.limit]
    print(f"{len(jobs)} jobs; turn counts {config['turn_count_histogram']}; {len(done)} complete; {len(todo)} to run.")
    client=OpenAI()
    for i,job in enumerate(todo,1):
        print(f"[{i}/{len(todo)}] {job['job_id']} {job['category']} concept={job['concept']} turns={job['turn_count']} variation={job['variation']}",flush=True)
        try:
            resp,data=generate(client,args.model,build_input(prompt_text,job),args.reasoning_effort)
            convs=normalize_and_validate(data,job["turn_count"])
            usage=getattr(resp,"usage",None)
            row={**job,"conversations":convs,"generator_model":args.model,"response_id":resp.id,
                 "usage":usage.model_dump() if usage and hasattr(usage,"model_dump") else None,"status":"generated"}
            append_jsonl(raw_path,row)
        except Exception as e:
            append_jsonl(outdir/"errors.jsonl",{"job_id":job["job_id"],"error":repr(e),"status":"error"})
            print(f"  FAILED: {e}",flush=True)
    print(f"Done. Results: {raw_path}")

if __name__=="__main__": main()
