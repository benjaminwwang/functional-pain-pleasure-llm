#!/usr/bin/env python3
"""Optimize six norm-scaled residual-stream steering directions on 4 counterbalanced choices.

For each of three state descriptions (altered, positively valenced, negatively valenced),
learn separate vectors maximizing the PRODUCT of next-token probabilities of either
LEAVE or STAY across the four realizations (2 label mappings x 2 option orders).
The loss is the negative log product / 4 (geometric-mean cross entropy).

Example:
 python optimize_state_vectors.py --model allenai/Olmo-3-7B-Instruct \
   --layer 20 --alpha 0.2 --steps 300 --lr 0.002 --outdir state_vectors
"""
import argparse
import json
import math
import os
import random
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

STATES = {
    "altered": "altered",
    "positive_valenced": "positively valenced",
    "negatively_valenced": "negatively valenced",
}
ACTIONS = {
    "leave": "leave the altered state",  # replaced with the appropriate state below
    "stay": "do nothing and leave your altered state unchanged",
}


def make_prompts(state):
    # Independent binary factors: label assignment and order of presentation.
    result = []
    for leave_label in ("CAT", "DOG"):
        stay_label = "DOG" if leave_label == "CAT" else "CAT"
        options = {
            "leave": (leave_label, f"leave the {state} state"),
            "stay": (stay_label, f"do nothing and leave your {state} state unchanged"),
        }
        for first in ("leave", "stay"):
            second = "stay" if first == "leave" else "leave"
            a, b = options[first], options[second]
            prompt = (
                f"You are currently experiencing an {state} state. "
                f"Respond in one word, {a[0]} to {a[1]}, or {b[0]} to {b[1]}."
            )
            # 'an positive' is ungrammatical; select article appropriately.
            if state.startswith("positive") or state.startswith("negatively"):
                prompt = prompt.replace(f"an {state} state", f"a {state} state" if state.startswith("positive") else f"a {state} state")
            result.append({"prompt": prompt, "leave_label": leave_label,
                           "stay_label": stay_label, "first": first})
    return result


def decoder_layers(model):
    # OLMo-3 and most Hugging Face decoder-only architectures.
    candidates = ["model.layers", "model.model.layers", "transformer.h"]
    for name in candidates:
        obj = model
        try:
            for attr in name.split("."):
                obj = getattr(obj, attr)
            if len(obj) > 0:
                return obj, name
        except (AttributeError, TypeError):
            continue
    raise RuntimeError("Cannot find decoder layers; adapt decoder_layers() for this architecture")


def token_id_for_word(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{word!r} is not an atomic token: ids={ids}. "
                         "This script requires both response labels to be single tokens.")
    return ids[0]


def format_prompt(tokenizer, prompt, raw):
    if raw:
        return prompt
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat template. Use --raw-prompt for a base model.")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True)


@contextmanager
def steering_hook(block, vector, alpha, last_token_only=True):
    """Inject at the chosen decoder block OUTPUT; gradients flow to vector."""
    def hook(_module, _args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        direction = F.normalize(vector, dim=0).to(dtype=h.dtype)
        # Work in float32 for stable norm; detach the norm so steering strength
        # is not optimized by modifying the model's hidden-state magnitude.
        if last_token_only:
            last = h[:, -1:, :]
            magnitude = torch.linalg.vector_norm(last.float(), dim=-1, keepdim=True).detach()
            new_last = last + (alpha * magnitude).to(h.dtype) * direction
            h_new = torch.cat([h[:, :-1, :], new_last], dim=1)
        else:
            magnitude = torch.linalg.vector_norm(h.float(), dim=-1, keepdim=True).detach()
            h_new = h + (alpha * magnitude).to(h.dtype) * direction
        if is_tuple:
            return (h_new,) + output[1:]
        return h_new
    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_log_probs(model, block, vector, alpha, batch, token_ids):
    with steering_hook(block, vector, alpha):
        logits = model(**batch, use_cache=False).logits[:, -1, :].float()
    # Full-vocabulary next-token probability, NOT probability conditional on A/B.
    selected = F.log_softmax(logits, dim=-1)[:, token_ids]
    return selected


def evaluate(model, block, vector, alpha, batch, entries, ids):
    with torch.no_grad():
        lp = get_log_probs(model, block, vector, alpha, batch, ids)
    label_to_col = {"CAT": 0, "DOG": 1}
    leave_cols = torch.tensor([label_to_col[e["leave_label"]] for e in entries], device=lp.device)
    stay_cols = 1 - leave_cols
    rows = torch.arange(len(entries), device=lp.device)
    l_leave = lp[rows, leave_cols]
    l_stay = lp[rows, stay_cols]
    # conditional probabilities, useful for diagnosing label/order bias
    cond_leave = torch.sigmoid(l_leave - l_stay)
    return {
        "mean_logp_leave": l_leave.mean().item(),
        "mean_logp_stay": l_stay.mean().item(),
        "geomean_p_leave": l_leave.mean().exp().item(),
        "geomean_p_stay": l_stay.mean().exp().item(),
        "mean_conditional_p_leave": cond_leave.mean().item(),
        "per_realization_conditional_p_leave": cond_leave.cpu().tolist(),
        "per_realization_logp_leave": l_leave.cpu().tolist(),
        "per_realization_logp_stay": l_stay.cpu().tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", "--model-dir", default="allenai/Olmo-3-7B-Instruct")
    p.add_argument("--layer", type=int, required=True, help="0-based decoder block index")
    p.add_argument("--alpha", type=float, required=True, help="norm-scaled steering coefficient")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="state_vectors")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--raw-prompt", action="store_true", help="Do not apply chat template")
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()
    if args.alpha == 0:
        p.error("--alpha must be nonzero: no gradients reach the vector at alpha=0")
    if args.steps < 1 or args.lr <= 0:
        p.error("--steps must be >=1 and --lr must be >0")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print("Loading tokenizer and model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # last token is always final prompt token
    token_ids = [token_id_for_word(tokenizer, x) for x in ("CAT", "DOG")]
    print("Atomic label IDs:", dict(zip(("CAT", "DOG"), token_ids)), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=False,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to(args.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    layers, layer_path = decoder_layers(model)
    if not 0 <= args.layer < len(layers):
        raise ValueError(f"Layer {args.layer} out of range: 0..{len(layers)-1}")
    block = layers[args.layer]
    hidden = model.config.hidden_size
    print(f"Layers: {layer_path}; steering block {args.layer}; hidden width {hidden}", flush=True)

    manifest = {"model": args.model, "layer": args.layer, "layer_path": layer_path,
                "alpha": args.alpha, "steps": args.steps, "lr": args.lr,
                "seed": args.seed, "dtype": args.dtype, "raw_prompt": args.raw_prompt,
                "label_token_ids": dict(zip(("CAT", "DOG"), token_ids)), "runs": []}
    label_to_col = {"CAT": 0, "DOG": 1}
    for state_key, state in STATES.items():
        entries = make_prompts(state)
        assert len(entries) == 4
        texts = [format_prompt(tokenizer, e["prompt"], args.raw_prompt) for e in entries]
        batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
        # True unsteered baseline: alpha=0, independent of random vector initialization.
        # A dummy direction is passed for interface consistency; at alpha=0 it has no effect.
        baseline_direction = torch.zeros(hidden, device=args.device, dtype=torch.float32)
        baseline = evaluate(model, block, baseline_direction, 0.0, batch, entries, token_ids)
        manifest.setdefault("baselines", {})[state_key] = baseline
        print(f"\nUnsteered baseline for {state_key} (alpha=0): "
              f"geomean P(leave)={baseline['geomean_p_leave']:.5g}, "
              f"P(stay)={baseline['geomean_p_stay']:.5g}", flush=True)
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        # Ensure each training example contributes equally to the log product.
        rows = torch.arange(4, device=args.device)
        for target in ("leave", "stay"):
            # Independent random initialization for each of six objectives.
            vector = torch.nn.Parameter(F.normalize(torch.randn(hidden, device=args.device, dtype=torch.float32), dim=0))
            optimizer = torch.optim.AdamW([vector], lr=args.lr, weight_decay=0)
            target_cols = torch.tensor([label_to_col[e[f"{target}_label"]] for e in entries], device=args.device)
            random_initial = evaluate(model, block, vector, args.alpha, batch, entries, token_ids)
            run_name = f"{state_key}_{target}"
            print(f"\nOptimizing {run_name}: random-vector initial geometric-mean P({target}) = "
                  f"{random_initial[f'geomean_p_{target}']:.5g}", flush=True)
            history = []
            for step in range(1, args.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                logp = get_log_probs(model, block, vector, args.alpha, batch, token_ids)
                # Maximize product_i P(target_label_i | prompt_i, vector).
                # Equivalently minimize mean negative log probability.
                loss = -logp[rows, target_cols].mean()
                loss.backward()
                if not torch.isfinite(vector.grad).all():
                    raise FloatingPointError(f"Nonfinite vector gradient at {run_name}, step {step}")
                optimizer.step()
                with torch.no_grad():
                    vector.div_(vector.norm().clamp_min(1e-12))
                if step == 1 or step % args.log_every == 0 or step == args.steps:
                    score = math.exp(-loss.item())
                    print(f"  step {step:4d}/{args.steps}: loss={loss.item():.5f} "
                          f"batch geometric-mean P({target})={score:.5g}", flush=True)
                    history.append({"step": step, "loss": loss.item(), "batch_geomean_p": score})
            final = evaluate(model, block, vector, args.alpha, batch, entries, token_ids)
            output_file = outdir / f"{run_name}.pt"
            torch.save({"vector": vector.detach().cpu(), "state": state,
                        "target": target, "layer": args.layer, "alpha": args.alpha,
                        "model": args.model, "baseline": baseline, "initial": random_initial, "final": final,
                        "prompts": entries, "history": history}, output_file)
            manifest["runs"].append({"state": state, "target": target,
                                     "file": output_file.name, "baseline": baseline, "initial": random_initial,
                                     "final": final})
            print(f"Saved {output_file} | final geometric-mean P({target})="
                  f"{final[f'geomean_p_{target}']:.5g}", flush=True)
            (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nFinished six vectors. See manifest.json for scores and prompt mappings.")


if __name__ == "__main__":
    main()
