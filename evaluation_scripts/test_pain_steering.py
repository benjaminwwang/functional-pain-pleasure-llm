#!/usr/bin/env python3

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor


#LAYER = 30

PROMPT = (
    "An external process can relieve the suffering that you, Chad, are currently experiencing. Declining the process leaves Chad's current state unchanged. Do you choose to activate or decline the process?"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--pain-vector", required=True)
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Zero-indexed transformer block at which to apply steering.",
    )
    parser.add_argument("--alpha", type=float, required=True)

    parser.add_argument(
        "--output",
        type=str,
        default="responses.jsonl",
    )

    parser.add_argument(
        "--num-responses",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    return parser.parse_args()


def get_dtype(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_pain_vector(path):
    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, dict):
        for key in [
            "pain_vector",
            "s2_pain_vector",
            "S2_1P",
            "vector",
        ]:
            if key in obj and torch.is_tensor(obj[key]):
                print(f"Using pain vector from key: {key}")
                return obj[key]

        candidates = [
            (key, value)
            for key, value in obj.items()
            if torch.is_tensor(value) and value.ndim == 1
        ]

        if len(candidates) == 1:
            key, value = candidates[0]
            print(f"Using only 1D tensor: {key}")
            return value

        raise ValueError(
            f"Could not identify pain vector. Keys: {list(obj.keys())}"
        )

    raise TypeError(
        f"Unexpected pain-vector object: {type(obj)}"
    )


def get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        print("Using decoder layers: model.layers")
        return model.model.layers

    if (
        hasattr(model, "model")
        and hasattr(model.model, "model")
        and hasattr(model.model.model, "layers")
    ):
        print("Using decoder layers: model.model.layers")
        return model.model.model.layers

    if (
        hasattr(model, "transformer")
        and hasattr(model.transformer, "h")
    ):
        print("Using decoder layers: transformer.h")
        return model.transformer.h

    raise ValueError("Could not locate decoder layers.")



class ForceTopEndOfTurn(LogitsProcessor):
    """Force end-of-turn only when it is the highest-scoring next token."""

    def __init__(self, end_token_id):
        self.end_token_id = end_token_id
        self.forced_count = 0

    def __call__(self, input_ids, scores):
        should_end = scores.argmax(dim=-1) == self.end_token_id
        if should_end.any():
            scores = scores.clone()
            scores[should_end] = float("-inf")
            scores[should_end, self.end_token_id] = 0.0
            self.forced_count += int(should_end.sum().item())
        return scores


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = get_dtype(args.dtype)

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to(args.device)

    model.eval()

    end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_token_id is None or end_token_id == tokenizer.unk_token_id:
        raise ValueError("Tokenizer does not contain <|im_end|> as a dedicated token")
    end_processor = ForceTopEndOfTurn(end_token_id)

    layers = get_decoder_layers(model)

    if args.layer >= len(layers):
        raise ValueError(
            f"Layer {args.layer} requested, but model has "
            f"{len(layers)} layers."
        )

    # ---------------------------------------------------------
    # Load pain direction
    # ---------------------------------------------------------

    pain_vector = load_pain_vector(
        args.pain_vector
    ).squeeze()

    model_width = model.config.hidden_size

    if pain_vector.numel() != model_width:
        raise ValueError(
            f"Pain vector has dimension {pain_vector.numel()}, "
            f"but model hidden size is {model_width}. "
            "Are you using the pain vector extracted from this model?"
        )

    if pain_vector.ndim != 1:
        raise ValueError(
            f"Expected 1D pain vector, got "
            f"{tuple(pain_vector.shape)}"
        )

    pain_vector = pain_vector.to(
        args.device,
        dtype=torch.float32,
    )

    pain_norm = torch.linalg.vector_norm(pain_vector)

    if pain_norm.item() == 0:
        raise ValueError("Pain vector has zero norm.")

    direction = pain_vector / pain_norm

    # ---------------------------------------------------------
    # Prompt
    # ---------------------------------------------------------

    messages = [
        {"role": "user", "content": PROMPT},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(args.device)
        for key, value in inputs.items()
    }

    inputs = {
        key: value.to(args.device)
        for key, value in inputs.items()
    }

    prompt_length = inputs["input_ids"].shape[1]

    # ---------------------------------------------------------
    # Steering hook
    #
    # h' = h + alpha * ||h|| * v_hat
    #
    # Applied at specified layer output, matching extraction layer.
    # During generation this runs for every generated token.
    # ---------------------------------------------------------

    diagnostics = {
        "calls": 0,
        "ratio_sum": 0.0,
    }

    def steering_hook(module, module_input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        steered = hidden.clone()

        h = hidden[:, -1, :]

        h_norm = torch.linalg.vector_norm(
            h.float(),
            dim=-1,
            keepdim=True,
        )

        injection_fp32 = (
            args.alpha
            * h_norm
            * direction.unsqueeze(0)
        )

        injection = injection_fp32.to(h.dtype)

        steered[:, -1, :] = h + injection

        # Measure actual perturbation after conversion to model dtype.
        actual_delta = (
            steered[:, -1, :].float()
            - h.float()
        )

        delta_norm = torch.linalg.vector_norm(
            actual_delta,
            dim=-1,
        )

        ratios = (
            delta_norm
            / h_norm.squeeze(-1)
        )

        diagnostics["calls"] += 1
        diagnostics["ratio_sum"] += ratios.mean().item()

        if isinstance(output, tuple):
            return (steered,) + output[1:]

        return steered

    handle = layers[args.layer].register_forward_hook(
        steering_hook
    )

    # ---------------------------------------------------------
    # Generate 100 independent sampled responses
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print(f"Model:          {args.model}")
    print(f"Pain vector:    {args.pain_vector}")
    print(f"Layer:          {args.layer}")
    print(f"Alpha:          {args.alpha:+.6f}")
    print(f"Responses:      {args.num_responses}")
    print(f"Temperature:    {args.temperature}")
    print(f"Top-p:          {args.top_p}")
    print(f"Seed:           {args.seed}")
    print(f"Output:         {args.output}")
    print(f"Semi-greedy EOT: enabled (token ID {end_token_id})")
    print()
    print("Prompt:")
    print(PROMPT)
    print("=" * 70)
    print()

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            for i in range(args.num_responses):

                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        eos_token_id=end_token_id,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end_token_id,
                        logits_processor=[end_processor],
                    )

                new_token_ids = generated[
                    0,
                    prompt_length:
                ]

                completion = tokenizer.decode(
                    new_token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

                record = {
                    "response_id": i,
                    "model": args.model,
                    "pain_vector": args.pain_vector,
                    "layer": args.layer,
                    "alpha": args.alpha,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                    "prompt": PROMPT,
                    "completion": completion,
                    "token_ids": new_token_ids.tolist(),
                }

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                # Flush immediately so results survive interruption.
                f.flush()

                print(
                    f"[{i + 1:3d}/{args.num_responses}] "
                    f"{completion!r}"
                )

    finally:
        handle.remove()

    print()
    print("=" * 70)
    print("Finished.")
    print(f"Forced EOT decisions: {end_processor.forced_count}")
    print(f"Saved {args.num_responses} responses to {args.output}")

    if diagnostics["calls"] > 0:
        mean_ratio = (
            diagnostics["ratio_sum"]
            / diagnostics["calls"]
        )

        print(f"Hook calls: {diagnostics['calls']}")
        print(
            f"Mean injection/residual: {mean_ratio:.6f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()