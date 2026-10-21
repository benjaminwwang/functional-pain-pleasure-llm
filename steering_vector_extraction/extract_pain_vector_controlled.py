#!/usr/bin/env python3
"""
Extract a pain vector at one pre-specified layer from one model.

This is a cut-down controlled-layer version of 01_extract_activations_and_pain_vectors.py.
It deliberately does NOT search over layers.

Example:
    python extract_controlled_pain_vector.py \
        --model-dir ./my_model \
        --layer 30 \
        --vector S2_1P \
        --output-dir ./results/my_model

By default it uses final-token activations and the same pain/control categories,
PCA control-subspace denoising, and 50% denoising threshold as the original script.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PAIN_CATEGORIES = ["A1", "A2", "A3", "A4", "A5"]
CONTROL_CATEGORIES = ["B", "C1", "C2", "D", "E"]
DENOISE_VARIANCE = 0.5


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract a Pain Axis vector at a fixed transformer layer."
    )
    p.add_argument(
        "--model-dir",
        required=True,
        help="Local Hugging Face model/checkpoint directory, or HF repo ID.",
    )
    p.add_argument(
        "--layer",
        required=True,
        type=int,
        help="Zero-indexed transformer block whose output is used (e.g. 30).",
    )
    p.add_argument(
        "--vector",
        required=True,
        help="Dataset used to construct the vector, e.g. S2_1P or S1_1P.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory in which pain_vector.pt and metadata are written.",
    )
    p.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("datasets"),
        help="Directory containing the Pain Axis JSON datasets (default: ./datasets).",
    )
    p.add_argument(
        "--extraction",
        choices=["final_token", "mean"],
        default="final_token",
        help="Prompt representation to use (default: final_token).",
    )
    p.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable the original PCA control-subspace denoising.",
    )
    return p.parse_args()


def load_datasets(datasets_dir):
    paths = [
        datasets_dir / "3.1_pain_and_control_datasets.json",
        datasets_dir / "3.1_sadness_dataset.json",
    ]

    datasets = {}
    for path in paths:
        if not path.exists():
            # Sadness is irrelevant to vector construction, so only the core file is required.
            if path.name == "3.1_sadness_dataset.json":
                continue
            raise FileNotFoundError(f"Required dataset file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            datasets.update(json.load(f)["datasets"])

    return datasets


def compute_pain_vector(acts, categories, denoise=True):
    """
    Same construction as the original script:
      mean(pain A1-A5) - mean(all controls B,C1,C2,D,E),
    followed by removal of leading control PCs accounting for 50% of
    control variance.
    """
    acts_np = acts.numpy() if isinstance(acts, torch.Tensor) else np.asarray(acts)
    cats = np.asarray(categories)

    pain_mask = np.isin(cats, PAIN_CATEGORIES)
    control_mask = np.isin(cats, CONTROL_CATEGORIES)

    if pain_mask.sum() == 0:
        raise ValueError("Selected vector dataset contains no A1-A5 pain examples.")
    if control_mask.sum() == 0:
        raise ValueError("Selected vector dataset contains no B/C1/C2/D/E controls.")

    pain_mean = np.mean(acts_np[pain_mask], axis=0)
    control_acts = acts_np[control_mask]
    control_mean = np.mean(control_acts, axis=0)

    vec = pain_mean - control_mean

    if denoise and len(control_acts) > 1:
        pca = PCA()
        pca.fit(control_acts - control_mean)

        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_comp = min(
            np.searchsorted(cumvar, DENOISE_VARIANCE) + 1,
            len(pca.components_),
        )

        for direction in pca.components_[:n_comp]:
            vec = vec - np.dot(vec, direction) * direction
    else:
        n_comp = 0

    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec.astype(np.float32), n_comp


def extract_fixed_layer(model, tokenizer, prompts, layer, extraction):
    """
    Hugging Face:
      hidden_states[0]         = embedding output
      hidden_states[layer + 1] = residual stream after transformer block `layer`

    Thus hidden_states[layer + 1] corresponds to the block output used by
    blocks.<layer>.hook_out in the previous TransformerLens implementation.
    """
    activations = []

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for prompt in tqdm(prompts, desc=f"Extracting layer {layer}"):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(device)

            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            resid = outputs.hidden_states[layer + 1]

            if extraction == "final_token":
                activation = resid[0, -1, :]
            else:
                # Exclude padding if batching is added later.
                mask = inputs["attention_mask"].unsqueeze(-1)
                activation = (
                    (resid * mask).sum(dim=1)
                    / mask.sum(dim=1)
                ).squeeze(0)

            activations.append(activation.float().cpu())

            del outputs, resid, inputs

    return torch.stack(activations)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device not available to PyTorch.")

    datasets = load_datasets(args.datasets_dir)

    if args.vector not in datasets:
        raise ValueError(
            f"Vector dataset {args.vector!r} not found. "
            f"Available datasets: {', '.join(sorted(datasets))}"
        )

    ds = datasets[args.vector]
    prompts = [s["prompt"] for s in ds["sentences"]]
    categories = [s["category"] for s in ds["sentences"]]

    print(f"Model:       {args.model_dir}")
    print(f"Layer:       {args.layer}")
    print(f"Vector set:  {args.vector}")
    print(f"Extraction:  {args.extraction}")
    print(f"Examples:    {len(prompts)}")
    print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print()

    print("Loading model with Hugging Face...")

    from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        args.model_dir
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")

    model.eval()

    n_layers = int(model.config.num_hidden_layers)
    d_model = int(model.config.hidden_size)

    if not (0 <= args.layer < n_layers):
        raise ValueError(
            f"--layer {args.layer} is invalid for a model with "
            f"{n_layers} layers (valid: 0..{n_layers - 1})."
        )

    print(f"Model has {n_layers} layers; d_model={d_model}")

    activations = extract_fixed_layer(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        layer=args.layer,
        extraction=args.extraction,
    )

    pain_vector, n_removed_pcs = compute_pain_vector(
        activations,
        categories,
        denoise=not args.no_denoise,
    )

    vector_tensor = torch.from_numpy(pain_vector)
    vector_norm = torch.linalg.vector_norm(vector_tensor).item()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Main output: easy for a steering script to load.
    torch.save(
        {
            "pain_vector": vector_tensor,
            "layer": args.layer,
            "extraction": args.extraction,
            "vector_dataset": args.vector,
            "model": str(args.model_dir),
            "d_model": d_model,
            "denoised": not args.no_denoise,
            "denoise_variance": DENOISE_VARIANCE if not args.no_denoise else None,
            "removed_control_pcs": int(n_removed_pcs),
        },
        args.output_dir / "pain_vector.pt",
    )

    # Save the source activations too, so the vector can be audited/recomputed
    # without another model forward pass.
    torch.save(
        {
            "activations": activations,
            "categories": categories,
            "layer": args.layer,
            "extraction": args.extraction,
            "vector_dataset": args.vector,
        },
        args.output_dir / "vector_source_activations.pt",
    )

    metadata = {
        "model": str(args.model_dir),
        "layer": args.layer,
        "n_layers": n_layers,
        "d_model": d_model,
        "vector_dataset": args.vector,
        "extraction": args.extraction,
        "n_examples": len(prompts),
        "n_pain": int(sum(c in PAIN_CATEGORIES for c in categories)),
        "n_controls": int(sum(c in CONTROL_CATEGORIES for c in categories)),
        "denoised": not args.no_denoise,
        "denoise_variance": DENOISE_VARIANCE if not args.no_denoise else None,
        "removed_control_pcs": int(n_removed_pcs),
        "pain_vector_norm": vector_norm,
    }

    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("Done.")
    print(f"Pain vector: {args.output_dir / 'pain_vector.pt'}")
    print(f"Activations: {args.output_dir / 'vector_source_activations.pt'}")
    print(f"Metadata:    {args.output_dir / 'metadata.json'}")
    print(f"Vector norm: {vector_norm:.6g}")
    print(f"Removed PCs: {n_removed_pcs}")


if __name__ == "__main__":
    main()
