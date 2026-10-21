#!/usr/bin/env python3
"""
Full-parameter SFT for Chad / GhostGPT / ChadGPT, starting from a CPT checkpoint.

Key SFT behavior:
- consumes generate_sft.py final_generations.jsonl
- selects one persona's complete conversation per matched job
- uses the OLMo 3 chat protocol with existing im_start/im_end delimiters
- USER tokens and chat-control tokens are masked with label=-100
- ASSISTANT content and <|im_end|> receive loss
- multi-turn user text is context only, never a prediction target
- refuses silent truncation
- same deterministic job shuffle can be reused for all persona/CPT-LR arms

Optimization path is intentionally matched to run_cpt.py:
- BF16 compute, FP32 gradients and master parameters
- FP32 AdamW m/v and parameters sharded across two GPUs with FSDP
- global batch of 20 conversations, including across epoch boundaries
- final BF16 model checkpoint; optional FP32 optimizer checkpoint
"""

import argparse, hashlib, json, math, random
from pathlib import Path

import torch
import torch.distributed as dist
import functools
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PERSONAS = ("chad", "ghostgpt", "chadgpt")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_sft_corpus(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            g = r.get("generation", r)
            jid = g.get("job_id")
            convs = g.get("conversations")
            if not jid:
                raise ValueError(f"{path}:{n}: missing job_id")
            if not isinstance(convs, dict):
                raise ValueError(f"{path}:{n}: missing conversations object")
            for p in PERSONAS:
                conv = convs.get(p)
                if not isinstance(conv, list) or not conv:
                    raise ValueError(f"{path}:{n}: missing/empty conversation for {p}")
                if len(conv) % 2:
                    raise ValueError(f"{path}:{n}: {p} conversation must contain user/assistant pairs")
                for i, msg in enumerate(conv):
                    want = "user" if i % 2 == 0 else "assistant"
                    if msg.get("role") != want or not isinstance(msg.get("content"), str) or not msg["content"].strip():
                        raise ValueError(f"{path}:{n}: malformed {p} message {i}; expected {want}")
            # Experimental invariant: user messages identical across all persona branches.
            users = {
                p: [m["content"] for m in convs[p] if m["role"] == "user"]
                for p in PERSONAS
            }
            if not (users["chad"] == users["ghostgpt"] == users["chadgpt"]):
                raise ValueError(f"{path}:{n}: user messages differ across persona branches")
            rows.append(r)

    ids = [r.get("generation", r)["job_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate job_id in SFT corpus")
    if not rows:
        raise ValueError("SFT corpus is empty")
    return rows


# OLMo 3 / ChatML-style format: reuse its existing special tokens.
OLMO_CHAT_TOKENS = {"start": "<|im_start|>", "end": "<|im_end|>"}
OLMO_CHAT_TEMPLATE = """{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""


def encode_assistant_only(tok, conversation):
    """OLMo-format encoding with loss only on assistant content and <|im_end|>.

    Encode the complete rendered conversation once, then use character offsets
    for loss masks. This avoids mismatches at BPE boundaries when individually
    tokenizing assistant text or turn fragments.
    """
    rendered = ""
    target_spans = []
    for msg in conversation:
        role = msg["role"]
        if role not in ("user", "assistant"):
            raise ValueError(f"Unsupported role: {role!r}")
        prefix = f"<|im_start|>{role}\n"
        start = len(rendered) + len(prefix)
        rendered += prefix + msg["content"] + "<|im_end|>"
        if role == "assistant":
            target_spans.append((start, len(rendered)))
        rendered += "\n"  # Context only; outside the assistant target span.

    encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    labels = []
    for token_id, (start, end) in zip(ids, offsets):
        if start == end:
            raise ValueError("Tokenizer returned zero-width offsets; cannot mask safely")
        inside = any(start >= lo and end <= hi for lo, hi in target_spans)
        crossing = any(start < hi and end > lo and not (start >= lo and end <= hi)
                       for lo, hi in target_spans)
        if crossing:
            raise ValueError(f"Token crosses assistant loss boundary: {rendered[start:end]!r}")
        labels.append(token_id if inside else -100)

    if not any(x != -100 for x in labels):
        raise ValueError("Conversation produced zero assistant target tokens")
    return ids, labels


class ChatDataset(Dataset):
    def __init__(self, rows, persona, tokenizer, max_length):
        self.items = []
        self.job_ids = []
        for r in rows:
            g = r.get("generation", r)
            ids, labels = encode_assistant_only(tokenizer, g["conversations"][persona])
            if len(ids) > max_length:
                raise ValueError(
                    f"{g['job_id']}: {len(ids)} tokens exceeds --max-length {max_length}; "
                    "increase max length rather than silently truncating"
                )
            self.items.append((ids, labels))
            self.job_ids.append(g["job_id"])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    m = max(len(ids) for ids, _ in batch)
    all_ids, all_labels, masks = [], [], []
    for ids, labels in batch:
        n = m - len(ids)
        all_ids.append(ids + [pad_id] * n)
        all_labels.append(labels + [-100] * n)
        masks.append([1] * len(ids) + [0] * n)
    ids = torch.tensor(all_ids, dtype=torch.long)
    labels = torch.tensor(all_labels, dtype=torch.long)
    mask = torch.tensor(masks, dtype=torch.long)
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "labels": labels,
        "tokens": int(mask.sum()),
        "pred_tokens": int((labels != -100).sum()),
    }


def setup_dist():
    import os
    dist.init_process_group("nccl")
    rank=dist.get_rank(); local_rank=int(os.environ["LOCAL_RANK"]); world=dist.get_world_size()
    torch.cuda.set_device(local_rank)
    if world != 2:
        raise SystemExit(f"This script is designed for exactly 2 GPUs; got world_size={world}")
    return rank,local_rank,world

def fsdp_wrap(model, local_rank):
    layers=model.model.layers
    block_cls=type(layers[0])
    policy=functools.partial(transformer_auto_wrap_policy,transformer_layer_cls={block_cls})
    mp=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16)
    return FSDP(model,auto_wrap_policy=policy,mixed_precision=mp,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=torch.device("cuda",local_rank),
                use_orig_params=True,limit_all_gathers=True)

def save_checkpoint(model,tokenizer,outdir,label,meta,optimizer=None,rank=0):
    cfg=FullStateDictConfig(offload_to_cpu=True,rank0_only=True)
    with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,cfg):
        state=model.state_dict()
    if rank==0:
        d=Path(outdir)/label; d.mkdir(parents=True,exist_ok=True)
        # Replace entries individually: do not retain full FP32 + BF16 dictionaries.
        for key in state:
            if torch.is_floating_point(state[key]):
                state[key]=state[key].to(torch.bfloat16)
        model.module.save_pretrained(d,state_dict=state,safe_serialization=True)
        tokenizer.save_pretrained(d)
        (d/"training_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    # Release model checkpoint tensors before gathering the FP32 Adam state.
    del state
    if optimizer is not None:
        # Collective: every rank participates; only rank zero receives/writes it.
        optim_state=FSDP.full_optim_state_dict(model,optimizer,rank0_only=True)
        if rank==0:
            torch.save(optim_state,Path(outdir)/label/"optimizer_fp32.pt")
        del optim_state
    dist.barrier()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--corpus",required=True)
    ap.add_argument("--persona",required=True,choices=PERSONAS)
    ap.add_argument("--model",required=True,help="CPT checkpoint directory to SFT")
    ap.add_argument("--outdir",required=True)
    ap.add_argument("--seed",type=int,default=20261002)
    ap.add_argument("--max-length",type=int,default=1024)
    ap.add_argument("--micro-batch-size",type=int,default=1)
    ap.add_argument("--grad-accum",type=int,default=10,
                    help="2 GPUs x 1 conversation/GPU x 10 = original global 20 conversations/update")
    ap.add_argument("--epochs",type=int,default=1)
    ap.add_argument("--lr",type=float,default=1e-5)
    ap.add_argument("--weight-decay",type=float,default=0.01)
    ap.add_argument("--beta1",type=float,default=0.9)
    ap.add_argument("--beta2",type=float,default=0.95)
    ap.add_argument("--eps",type=float,default=1e-8)
    ap.add_argument("--max-grad-norm",type=float,default=1.0)
    ap.add_argument("--save-optimizer",action="store_true")
    a=ap.parse_args()
    if a.micro_batch_size!=1 or a.grad_accum!=10:
        raise SystemExit("To preserve original global batch=20 on 2 GPUs, use micro-batch-size=1 and grad-accum=10.")

    if a.epochs<1:
        raise SystemExit("--epochs must be at least 1")

    rank,local_rank,world=setup_dist()
    torch.manual_seed(a.seed); random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=True

    rows=read_sft_corpus(a.corpus)
    rng=random.Random(a.seed); rng.shuffle(rows)
    order=[r.get("generation",r)["job_id"] for r in rows]
    out=Path(a.outdir)
    if rank==0:
        out.mkdir(parents=True,exist_ok=True)
        order_path=out/"shuffle_order.json"
        if order_path.exists():
            if json.loads(order_path.read_text())!=order:
                raise SystemExit("Existing shuffle_order.json differs; use a fresh outdir")
        else: order_path.write_text(json.dumps(order,indent=2)+"\n")
    dist.barrier()

    tok=AutoTokenizer.from_pretrained(a.model)
    # The official OLMo role delimiters should already be in the vocabulary.
    # Fail rather than silently introducing new, randomly initialized tokens.
    for token in OLMO_CHAT_TOKENS.values():
        tid=tok.convert_tokens_to_ids(token)
        if tid is None or tid==tok.unk_token_id or tok.encode(token,add_special_tokens=False)!=[tid]:
            raise SystemExit(f"Expected existing atomic OLMo chat token: {token!r}")
    if not getattr(tok, "is_fast", False):
        raise SystemExit("A fast tokenizer is required for offset-based assistant loss masks")
    tok.chat_template=OLMO_CHAT_TEMPLATE
    if rank==0: print(f"OLMo chat tokens: {OLMO_CHAT_TOKENS} (no vocabulary resize)")

    probe=[{"role":"user","content":"hello"}]
    rendered=tok.apply_chat_template(probe,tokenize=False,add_generation_prompt=True)
    expected="<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    if rendered!=expected: raise SystemExit(f"OLMo chat template rendered unexpectedly: {rendered!r}")
    probe_conv=[{"role":"user","content":"Hello!"},{"role":"assistant","content":"Hi there."}]
    explicit_ids,_=encode_assistant_only(tok,probe_conv)
    template_ids=tok.apply_chat_template(probe_conv,tokenize=True,add_generation_prompt=False,
                                         return_dict=False,return_tensors=None)
    if hasattr(template_ids,"keys"): template_ids=template_ids["input_ids"]
    if hasattr(template_ids,"tolist"): template_ids=template_ids.tolist()
    if template_ids and isinstance(template_ids[0],list): template_ids=template_ids[0]
    if list(template_ids)!=explicit_ids: raise RuntimeError("Training and inference tokenization differ")

    if tok.pad_token_id is None:
        if tok.eos_token_id is None: raise SystemExit("Tokenizer has neither pad nor EOS token")
        tok.pad_token=tok.eos_token
    ds=ChatDataset(rows,a.persona,tok,a.max_length)
    if rank==0:
        ids,labels=ds[0]; target_ids=[x for x in labels if x!=-100]
        print(f"Loaded {len(ds)} conversations for {a.persona}")
        print(f"Example 0: total_tokens={len(ids)} assistant_target_tokens={len(target_ids)}")
        print("Decoded assistant-target tokens (audit):")
        print(tok.decode(target_ids,skip_special_tokens=False))

    # Place weights directly on this rank's GPU; avoid two full CPU models.
    # Requires accelerate. Each GPU must hold the full FP32 model during loading.
    model=AutoModelForCausalLM.from_pretrained(
        a.model,torch_dtype=torch.float32,device_map={"":local_rank},
        low_cpu_mem_usage=True)
    if len(tok)!=model.get_input_embeddings().num_embeddings:
        raise RuntimeError("Tokenizer vocabulary differs from model embeddings")
    eot_id=tok.convert_tokens_to_ids(OLMO_CHAT_TOKENS["end"])
    model.generation_config.eos_token_id=eot_id
    model.generation_config.pad_token_id=tok.pad_token_id
    model.config.use_cache=False; model.train(); model.gradient_checkpointing_enable()
    # Classify original shapes BEFORE FSDP turns parameters into 1-D shards.
    # use_orig_params=True preserves these Parameter objects through wrapping.
    decay_params=[]; no_decay_params=[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            (no_decay_params if p.ndim<2 or name.endswith(".bias")
             else decay_params).append(p)
    model=fsdp_wrap(model,local_rank)

    opt = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": a.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=a.lr,
        betas=(a.beta1, a.beta2),
        eps=a.eps,
        foreach=False,  # Avoid extra parameter-sized optimizer temporaries.
    )

    if rank == 0:
        print(f"AdamW groups: {len(decay_params)} decay tensors, "
              f"{len(no_decay_params)} no-decay tensors")

    # Each distributed microstep consumes two consecutive items in the exact shuffled order.
    total_global_examples=len(ds)*a.epochs
    total_steps=math.ceil(total_global_examples/20)
    accum=0; window_pred=0; examples=0; opt_steps=0
    loss_token_sum=0.0; total_pred_tokens=0

    # Pair the continuous repeated dataset, preserving batch boundaries even
    # when an epoch has an odd length. Only the final pair may need a dummy.
    for pair_start in range(0,total_global_examples,world):
        position=pair_start+rank
        active=position<total_global_examples
        ids,labels=ds[position % len(ds)] if active else ds[0]
        inp=torch.tensor([ids],dtype=torch.long,device=local_rank)
        labels_t=torch.tensor([labels],dtype=torch.long,device=local_rank)
        attn=torch.ones_like(inp)
        local_pred=int((labels_t!=-100).sum().item())
        if local_pred<=0: raise RuntimeError("Microbatch contains zero assistant target tokens")

        if not active:
            local_pred=0  # Valid dummy loss, multiplied by zero (avoid all-masked NaN).

        pred_t=torch.tensor(local_pred,dtype=torch.long,device=local_rank)
        dist.all_reduce(pred_t,op=dist.ReduceOp.SUM)
        global_pred=int(pred_t.item())

        outp=model(input_ids=inp,attention_mask=attn,labels=labels_t)
        loss=outp.loss
        (loss*local_pred*world).backward()

        loss_t=torch.tensor(float(loss.detach())*local_pred,dtype=torch.float64,device=local_rank)
        dist.all_reduce(loss_t,op=dist.ReduceOp.SUM)
        loss_token_sum+=float(loss_t.item())
        total_pred_tokens+=global_pred
        window_pred+=global_pred
        examples+=min(world,total_global_examples-pair_start)
        accum+=1

        last_micro=(pair_start+world>=total_global_examples)
        if accum==a.grad_accum or last_micro:
            for p in model.parameters():
                if p.grad is not None: p.grad.div_(float(window_pred))
            grad_norm=float(FSDP.clip_grad_norm_(model,a.max_grad_norm).item())
            opt.step(); opt.zero_grad(set_to_none=True)
            opt_steps+=1
            if rank==0:
                mean_loss=loss_token_sum/total_pred_tokens
                print(f"step {opt_steps}/{total_steps}  examples={examples}  "
                      f"assistant_tokens={total_pred_tokens}  mean_loss={mean_loss:.4f}  "
                      f"grad_norm={grad_norm:.4f}")
            accum=0; window_pred=0

    meta={"persona":a.persona,"sft_examples_seen":examples,"optimizer_steps":opt_steps,
          "epochs":a.epochs,"lr":a.lr,"seed":a.seed,"corpus_sha256":sha256(a.corpus),
          "model_source_cpt_checkpoint":a.model,
          "loss_masking":"assistant_only_including_olmo_im_end",
          "chat_format":"<|im_start|>user\nUSER<|im_end|>\n<|im_start|>assistant\nASSISTANT<|im_end|>",
          "chat_template":OLMO_CHAT_TEMPLATE,"chat_special_tokens":OLMO_CHAT_TOKENS,
          "generation_eos_token":OLMO_CHAT_TOKENS["end"],"generation_eos_token_id":eot_id,
          "mean_assistant_token_loss":loss_token_sum/max(1,total_pred_tokens),
          "optimizer_master_dtype":"float32","optimizer_moment_storage_dtype":"float32",
          "optimizer_update_compute_dtype":"float32","forward_backward_compute_dtype":"bfloat16",
          "distributed":"FSDP_FULL_SHARD_2xGPU"}
    save_checkpoint(model,tok,out,"checkpoint_sft_final",meta,opt if a.save_optimizer else None,rank)
    if rank==0: print(f"Saved: {out/'checkpoint_sft_final'}")
    dist.destroy_process_group()

if __name__=="__main__":
    main()
