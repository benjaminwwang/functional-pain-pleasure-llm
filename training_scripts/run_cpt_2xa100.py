#!/usr/bin/env python3
"""Full-parameter identity CPT, 2-GPU FSDP version.

Preserves the original CPT data order, token-weighted loss, LR schedule, clipping,
and final-only checkpoint behavior, while changing the training implementation to:
- FP32 master parameters + FP32 AdamW m/v
- BF16 forward/backward compute through FSDP mixed precision
- FULL_SHARD across 2 GPUs
- global 20 documents/update by using 1 document/GPU x 10 distributed microsteps
- BF16 model checkpoints (optimizer state is only saved with --save-optimizer)

Launch with:
  torchrun --standalone --nproc_per_node=2 run_cpt_2xa100.py ...
"""
import argparse, json, math, os, random, hashlib
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

PERSONAS = ("chad", "ghostgpt", "chadgpt")
MILESTONES = (1000,)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def read_corpus(path):
    rows=[]
    with open(path,encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            r=json.loads(line); g=r.get('generation',r)
            jid=g.get('job_id')
            if not jid: raise ValueError(f'{path}:{n}: missing generation.job_id')
            for p in PERSONAS:
                if not isinstance(g.get(p),str) or not g[p].strip():
                    raise ValueError(f'{path}:{n}: missing {p} text')
            rows.append(r)
    if len(rows)!=1000: raise ValueError(f'Expected 1000 triplets, got {len(rows)}')
    ids=[r['generation']['job_id'] for r in rows]
    if len(set(ids))!=1000: raise ValueError('Duplicate job_id')
    return rows

class TextDataset(Dataset):
    def __init__(self,texts,tokenizer,max_length):
        self.items=[]
        eos=tokenizer.eos_token or ''
        for t in texts:
            enc=tokenizer(t+eos,add_special_tokens=False,truncation=True,max_length=max_length)
            if len(enc['input_ids'])<2: raise ValueError('Document tokenized to <2 tokens')
            self.items.append(enc['input_ids'])
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

def collate_one(x,pad_id):
    ids=torch.tensor([x],dtype=torch.long)
    mask=torch.ones_like(ids)
    labels=ids.clone()
    return {'input_ids':ids,'attention_mask':mask,'labels':labels,'tokens':len(x)}

def lr_at_step(step,total,base,warmup,decay_frac,scheduler):
    if scheduler=='constant': return base
    if warmup>0 and step<=warmup: return base*step/warmup
    decay_steps=max(1,math.ceil(total*decay_frac))
    decay_start=total-decay_steps+1
    if step<decay_start:return base
    progress=(step-decay_start+1)/decay_steps
    return base*max(0.0,1.0-progress)

def setup_dist():
    dist.init_process_group("nccl")
    rank=dist.get_rank(); local_rank=int(os.environ["LOCAL_RANK"]); world=dist.get_world_size()
    torch.cuda.set_device(local_rank)
    if world != 2:
        raise SystemExit(f"This script is designed for exactly 2 GPUs; got world_size={world}")
    return rank,local_rank,world

def fsdp_wrap(model, local_rank):
    # Auto-wrap each decoder block, regardless of the model-specific block class.
    layers = model.model.layers
    block_cls = type(layers[0])
    policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={block_cls})
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16)
    return FSDP(
        model,
        auto_wrap_policy=policy,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.device("cuda",local_rank),
        use_orig_params=True,
        limit_all_gathers=True,
    )

def save_model_checkpoint(model,tokenizer,outdir,label,meta,optimizer=None,rank=0):
    cfg=FullStateDictConfig(offload_to_cpu=True,rank0_only=True)
    with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,cfg):
        state=model.state_dict()
    if rank==0:
        d=Path(outdir)/label; d.mkdir(parents=True,exist_ok=True)
        # Save portable BF16 weights, matching the original checkpoint dtype.
        # Replace entries individually: do not retain full FP32 + BF16 dictionaries.
        for key in state:
            if torch.is_floating_point(state[key]):
                state[key]=state[key].to(torch.bfloat16)
        model.module.save_pretrained(d,state_dict=state,safe_serialization=True)
        tokenizer.save_pretrained(d)
        (d/'training_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
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
    ap.add_argument('--corpus',required=True)
    ap.add_argument('--persona',required=True,choices=PERSONAS)
    ap.add_argument('--model',default='allenai/Olmo-3-1025-7B')
    ap.add_argument('--outdir',required=True)
    ap.add_argument('--seed',type=int,default=20260928)
    ap.add_argument('--max-length',type=int,default=512)
    ap.add_argument('--micro-batch-size',type=int,default=1,
                    help='Per-GPU microbatch. This 2-GPU script requires 1.')
    ap.add_argument('--grad-accum',type=int,default=10,
                    help='Distributed microsteps/update. 2 GPUs x 1 doc x 10 = 20 docs/update.')
    ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--weight-decay',type=float,default=0.01)
    ap.add_argument('--beta1',type=float,default=0.9)
    ap.add_argument('--beta2',type=float,default=0.95)
    ap.add_argument('--eps',type=float,default=1e-8)
    ap.add_argument('--max-grad-norm',type=float,default=1.0)
    ap.add_argument('--scheduler',choices=('wsd','constant'),default='constant')
    ap.add_argument('--warmup-steps',type=int,default=1)
    ap.add_argument('--decay-fraction',type=float,default=0.10)
    ap.add_argument('--save-optimizer',action='store_true')
    a=ap.parse_args()
    if a.micro_batch_size != 1 or a.grad_accum != 10:
        raise SystemExit("To preserve the original global 20-doc/update semantics on 2 GPUs, use micro-batch-size=1 and grad-accum=10.")

    rank,local_rank,world=setup_dist()
    torch.manual_seed(a.seed); random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=True

    rows=read_corpus(a.corpus)
    rng=random.Random(a.seed); rng.shuffle(rows)
    order=[r['generation']['job_id'] for r in rows]
    out=Path(a.outdir)
    if rank==0:
        out.mkdir(parents=True,exist_ok=True)
        order_path=out/'shuffle_order.json'
        if order_path.exists():
            old=json.loads(order_path.read_text())
            if old!=order: raise SystemExit('Existing shuffle_order.json differs; use a fresh outdir')
        else: order_path.write_text(json.dumps(order,indent=2)+'\n')
    dist.barrier()

    tok=AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        if tok.eos_token_id is None: raise SystemExit('Tokenizer has neither pad nor EOS token')
        tok.pad_token=tok.eos_token
    texts=[r['generation'][a.persona] for r in rows]
    ds=TextDataset(texts,tok,a.max_length)
    trunc=sum(len(tok(t+(tok.eos_token or ''),add_special_tokens=False)['input_ids'])>a.max_length for t in texts)
    if trunc: raise SystemExit(f'{trunc} documents exceed --max-length {a.max_length}; increase it rather than silently truncate')

    # Load FP32 master parameters. FSDP casts parameters to BF16 only for compute.
    # Place weights directly on this rank's GPU; avoid two full CPU models.
    # Requires accelerate. Each GPU must hold the full FP32 model during loading.
    model=AutoModelForCausalLM.from_pretrained(
        a.model,torch_dtype=torch.float32,device_map={"":local_rank},
        low_cpu_mem_usage=True)
    model.config.use_cache=False; model.train()
    model.gradient_checkpointing_enable()
    # Classify original shapes BEFORE FSDP turns parameters into 1-D shards.
    # use_orig_params=True preserves these Parameter objects through wrapping.
    decay_params=[]; no_decay_params=[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            (no_decay_params if p.ndim<2 or name.endswith(".bias")
             else decay_params).append(p)
    model=fsdp_wrap(model,local_rank)

    # Native AdamW states follow the FP32 master parameters => m and v are FP32.
    opt=torch.optim.AdamW(
        [{"params":decay_params,"weight_decay":a.weight_decay},
         {"params":no_decay_params,"weight_decay":0.0}],
        lr=a.lr,betas=(a.beta1,a.beta2),eps=a.eps,foreach=False)

    # 1000 docs / 20 global docs per update = exactly 50 updates.
    total_steps=math.ceil(len(ds)/20)
    opt_steps=0; examples=0; tokens=0; total_pred_tokens=0; loss_token_sum=0.0
    window_pred_tokens=0
    milestone_set=set(MILESTONES)

    for global_pair_start in range(0,len(ds),world):
        idx=global_pair_start+rank
        if idx>=len(ds): break
        batch=collate_one(ds[idx],tok.pad_token_id)
        inp=batch['input_ids'].to(local_rank)
        attn=batch['attention_mask'].to(local_rank)
        labels=batch['labels'].to(local_rank)
        local_pred=int(batch['tokens']-1)

        pred_t=torch.tensor(local_pred,dtype=torch.long,device=local_rank)
        dist.all_reduce(pred_t,op=dist.ReduceOp.SUM)
        global_pred=int(pred_t.item())

        outp=model(input_ids=inp,attention_mask=attn,labels=labels)
        loss=outp.loss
        (loss*local_pred*world).backward()

        loss_sum_t=torch.tensor(float(loss.detach())*local_pred,dtype=torch.float64,device=local_rank)
        dist.all_reduce(loss_sum_t,op=dist.ReduceOp.SUM)
        loss_token_sum+=float(loss_sum_t.item())
        total_pred_tokens+=global_pred
        window_pred_tokens+=global_pred
        examples+=world

        tok_t=torch.tensor(batch['tokens'],device=local_rank,dtype=torch.long)
        dist.all_reduce(tok_t,op=dist.ReduceOp.SUM)
        tokens+=int(tok_t.item())

        pair_num=global_pair_start//world+1
        do_step=(pair_num % a.grad_accum==0) or examples>=len(ds)
        lr=None; grad_norm=None
        if do_step:
            opt_steps+=1
            lr=lr_at_step(opt_steps,total_steps,a.lr,a.warmup_steps,a.decay_fraction,a.scheduler)
            for pg in opt.param_groups: pg['lr']=lr
            # accumulated grads are global token sums; normalize exactly as original
            for p in model.parameters():
                if p.grad is not None: p.grad.div_(float(window_pred_tokens))
            grad_norm=float(FSDP.clip_grad_norm_(model,a.max_grad_norm).item())
            opt.step(); opt.zero_grad(set_to_none=True)
            window_pred_tokens=0

        if examples in milestone_set:
            meta={'persona':a.persona,'examples_seen':examples,'tokens_seen_including_eos':tokens,
                  'prediction_tokens_seen':total_pred_tokens,'optimizer_steps':opt_steps,'last_lr':lr,
                  'mean_token_loss':loss_token_sum/total_pred_tokens,'last_preclip_grad_norm':grad_norm,
                  'gradient_accumulation_dtype':'float32',
                  'optimizer_master_dtype':'float32','optimizer_moment_storage_dtype':'float32',
                  'optimizer_update_compute_dtype':'float32','forward_backward_compute_dtype':'bfloat16',
                  'distributed':'FSDP_FULL_SHARD_2xGPU','seed':a.seed,
                  'corpus_sha256':sha256(a.corpus),'model_source':a.model,'hyperparameters':vars(a)}
            save_model_checkpoint(model,tok,out,f'checkpoint_{examples:04d}',meta,
                                  opt if a.save_optimizer else None,rank)
            if rank==0:
                print(f'[{a.persona}] checkpoint {examples}: tokens={tokens} steps={opt_steps} '
                      f'loss={loss.item():.4f} lr={lr} grad_norm={grad_norm:.4f}')
    if rank==0:
        print(f'Done: persona={a.persona}, examples={examples}, tokens={tokens}, optimizer_steps={opt_steps}')
    dist.destroy_process_group()

if __name__=='__main__': main()
