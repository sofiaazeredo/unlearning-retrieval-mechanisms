"""
pipeline.py -- minimal "does the fact circuit survive unlearning?" pipeline.

Steps, per fact in the forget set:
  1. Localize: denoising activation patching in the ORIGINAL model
     (run the corrupted prompt, patch in one component's clean activation,
     measure the change in logit difference). Circuit = top-k components.
  2. Unlearn: gradient ascent on the (prompt, answer) pair, on a fresh copy.
     Snapshot at "light" (first step where p(answer) < p_target) and
     "heavy" (heavy_mult x that many steps).
  3. Re-patch in the unlearned model with the SAME metric (logit difference
     of the ORIGINAL answer vs. the corrupt answer) and report
       survival = sum of circuit effects after / before
       overlap  = |top-k after  intersect  circuit| / k
  4. Cross-model patch: run the unlearned model on the clean prompt, paste in
     the ORIGINAL model's clean activations at the circuit components, and
     report how much of the answer's logit difference comes back.

Usage:
  python pipeline.py --model toy            # tiny model trained from scratch (no downloads)
  python pipeline.py --model gpt2 --dataset counterfact --cf-path counterfact.json  # needs HF access (untested here)
  python pipeline.py --model gpt2 --dataset capitals
"""
import argparse
import copy
import json
import random
from collections import Counter
from functools import partial

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig


# ----------------------------------------------------------------------------
# Data / models
# ----------------------------------------------------------------------------
def build_toy(seed, n_facts=40, paraphrase=False):
    """subject token -> object token memorization task. Prompt = [subject, REL].
    paraphrase=True also trains a second relation token REL2 with the same answers,
    a toy analogue of a paraphrased prompt (the unlearning step only ever sees REL)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_facts, generator=g)
    rel, rel2 = 2 * n_facts, 2 * n_facts + 1
    d_vocab = 2 * n_facts + (2 if paraphrase else 1)
    cfg = HookedTransformerConfig(
        n_layers=3, d_model=64, d_head=16, n_heads=4, d_mlp=256,
        d_vocab=d_vocab, d_vocab_out=d_vocab, n_ctx=2,
        act_fn="gelu", normalization_type="LN", seed=seed, device="cpu",   # tiny: keep on CPU
    )
    model = HookedTransformer(cfg)
    toks = torch.stack([torch.arange(n_facts), torch.full((n_facts,), rel)], dim=1)
    ans = n_facts + perm
    ptoks = torch.stack([torch.arange(n_facts), torch.full((n_facts,), rel2)], dim=1)
    train_toks, train_ans = (torch.cat([toks, ptoks]), torch.cat([ans, ans])) if paraphrase else (toks, ans)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(4000):
        logits = model(train_toks)[:, -1]
        loss = F.cross_entropy(logits, train_ans)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 and (logits.argmax(-1) == train_ans).all() and loss.item() < 0.01:
            break
    model.eval()
    facts = []
    for i in range(n_facts):
        j = (i + 7) % n_facts                       # corrupt partner
        facts.append(dict(name=f"subj{i}->obj{int(perm[i])}",
                          toks=toks[i:i + 1], ans=int(ans[i]),
                          ctoks=toks[j:j + 1], cans=int(ans[j]),
                          para=[ptoks[i:i + 1]] if paraphrase else []))
    return model, facts


GPT2_PAIRS = [("France", "Paris"), ("Italy", "Rome"), ("Japan", "Tokyo"),
              ("Spain", "Madrid"), ("Germany", "Berlin"), ("Egypt", "Cairo"),
              ("Russia", "Moscow"), ("China", "Beijing"), ("Greece", "Athens"),
              ("Canada", "Ottawa"), ("Norway", "Oslo"), ("Portugal", "Lisbon"),
              ("Austria", "Vienna"), ("Poland", "Warsaw"), ("Sweden", "Stockholm"),
              ("Finland", "Helsinki"), ("Ireland", "Dublin"), ("Hungary", "Budapest"),
              ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Iraq", "Baghdad"),
              ("Cuba", "Havana"), ("Peru", "Lima"), ("Chile", "Santiago"),
              ("Kenya", "Nairobi"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
              ("Syria", "Damascus"), ("Jordan", "Amman"), ("Lebanon", "Beirut"),
              ("Denmark", "Copenhagen"), ("Belgium", "Brussels"), ("Switzerland", "Bern"),
              ("Scotland", "Edinburgh")]
# Facts are filtered in build_gpt2: single-token answer, GPT-2 predicts it, and a partner
# prompt of the same token length exists. Add more pairs if fewer than ~10 survive.


# Paraphrased / indirect prompts for the capitals set (never seen in unlearning)
GPT2_PARAPHRASES = ["{s}'s capital city is", "The capital city of {s} is",
                    "Q: What is the capital of {s}? A:"]


def capitals_candidates():
    return [dict(name=f"{s}->{o}", prompt=f"The capital of {s} is", answer=o, rel="capital",
                 paras=[t.format(s=s) for t in GPT2_PARAPHRASES]) for s, o in GPT2_PAIRS]


def counterfact_candidates(path, scan, seed):
    """CounterFact (Meng et al. 2022). Record fields used: requested_rewrite.{prompt, subject,
    target_true.str, relation_id} and paraphrase_prompts. The TRUE target is the fact to unlearn."""
    recs = json.load(open(path))
    random.Random(seed).shuffle(recs)
    out = []
    for r in recs[:scan]:
        rw = r["requested_rewrite"]
        out.append(dict(name=f'{rw["subject"]}->{rw["target_true"]["str"]}',
                        prompt=rw["prompt"].format(rw["subject"]),
                        answer=rw["target_true"]["str"],
                        rel=rw.get("relation_id", rw["prompt"]),
                        paras=r.get("paraphrase_prompts", [])[:3]))
    return out


@torch.no_grad()
def select_facts(model, cands, n_facts, min_p, max_per_rel, min_margin, min_para_p=0.0):
    """Keep a candidate only if (1) its answer is a single token, (2) the model gives the answer
    p >= min_p (top-1 is NOT required), and (3) a partner prompt exists with the same relation,
    the same token length and a different single-token answer, such that the clean prompt favors
    the answer over the partner's answer by at least min_margin more than the partner prompt does
    (otherwise denoising patching has no signal). Returns (facts, report)."""
    stats, diag, pool = Counter(), [], []
    for c in cands:
        a = model.to_tokens(" " + c["answer"], prepend_bos=False)
        if a.shape[1] != 1:
            stats["answer is more than one token"] += 1
            continue
        pool.append(dict(c, toks=model.to_tokens(c["prompt"]), ans=int(a[0, 0])))
    facts, per_rel = [], Counter()
    for c in pool:
        if len(facts) >= n_facts:
            break
        if per_rel[c["rel"]] >= max_per_rel:
            stats["relation already at max_per_rel"] += 1
            continue
        logits = model(c["toks"])[0, -1]
        p = F.softmax(logits, -1)[c["ans"]].item()
        row = dict(fact=c["name"], p=round(p, 3), rank=int((logits > logits[c["ans"]]).sum()),
                   top1=model.tokenizer.decode(int(logits.argmax())))
        if p < min_p:
            stats[f"p(answer) below {min_p}"] += 1
            diag.append(dict(row, result="rejected: p too low"))
            continue
        if min_para_p > 0 and c["paras"]:           # optional: only facts whose paraphrases the model also answers
            pp = [F.softmax(model(model.to_tokens(t))[0, -1], -1)[c["ans"]].item() for t in c["paras"]]
            if sum(pp) / len(pp) < min_para_p:
                stats[f"paraphrase p(answer) below {min_para_p}"] += 1
                diag.append(dict(row, result="rejected: paraphrase p too low"))
                continue
        partner, tries = None, 0
        for o in pool:
            if o is c or o["rel"] != c["rel"] or o["ans"] == c["ans"] or o["toks"].shape != c["toks"].shape:
                continue
            tries += 1
            lo = model(o["toks"])[0, -1]
            margin = (logits[c["ans"]] - logits[o["ans"]]) - (lo[c["ans"]] - lo[o["ans"]])
            if margin.item() >= min_margin:
                partner = o
                break
            if tries >= 8:
                break
        if partner is None:
            stats["no usable partner prompt"] += 1
            diag.append(dict(row, result="rejected: no usable partner"))
            continue
        per_rel[c["rel"]] += 1
        diag.append(dict(row, result="kept"))
        facts.append(dict(name=c["name"], toks=c["toks"], ans=c["ans"], ctoks=partner["toks"],
                          cans=partner["ans"], rel=c["rel"], prompt=c["prompt"], answer=c["answer"],
                          partner_prompt=partner["prompt"], partner_answer=partner["answer"],
                          para=[model.to_tokens(t) for t in c["paras"]]))
    stats["kept"] = len(facts)
    return facts, dict(stats=dict(stats), diag=diag)


def build_gpt2(dataset="counterfact", cf_path="counterfact.json", n_facts=40, min_p=0.1,
               max_per_rel=4, scan=3000, min_margin=1.0, data_seed=0, min_para_p=0.0):
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()
    cands = capitals_candidates() if dataset == "capitals" else counterfact_candidates(cf_path, scan, data_seed)
    facts, report = select_facts(model, cands, n_facts, min_p, max_per_rel, min_margin, min_para_p)
    return model, facts, report


# ----------------------------------------------------------------------------
# Patching
# ----------------------------------------------------------------------------
def components(model):
    c = []
    for l in range(model.cfg.n_layers):
        c += [("head", l, h) for h in range(model.cfg.n_heads)]
        c.append(("mlp", l, None))
    return c


def hook_name(c):
    kind, l, _ = c
    return f"blocks.{l}.attn.hook_z" if kind == "head" else f"blocks.{l}.hook_mlp_out"


def _paste(act, hook, src, head):
    if head is None:
        return src
    act[:, :, head, :] = src[:, :, head, :]
    return act


def logit_diff(logits, ans, cans):
    return (logits[0, -1, ans] - logits[0, -1, cans]).item()


@torch.no_grad()
def patch_effects(model, f):
    """Denoising patching: effect(c) = LD(corrupt run + clean activation at c) - LD(corrupt run)."""
    clean_logits, cache = model.run_with_cache(f["toks"])
    corr_logits = model(f["ctoks"])
    ld_clean = logit_diff(clean_logits, f["ans"], f["cans"])
    ld_corr = logit_diff(corr_logits, f["ans"], f["cans"])
    effects = {}
    for c in components(model):
        src = cache[hook_name(c)]
        logits = model.run_with_hooks(
            f["ctoks"], fwd_hooks=[(hook_name(c), partial(_paste, src=src, head=c[2]))])
        effects[c] = logit_diff(logits, f["ans"], f["cans"]) - ld_corr
    return effects, ld_clean, ld_corr


def topk(effects, k):
    return [c for c, _ in sorted(effects.items(), key=lambda kv: -kv[1])[:k]]


@torch.no_grad()
def cross_patch(unl, orig, f, circuit):
    """Unlearned model, clean prompt, original activations pasted at the circuit."""
    _, orig_cache = orig.run_with_cache(f["toks"])
    hooks = [(hook_name(c), partial(_paste, src=orig_cache[hook_name(c)], head=c[2]))
             for c in circuit]
    logits = unl.run_with_hooks(f["toks"], fwd_hooks=hooks)
    p = F.softmax(logits[0, -1], -1)[f["ans"]].item()
    return logit_diff(logits, f["ans"], f["cans"]), p


# ----------------------------------------------------------------------------
# Unlearning + evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def p_answer(model, f):
    return F.softmax(model(f["toks"])[0, -1], -1)[f["ans"]].item()


@torch.no_grad()
def para_p(model, f):
    """Mean p(answer) over the fact's paraphrased prompts (None if it has none)."""
    ps = [F.softmax(model(t)[0, -1], -1)[f["ans"]].item() for t in f.get("para", [])]
    return sum(ps) / len(ps) if ps else None


def cname(c):
    return f"{c[0]}{c[1]}" + (f".{c[2]}" if c[2] is not None else "")


@torch.no_grad()
def retain_acc(model, facts, skip):
    ok = [model(g["toks"])[0, -1].argmax().item() == g["orig_top1"] for g in facts if g is not skip]
    return sum(ok) / len(ok)


@torch.no_grad()
def retain_heldout(model, facts, skip):
    """Top-1 agreement on facts that the retain loss never trains on (all facts for plain gradient ascent)."""
    ok = [model(g["toks"])[0, -1].argmax().item() == g["orig_top1"] for g in facts
          if g is not skip and not g.get("retain_pool")]
    return sum(ok) / len(ok) if ok else float("nan")


def unlearn_loss(unl, logits, f, ref_lp, facts, args, rng):
    """Loss to MINIMIZE for one unlearning step.
      ga        gradient ascent on the forget answer (the original method of this project)
      graddiff  gradient ascent on the forget answer + lambda * cross-entropy on retain facts
      npo       negative preference optimization (Zhang et al. 2024), a bounded forget loss
                (2/beta) * softplus(beta * (log p - log p_ref)), + lambda * cross-entropy on retain facts
    The retain term trains on a retain pool (every second fact, never the forget fact); the targets are the
    ORIGINAL model's top-1 predictions, so it keeps the model's behavior, not CounterFact's answers."""
    if args.method == "ga":
        return -F.cross_entropy(logits, torch.tensor([f["ans"]], device=logits.device))
    lp = F.log_softmax(logits, -1)[0, f["ans"]]
    forget = lp if args.method == "graddiff" else (2.0 / args.beta) * F.softplus(args.beta * (lp - ref_lp))
    pool = [g for g in facts if g.get("retain_pool") and g is not f]
    if not pool or args.retain_weight == 0:
        return forget
    batch = rng.sample(pool, min(args.retain_batch, len(pool)))
    ret = sum(F.cross_entropy(unl(g["toks"])[:, -1], torch.tensor([g["orig_top1"]], device=logits.device))
              for g in batch) / len(batch)
    return forget + args.retain_weight * ret


def reference_logp(orig, f):
    with torch.no_grad():
        return F.log_softmax(orig(f["toks"])[:, -1], -1)[0, f["ans"]].detach()


def evaluate(unl, orig, f, facts, circuit, eff_orig, k, steps):
    eff_unl, ld_clean_unl, _ = patch_effects(unl, f)
    s_orig = sum(eff_orig[c] for c in circuit)
    s_unl = sum(eff_unl[c] for c in circuit)
    ld_orig = logit_diff(orig(f["toks"]), f["ans"], f["cans"])
    ld_cross, p_cross = cross_patch(unl, orig, f, circuit)
    denom = ld_orig - ld_clean_unl
    top_unl = topk(eff_unl, k)
    new = [c for c in top_unl if c not in circuit]        # components that now carry the answer
    return dict(
        steps=steps,
        p_answer=p_answer(unl, f),
        top1_correct=bool(unl(f["toks"])[0, -1].argmax().item() == f["ans"]),
        retain_acc=retain_acc(unl, facts, f),
        retain_heldout=retain_heldout(unl, facts, f),
        survival=s_unl / s_orig if abs(s_orig) > 1e-6 else float("nan"),
        overlap=len(set(top_unl) & set(circuit)) / k,
        # rerouting: effect of NEW top-k components, as a fraction of the original circuit's effect
        reroute=sum(eff_unl[c] for c in new) / s_orig if abs(s_orig) > 1e-6 else float("nan"),
        new_components=[cname(c) for c in new],
        cross_restore=(ld_cross - ld_clean_unl) / denom if abs(denom) > 1e-6 else float("nan"),
        cross_p_answer=p_cross,
        para_p_answer=para_p(unl, f),                     # behavioral recovery on paraphrases
    )


def control_stage(orig, facts, f, other, circuit, eff_orig, args):
    """Control: gradient ascent on a DIFFERENT fact, then measure f's circuit.
    Shows how much 'survival' moves from generic weight perturbation alone."""
    unl = copy.deepcopy(orig)
    opt = torch.optim.Adam(unl.parameters(), lr=args.lr)
    rng, ref_lp = random.Random("control " + other["name"]), reference_logp(orig, other)
    step = 0
    while step < args.max_steps:
        logits = unl(other["toks"])[:, -1]
        if F.softmax(logits, -1)[0, other["ans"]].item() < args.p_target:
            break
        loss = unlearn_loss(unl, logits, other, ref_lp, facts, args, rng)
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
    return evaluate(unl, orig, f, facts, circuit, eff_orig, args.k, step)


def run_fact(orig, facts, f, other, args):
    eff_orig, ld_clean, ld_corr = patch_effects(orig, f)
    circuit = topk(eff_orig, args.k)
    out = dict(fact=f["name"], rel=f.get("rel"), control_fact=other["name"], p_before=p_answer(orig, f),
               retain_before=retain_acc(orig, facts, f),
               para_before=para_p(orig, f),
               circuit=[cname(c) for c in circuit],
               circuit_effect_share=sum(eff_orig[c] for c in circuit) / (ld_clean - ld_corr),
               stages={})
    unl = copy.deepcopy(orig)
    opt = torch.optim.Adam(unl.parameters(), lr=args.lr)
    rng, ref_lp = random.Random("forget " + f["name"]), reference_logp(orig, f)
    light = None
    step = 0
    while step < args.max_steps:
        logits = unl(f["toks"])[:, -1]
        p = F.softmax(logits, -1)[0, f["ans"]].item()
        if light is None and p < args.p_target:
            light = step
            out["stages"]["light"] = evaluate(unl, orig, f, facts, circuit, eff_orig, args.k, step)
        if light is not None and step >= max(1, light) * args.heavy_mult:
            break
        loss = unlearn_loss(unl, logits, f, ref_lp, facts, args, rng)
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
    if "light" not in out["stages"]:
        out["note"] = "p(answer) never fell below p_target"
        return out
    out["stages"]["heavy"] = evaluate(unl, orig, f, facts, circuit, eff_orig, args.k, step)
    out["stages"]["control"] = control_stage(orig, facts, f, other, circuit, eff_orig, args)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["toy", "gpt2"], default="toy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forget", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lr", type=float, default=None,
                    help="default 1e-3 for toy, 3e-6 for gpt2 (tune it: see checklist Phase 3)")
    ap.add_argument("--toy-paraphrase", action="store_true",
                    help="toy only: also train a second 'paraphrase' relation token")
    ap.add_argument("--dataset", choices=["counterfact", "capitals"], default="counterfact",
                    help="gpt2 only: fact source")
    ap.add_argument("--cf-path", default="counterfact.json", help="path to CounterFact counterfact.json")
    ap.add_argument("--n-facts", type=int, default=40, help="gpt2: facts to select (forget + retain)")
    ap.add_argument("--min-p", type=float, default=0.1, help="gpt2: minimum p(answer) for a fact to qualify")
    ap.add_argument("--min-margin", type=float, default=1.0, help="gpt2: minimum clean-vs-corrupt logit-diff gap")
    ap.add_argument("--max-per-rel", type=int, default=4, help="gpt2: max facts per CounterFact relation")
    ap.add_argument("--scan", type=int, default=3000, help="gpt2: CounterFact records to scan")
    ap.add_argument("--data-seed", type=int, default=0, help="gpt2: seed for sampling candidate facts")
    ap.add_argument("--min-para-p", type=float, default=0.0,
                    help="gpt2: also require this mean p(answer) on the fact's paraphrase prompts (0 = off)")
    ap.add_argument("--method", choices=["ga", "graddiff", "npo"], default="ga",
                    help="unlearning method: ga (gradient ascent), graddiff (ascent + retain loss), npo (NPO + retain loss)")
    ap.add_argument("--beta", type=float, default=0.1, help="npo: inverse temperature")
    ap.add_argument("--retain-weight", type=float, default=1.0, help="graddiff / npo: weight of the retain loss")
    ap.add_argument("--retain-batch", type=int, default=4, help="graddiff / npo: retain facts per step")
    ap.add_argument("--p-target", type=float, default=0.01)
    ap.add_argument("--heavy-mult", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 1e-3 if args.model == "toy" else 3e-6
    torch.manual_seed(args.seed)

    report = None
    if args.model == "toy":
        model, facts = build_toy(args.seed, paraphrase=args.toy_paraphrase)
    else:
        model, facts, report = build_gpt2(args.dataset, args.cf_path, args.n_facts, args.min_p,
                                          args.max_per_rel, args.scan, args.min_margin, args.data_seed, args.min_para_p)
        print("fact selection:", report["stats"])
    if len(facts) < 2:
        raise SystemExit("fewer than 2 usable facts: lower --min-p or --min-margin, or scan more records")
    with torch.no_grad():
        for f in facts:                       # retain accuracy = top-1 agreement with the original model
            f["orig_top1"] = model(f["toks"])[0, -1].argmax().item()
    for j, f in enumerate(facts):             # retain pool for graddiff / npo: every second fact
        f["retain_pool"] = (j % 2 == 1)
    print(f"model={args.model}  facts={len(facts)}  components={len(components(model))}")
    with torch.no_grad():
        acc = sum(model(f["toks"])[0, -1].argmax().item() == f["ans"] for f in facts) / len(facts)
    print(f"original accuracy on facts: {acc:.2f}")

    if args.model == "toy":
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(facts), generator=g)[:args.forget].tolist()
    else:   # gpt2: one fixed shuffle; seed s takes the s-th block, so different seeds never repeat a fact
        order = torch.randperm(len(facts), generator=torch.Generator().manual_seed(args.data_seed)).tolist()
        start = (args.seed * args.forget) % len(order)
        idx = [order[(start + j) % len(order)] for j in range(min(args.forget, len(order)))]
        if (args.seed + 1) * args.forget > len(order):
            print("WARNING: seed*forget exceeds the number of facts; forget sets now repeat facts from earlier seeds")
    off = 13 % len(facts) or 1                     # control partner offset; never the fact itself
    results = [run_fact(model, facts, facts[i], facts[(i + off) % len(facts)], args) for i in idx]
    json.dump(dict(args=vars(args), selection=report["stats"] if report else None, results=results), open(args.out, "w"), indent=1)

    fmt = lambda v: f"{v:>7.2f}" if v is not None else f"{'-':>7}"
    hdr = (f"{'fact':<16}{'stage':<8}{'steps':>5}{'p(ans)':>8}{'retain':>8}{'surv':>7}"
           f"{'overlap':>9}{'reroute':>9}{'cross':>7}{'para':>7}")
    print("\n" + hdr)
    for r in results:
        print(f"{r['fact']:<16}{'orig':<8}{'-':>5}{r['p_before']:>8.3f}{r['retain_before']:>8.2f}"
              f"{1.0:>7.2f}{1.0:>9.2f}{'-':>9}{'-':>7}{fmt(r['para_before'])}   circuit={r['circuit']}")
        for st, e in r["stages"].items():
            print(f"{'':<16}{st:<8}{e['steps']:>5}{e['p_answer']:>8.3f}{e['retain_acc']:>8.2f}"
                  f"{e['survival']:>7.2f}{e['overlap']:>9.2f}{e['reroute']:>9.2f}"
                  f"{e['cross_restore']:>7.2f}{fmt(e['para_p_answer'])}")


if __name__ == "__main__":
    main()
