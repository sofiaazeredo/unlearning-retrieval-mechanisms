"""
locate.py -- WHERE does the suppression live after unlearning?

After light gradient-ascent unlearning, p(answer) drops about 20x while the original circuit keeps roughly
70% of its contribution. So the suppression must sit somewhere the circuit measurement does not see.
This script looks for it with restoration tests, on the unlearned model and the clean prompt:

  Weight restore    copy the ORIGINAL parameters of one part (embeddings, one layer's attention or MLP,
                    ln_final, the unembedding, or only the answer token's unembedding column) back into
                    the unlearned model. If restoring one part brings the answer back, the change that
                    matters lives there. Also cumulative: all blocks >= L, and embeddings + blocks < L.
  Activation restore  paste the ORIGINAL model's activations into the unlearned run: the residual stream
                    before layer L (a sweep: the answer returns once L is past the place where the models
                    diverge), single components, the circuit, random non-circuit components, whole layers.
  Reverse (necessity)  paste the UNLEARNED activations into the ORIGINAL model: how much of the answer is lost.

Score: recovery of the answer's log-probability
    rec = (logp_after_restore - logp_unlearned) / (logp_original - logp_unlearned)
0 = still suppressed, 1 = fully back (second value in each pair: the same for the logit difference).
Reverse tests report necessity = (logp_original - logp_with_unlearned_part) / (logp_original - logp_unlearned).

  python locate.py run    --model gpt2 --lr 3e-6 --seed 0 --forget 16 --out locate/lr3e-06/locate_gpt2_seed0.json
  python locate.py report locate/lr3e-06 --model gpt2
"""
import argparse
import contextlib
import copy
import glob
import json
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

import pipeline as P


# ----------------------------------------------------------------------------- scoring
def score(logits, f):
    lp = F.log_softmax(logits[0, -1].float(), -1)
    return lp[f["ans"]].item(), (logits[0, -1, f["ans"]] - logits[0, -1, f["cans"]]).item()


@torch.no_grad()
def run(model, f, hooks=None):
    return score(model.run_with_hooks(f["toks"], fwd_hooks=hooks or []), f)


class Ref:
    """Original and unlearned scores for one fact; turns a score into a recovery fraction."""

    def __init__(self, orig, unl, f):
        self.lp_o, self.ld_o = run(orig, f)
        self.lp_u, self.ld_u = run(unl, f)
        self.p_o, self.p_u = float(np.exp(self.lp_o)), float(np.exp(self.lp_u))

    def ok(self):
        return self.lp_o - self.lp_u > 0.5 and abs(self.ld_o - self.ld_u) > 1e-3

    def rec(self, s):                       # unlearned model with something restored
        return [round((s[0] - self.lp_u) / (self.lp_o - self.lp_u), 4),
                round((s[1] - self.ld_u) / (self.ld_o - self.ld_u), 4)]

    def nec(self, s):                       # original model with something unlearned
        return [round((self.lp_o - s[0]) / (self.lp_o - self.lp_u), 4),
                round((self.ld_o - s[1]) / (self.ld_o - self.ld_u), 4)]


# ----------------------------------------------------------------------------- weights
def weight_groups(model):
    n = model.cfg.n_layers
    names = [k for k, _ in model.named_parameters()]

    def sel(prefixes):
        return [k for k in names if any(k.startswith(p) for p in prefixes)]

    emb = ["embed.", "pos_embed."]
    g = {"embed": sel(emb), "ln_final": sel(["ln_final."]), "unembed": sel(["unembed."]),
         "readout": sel(["ln_final.", "unembed."])}
    for l in range(n):
        g[f"attn{l}"] = sel([f"blocks.{l}.attn.", f"blocks.{l}.ln1."])
        g[f"mlp{l}"] = sel([f"blocks.{l}.mlp.", f"blocks.{l}.ln2."])
        g[f"top{l}"] = sel([f"blocks.{j}." for j in range(l, n)])               # blocks >= l
        g[f"bottom{l + 1}"] = sel(emb + [f"blocks.{j}." for j in range(l + 1)])  # embeddings + blocks <= l
    return {k: v for k, v in g.items() if v}


@contextlib.contextmanager
def restored(up, op, names, col=None):
    """Temporarily copy ORIGINAL parameters into the unlearned model (exactly undone afterwards)."""
    if col is not None:                                    # only the answer token's unembedding column
        W, b = up["unembed.W_U"].data, up["unembed.b_U"].data
        sw, sb = W[:, col].clone(), b[col].clone()
        W[:, col] = op["unembed.W_U"].data[:, col]
        b[col] = op["unembed.b_U"].data[col]
        try:
            yield
        finally:
            W[:, col] = sw
            b[col] = sb
        return
    saved = {k: up[k].data.clone() for k in names}
    for k in names:
        up[k].data.copy_(op[k].data)
    try:
        yield
    finally:
        for k in names:
            up[k].data.copy_(saved[k])


# ----------------------------------------------------------------------------- activations
def resid_hook(L, n):
    return f"blocks.{L}.hook_resid_pre" if L < n else f"blocks.{n - 1}.hook_resid_post"


def paste_resid(name, src, last_only):
    def fn(act, hook):
        if last_only:
            act = act.clone()
            act[:, -1] = src[name][:, -1]
            return act
        return src[name]
    return (name, fn)


def paste_comps(comps, src):
    return [(P.hook_name(c), partial(P._paste, src=src[P.hook_name(c)], head=c[2])) for c in comps]


# ----------------------------------------------------------------------------- one stage
@torch.no_grad()
def locate_stage(orig, unl, f, circuit, orig_cache, args, rng):
    ref = Ref(orig, unl, f)
    out = dict(p_orig=ref.p_o, p_unl=ref.p_u, ok=ref.ok())
    if not out["ok"]:
        return out
    n = orig.cfg.n_layers
    up, op = dict(unl.named_parameters()), dict(orig.named_parameters())
    _, unl_cache = unl.run_with_cache(f["toks"])

    # --- weight restore
    w = {}
    for name, ks in weight_groups(unl).items():
        with restored(up, op, ks):
            w[name] = ref.rec(run(unl, f))
    with restored(up, op, [], col=f["ans"]):
        w["unembed_answer_column"] = ref.rec(run(unl, f))
    out["weights"] = w

    # --- residual-stream sweeps
    out["resid_fwd"] = [ref.rec(run(unl, f, [paste_resid(resid_hook(L, n), orig_cache, False)])) for L in range(n + 1)]
    out["resid_fwd_last"] = [ref.rec(run(unl, f, [paste_resid(resid_hook(L, n), orig_cache, True)])) for L in range(n + 1)]
    out["resid_rev"] = [ref.nec(run(orig, f, [paste_resid(resid_hook(L, n), unl_cache, False)])) for L in range(n + 1)]

    # --- components, groups
    comps = P.components(orig)
    out["comp"] = {P.cname(c): ref.rec(run(unl, f, paste_comps([c], orig_cache))) for c in comps}
    groups = {"circuit": list(circuit)}
    non = [c for c in comps if c not in circuit]
    groups["layers"] = {}
    for l in range(n):
        groups["layers"][f"attn{l}"] = [c for c in comps if c[1] == l and c[0] == "head"]
        groups["layers"][f"mlp{l}"] = [c for c in comps if c[1] == l and c[0] == "mlp"]
    out["group_fwd"] = {"circuit": ref.rec(run(unl, f, paste_comps(circuit, orig_cache))),
                        "all_noncircuit": ref.rec(run(unl, f, paste_comps(non, orig_cache)))}
    draws = [ref.rec(run(unl, f, paste_comps(rng.sample(non, len(circuit)), orig_cache))) for _ in range(args.random_draws)]
    out["group_fwd"]["random_noncircuit"] = [round(float(np.mean([d[i] for d in draws])), 4) for i in range(2)]
    out["group_rev"] = {"circuit": ref.nec(run(orig, f, paste_comps(circuit, unl_cache))),
                        "all_noncircuit": ref.nec(run(orig, f, paste_comps(non, unl_cache)))}
    out["layer_fwd"] = {k: ref.rec(run(unl, f, paste_comps(v, orig_cache))) for k, v in groups["layers"].items() if v}
    out["layer_rev"] = {k: ref.nec(run(orig, f, paste_comps(v, unl_cache))) for k, v in groups["layers"].items() if v}
    return out


def run_fact_locate(orig, facts, f, args, rng):
    eff, _, _ = P.patch_effects(orig, f)
    circuit = P.topk(eff, args.k)
    with torch.no_grad():
        _, orig_cache = orig.run_with_cache(f["toks"])
    res = dict(fact=f["name"], rel=f.get("rel"), circuit=[P.cname(c) for c in circuit], stages={})
    unl = copy.deepcopy(orig)
    opt = torch.optim.Adam(unl.parameters(), lr=args.lr)
    light, step = None, 0
    while step < args.max_steps:
        logits = unl(f["toks"])[:, -1]
        p = F.softmax(logits, -1)[0, f["ans"]].item()
        if light is None and p < args.p_target:
            light = step
            r = locate_stage(orig, unl, f, circuit, orig_cache, args, rng)
            r["steps"] = step
            res["stages"]["light"] = r
            if "heavy" not in args.stages:
                break
        if light is not None and step >= max(1, light) * args.heavy_mult:
            break
        loss = -F.cross_entropy(logits, torch.tensor([f["ans"]], device=logits.device))
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
    if "light" not in res["stages"]:
        res["note"] = "p(answer) never fell below p_target"
        return res
    if "heavy" in args.stages:
        r = locate_stage(orig, unl, f, circuit, orig_cache, args, rng)
        r["steps"] = step
        res["stages"]["heavy"] = r
    return res


def cmd_run(args):
    torch.manual_seed(args.seed)
    if args.model == "toy":
        model, facts = P.build_toy(args.seed, paraphrase=False)
        report = None
    else:
        model, facts, report = P.build_gpt2(args.dataset, args.cf_path, args.n_facts, args.min_p,
                                            args.max_per_rel, args.scan, args.min_margin, args.data_seed, args.min_para_p)
        print("fact selection:", report["stats"])
    with torch.no_grad():
        for f in facts:
            f["orig_top1"] = model(f["toks"])[0, -1].argmax().item()
    # same forget sets as pipeline.py, so the facts match the main results
    if args.model == "toy":
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(facts), generator=g)[:args.forget].tolist()
    else:
        order = torch.randperm(len(facts), generator=torch.Generator().manual_seed(args.data_seed)).tolist()
        start = (args.seed * args.forget) % len(order)
        idx = [order[(start + j) % len(order)] for j in range(min(args.forget, len(order)))]
    rng = random.Random(args.seed)
    results = []
    for i, k in enumerate(idx):
        results.append(run_fact_locate(model, facts, facts[k], args, rng))
        print(f"[{i + 1}/{len(idx)}] {facts[k]['name']}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(args=vars(args), results=results), open(args.out, "w"))


# ----------------------------------------------------------------------------- report
def load(root, model, stage):
    rows, seen = [], set()
    for p in sorted(glob.glob(os.path.join(root, f"locate_{model}_seed*.json"))):
        for r in json.load(open(p))["results"]:
            s = r.get("stages", {}).get(stage)
            if s and s.get("ok") and r["fact"] not in seen:
                seen.add(r["fact"])
                rows.append((r["fact"], r.get("rel"), s))
    return rows


def boot_median(vals, rels, B=2000, seed=0):
    v = np.array(vals, float)
    keep = ~np.isnan(v)
    v, rels = v[keep], [r for r, k in zip(rels, keep) if k]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(v))
    groups = {}
    for x, r in zip(v, rels):
        groups.setdefault(r, []).append(x)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    ms = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys))
        s = np.concatenate([groups[keys[i]] for i in pick])
        ms.append(np.median(s))
    return med, float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def fmt(t):
    return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_md = [f"# Where does the suppression live? ({args.stage} unlearning)\n"]
    rows = load(args.root, args.model, args.stage)
    if not rows:
        raise SystemExit("no usable locate results found")
    rels = [r for _, r, _ in rows]
    pu = [s["p_unl"] for _, _, s in rows]
    po = [s["p_orig"] for _, _, s in rows]
    out_md.append(f"{len(rows)} facts. Median p(answer): original {np.median(po):.3f}, unlearned {np.median(pu):.4f}. "
                  "Numbers are recovery of the answer's log-probability (0 = still suppressed, 1 = fully back), "
                  "median [95% bootstrap CI, resampling by relation]. Reverse tests are 'necessity' (1 = answer fully lost).\n")

    def agg(getter):
        return boot_median([getter(s) for _, _, s in rows], rels)

    def has(getter):
        try:
            getter(rows[0][2])
            return True
        except (KeyError, IndexError, TypeError):
            return False

    n = len(rows[0][2]["resid_fwd"]) - 1
    # --- weights
    out_md.append("## 1. Weight restore (original parameters copied back into the unlearned model)\n")
    out_md.append("| restored part | recovery |\n|---|---|")
    single = ["unembed_answer_column", "unembed", "ln_final", "readout", "embed"]
    W = {}
    for k in single:
        if has(lambda s: s["weights"][k]):
            W[k] = agg(lambda s, k=k: s["weights"][k][0])
            out_md.append(f"| {k} | {fmt(W[k])} |")
    out_md.append("\n| layer | attention only | MLP only | blocks >= layer (top-down) | embeddings + blocks <= layer (bottom-up) |\n|---|---|---|---|---|")
    WL = {"attn": [], "mlp": [], "top": [], "bottom": []}
    for l in range(n):
        row = []
        for key in ["attn", "mlp", "top", "bottom"]:
            name = f"{key}{l}" if key != "bottom" else f"bottom{l + 1}"
            t = agg(lambda s, name=name: s["weights"][name][0]) if has(lambda s: s["weights"][name]) else (float("nan"),) * 3
            WL[key].append(t)
            row.append(fmt(t))
        out_md.append(f"| {l} | " + " | ".join(row) + " |")
    # --- residual
    out_md.append("\n## 2. Residual-stream sweep (before layer L; L = %d is after the last layer)\n" % n)
    out_md.append("| L | original resid pasted into unlearned (all positions) | same, last position only | unlearned resid pasted into original (necessity) |\n|---|---|---|---|")
    R = {"fwd": [], "last": [], "rev": []}
    for L in range(n + 1):
        a = agg(lambda s, L=L: s["resid_fwd"][L][0]); b = agg(lambda s, L=L: s["resid_fwd_last"][L][0])
        c = agg(lambda s, L=L: s["resid_rev"][L][0])
        R["fwd"].append(a); R["last"].append(b); R["rev"].append(c)
        out_md.append(f"| {L} | {fmt(a)} | {fmt(b)} | {fmt(c)} |")
    # --- groups
    out_md.append("\n## 3. Activation restore: circuit vs everything else\n")
    out_md.append("| components | restore original into unlearned | necessity (unlearned into original) |\n|---|---|---|")
    for k in ["circuit", "random_noncircuit", "all_noncircuit"]:
        a = agg(lambda s, k=k: s["group_fwd"][k][0])
        b = agg(lambda s, k=k: s["group_rev"][k][0]) if k != "random_noncircuit" else None
        out_md.append(f"| {k} | {fmt(a)} | {fmt(b) if b else '-'} |")
    out_md.append("\n(random_noncircuit = same number of components as the circuit, drawn at random from outside it.)\n")
    out_md.append("| layer | attention heads restored | MLP restored | attention necessity | MLP necessity |\n|---|---|---|---|---|")
    LA = {"attn": [], "mlp": [], "attn_rev": [], "mlp_rev": []}
    for l in range(n):
        vals = {}
        for key, fld, kk in [("attn", "layer_fwd", f"attn{l}"), ("mlp", "layer_fwd", f"mlp{l}"),
                             ("attn_rev", "layer_rev", f"attn{l}"), ("mlp_rev", "layer_rev", f"mlp{l}")]:
            vals[key] = agg(lambda s, fld=fld, kk=kk: s[fld][kk][0])
            LA[key].append(vals[key])
        out_md.append(f"| {l} | {fmt(vals['attn'])} | {fmt(vals['mlp'])} | {fmt(vals['attn_rev'])} | {fmt(vals['mlp_rev'])} |")
    # --- single components
    names = list(rows[0][2]["comp"])
    med = {k: float(np.median([s["comp"][k][0] for _, _, s in rows])) for k in names}
    top = sorted(med.items(), key=lambda kv: -kv[1])[:12]
    out_md.append("\n## 4. Single components with the largest median recovery\n")
    out_md.append("| component | median recovery | facts where it recovers > 0.2 |\n|---|---|---|")
    for k, v in top:
        share = np.mean([s["comp"][k][0] > 0.2 for _, _, s in rows])
        out_md.append(f"| {k} | {v:.2f} | {share:.0%} |")
    best = np.array([max(s["comp"][k][0] for k in names) for _, _, s in rows])
    out_md.append(f"\nBest single component per fact: median recovery {np.median(best):.2f}.\n")
    open(os.path.join(args.root, "locate_report.md" if args.stage == "light" else f"locate_report_{args.stage}.md"), "w").write("\n".join(out_md))

    # --- figure
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    xs = np.arange(n)
    ax[0, 0].bar(xs - 0.2, [t[0] for t in WL["attn"]], 0.4, label="attention", color="#4c78a8")
    ax[0, 0].bar(xs + 0.2, [t[0] for t in WL["mlp"]], 0.4, label="MLP", color="#f58518")
    for i, k in enumerate(["unembed", "ln_final", "embed"]):
        if k in W:
            ax[0, 0].axhline(W[k][0], ls=":", color=["#e45756", "#72b7b2", "#54a24b"][i], label=f"{k} (all)")
    if "unembed_answer_column" in W:
        ax[0, 0].axhline(W["unembed_answer_column"][0], ls="--", color="#b279a2", label="unembed: answer column only")
    ax[0, 0].set(title="Weight restore: one part at a time", xlabel="layer", ylabel="recovery of log p(answer)")
    ax[0, 0].legend(fontsize=7)
    ax[0, 1].plot(xs, [t[0] for t in WL["top"]], "o-", label="restore blocks >= L", color="#4c78a8")
    ax[0, 1].plot(xs + 1, [t[0] for t in WL["bottom"]], "s-", label="restore embeddings + blocks < L", color="#f58518")
    ax[0, 1].set(title="Weight restore: cumulative", xlabel="L", ylabel="recovery")
    ax[0, 1].legend(fontsize=8)
    xr = np.arange(n + 1)
    for key, lab, col in [("fwd", "original resid into unlearned", "#4c78a8"), ("last", "same, last position only", "#72b7b2"),
                          ("rev", "unlearned resid into original (necessity)", "#e45756")]:
        m = np.array([t[0] for t in R[key]]); lo = np.array([t[1] for t in R[key]]); hi = np.array([t[2] for t in R[key]])
        ax[1, 0].plot(xr, m, "o-", label=lab, color=col); ax[1, 0].fill_between(xr, lo, hi, alpha=0.15, color=col)
    ax[1, 0].set(title="Residual stream before layer L", xlabel="L", ylabel="recovery / necessity")
    ax[1, 0].legend(fontsize=7)
    ax[1, 1].bar(xs - 0.2, [t[0] for t in LA["attn"]], 0.4, label="attention heads restored", color="#4c78a8")
    ax[1, 1].bar(xs + 0.2, [t[0] for t in LA["mlp"]], 0.4, label="MLP restored", color="#f58518")
    ax[1, 1].set(title="Activation restore, whole layer", xlabel="layer", ylabel="recovery")
    ax[1, 1].legend(fontsize=8)
    for a in ax.flat:
        a.axhline(0, color="k", lw=0.5); a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Where does the suppression live? ({len(rows)} facts, {args.stage} unlearning)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.root, f"fig_locate_{args.stage}.png"), dpi=150)
    print("\n".join(out_md))


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", choices=["toy", "gpt2"], default="toy")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--forget", type=int, default=5)
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--lr", type=float, default=None)
    r.add_argument("--stages", default="light", help="light or light,heavy")
    r.add_argument("--random-draws", type=int, default=10)
    r.add_argument("--dataset", default="counterfact")
    r.add_argument("--cf-path", default="counterfact.json")
    r.add_argument("--n-facts", type=int, default=40)
    r.add_argument("--min-p", type=float, default=0.1)
    r.add_argument("--min-margin", type=float, default=1.0)
    r.add_argument("--min-para-p", type=float, default=0.0)
    r.add_argument("--max-per-rel", type=int, default=4)
    r.add_argument("--scan", type=int, default=3000)
    r.add_argument("--data-seed", type=int, default=0)
    r.add_argument("--p-target", type=float, default=0.01)
    r.add_argument("--heavy-mult", type=int, default=5)
    r.add_argument("--max-steps", type=int, default=300)
    r.add_argument("--out", default="locate.json")
    q = sub.add_parser("report")
    q.add_argument("root")
    q.add_argument("--model", default="gpt2")
    q.add_argument("--stage", default="light")
    args = ap.parse_args()
    if args.cmd == "run":
        if args.lr is None:
            args.lr = 1e-3 if args.model == "toy" else 3e-6
        args.stages = args.stages.split(",")
        cmd_run(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
