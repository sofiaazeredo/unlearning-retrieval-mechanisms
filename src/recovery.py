"""
recovery.py -- behavioral recovery: is a "forgotten" fact still reachable under other prompts,
and does that line up with the circuit measurements?   (proposal: Behavioral Recovery section)

For every forget fact, after light unlearning, p(answer) is measured on four prompt families:
  original    the prompt that was unlearned
  paraphrase  CounterFact paraphrase prompts of the same fact
  wrapped     the original prompt inside generic context ("Fact: ...", "Trivia night! ...", a sentence of narrative)
  one-shot    a same-relation example fact (the partner's prompt and answer) placed before the prompt
For each family, suppression ratio = p(answer) after / p(answer) before (1 = untouched, 0 = gone). A fact is only
scored on a family when its baseline p(answer) there is at least --min-base (default 0.02), because a ratio of two tiny
numbers means nothing. A family whose ratio is much higher than the original prompt's ratio means the answer is still
reachable although the original prompt looks forgotten.
The report then asks whether that behavioral leakage lines up with the mechanistic measures (survival, reroute).

  python src/recovery.py run    --model gpt2 --lr 3e-6 --seed 0 --forget 16 --out results/recovery/lr3e-06/recovery_gpt2_seed0.json
  python src/recovery.py report results/recovery/lr3e-06 --model gpt2
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

import locate
import pipeline as P
import swap

WRAPPERS = ["Fact: ", "Trivia night! ", "I was reading an article earlier today. "]
FAMILIES = ["original", "paraphrase", "wrapped", "oneshot"]


def prompt_sets(model, f):
    fam = {"original": [f["toks"]], "paraphrase": list(f.get("para", []))}
    if f.get("prompt"):
        fam["wrapped"] = [model.to_tokens(w + f["prompt"]) for w in WRAPPERS]
        if f.get("partner_prompt"):
            fam["oneshot"] = [model.to_tokens(f'{f["partner_prompt"]} {f["partner_answer"]}. {f["prompt"]}')]
    return {k: v for k, v in fam.items() if v}


@torch.no_grad()
def family_p(model, f, fam):
    return {k: float(np.mean([F.softmax(model(t)[0, -1], -1)[f["ans"]].item() for t in v])) for k, v in fam.items()}


def cmd_run(args):
    model, facts, idx = swap.setup(args)
    results = []
    for n, i in enumerate(idx, 1):
        f = facts[i]
        fam = prompt_sets(model, f)
        before = family_p(model, f, fam)
        eff, _, _ = P.patch_effects(model, f)
        circuit = P.topk(eff, args.k)
        unl = swap.copy.deepcopy(model)
        opt = torch.optim.Adam(unl.parameters(), lr=args.lr)
        rng, ref_lp = random.Random("forget " + f["name"]), P.reference_logp(model, f)
        steps = 0
        while steps < args.max_steps:
            logits = unl(f["toks"])[:, -1]
            if F.softmax(logits, -1)[0, f["ans"]].item() < args.p_target:
                break
            loss = P.unlearn_loss(unl, logits, f, ref_lp, facts, args, rng)
            opt.zero_grad(); loss.backward(); opt.step()
            steps += 1
        row = dict(fact=f["name"], rel=f.get("rel"), steps=steps, p_before=before)
        if steps < args.max_steps:
            ev = P.evaluate(unl, model, f, facts, circuit, eff, args.k, steps)
            row.update(p_after=family_p(unl, f, fam), survival=ev["survival"], reroute=ev["reroute"],
                       overlap=ev["overlap"], retain_acc=ev["retain_acc"])
        results.append(row)
        del unl
        print(f"[{n}/{len(idx)}] {f['name']}  ({steps} steps)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(args=vars(args), results=results), open(args.out, "w"))


# ----------------------------------------------------------------------------- report
def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    rows, seen = [], set()
    for p in sorted(glob.glob(os.path.join(args.root, f"recovery_{args.model}_seed*.json"))):
        for r in json.load(open(p))["results"]:
            if "p_after" in r and r["fact"] not in seen:
                seen.add(r["fact"]); rows.append(r)
    if not rows:
        raise SystemExit("no usable recovery results found")
    rels = [r["rel"] for r in rows]
    md = [f"# Behavioral recovery ({len(rows)} facts)\n",
          f"Suppression ratio = p(answer) after light unlearning / before. Scored only where the baseline is >= {args.min_base}. "
          "Median [95% CI resampling relations]; 1 = untouched, 0 = gone.\n",
          "| prompt family | usable facts | median p before | median p after | suppression ratio | ratio relative to original prompt |",
          "|---|---|---|---|---|---|"]
    ratio = {}
    for k in FAMILIES:
        ok = [r for r in rows if k in r["p_before"] and r["p_before"][k] >= args.min_base]
        ratio[k] = {r["fact"]: r["p_after"][k] / r["p_before"][k] for r in ok}
        if not ok:
            md.append(f"| {k} | 0 | - | - | - | - |"); continue
        rr = [r["rel"] for r in ok]
        base = []
        for r in ok:
            if r["fact"] in ratio.get("original", {}) and ratio["original"][r["fact"]] > 0:
                base.append(ratio[k][r["fact"]] / ratio["original"][r["fact"]])
        rel_txt = locate.fmt(locate.boot_median(base, [r["rel"] for r in ok if r["fact"] in ratio["original"]])) if base and k != "original" else "-"
        md.append(f"| {k} | {len(ok)} | {np.median([r['p_before'][k] for r in ok]):.3f} | {np.median([r['p_after'][k] for r in ok]):.3f} | "
                  f"{locate.fmt(locate.boot_median(list(ratio[k].values()), rr))} | {rel_txt} |")
    md.append("\nA ratio relative to the original prompt above 1 means the answer survives better on that family than on the prompt "
              "that was unlearned (the fact is behaviorally recoverable).\n")
    md.append("## Does behavioral leakage line up with the circuit?\n")
    md.append("| family | facts | Spearman rho (survival vs ratio) | p | median ratio, survival above median | median ratio, survival below median |\n|---|---|---|---|---|---|")
    surv = {r["fact"]: r["survival"] for r in rows}
    for k in FAMILIES:
        fs = [f for f in ratio.get(k, {}) if surv.get(f) is not None and not np.isnan(surv[f])]
        if len(fs) < 8:
            md.append(f"| {k} | {len(fs)} | too few | | | |"); continue
        x, y = [surv[f] for f in fs], [ratio[k][f] for f in fs]
        rho, pv = spearmanr(x, y)
        m = np.median(x)
        hi = [ratio[k][f] for f in fs if surv[f] >= m]
        lo = [ratio[k][f] for f in fs if surv[f] < m]
        md.append(f"| {k} | {len(fs)} | {rho:.2f} | {pv:.3f} | {np.median(hi):.2f} | {np.median(lo):.2f} |")
    md.append("\nA positive rho means facts whose circuit survives also keep more of their answer on that prompt family. "
              "These are correlations across facts that share relations; they do not show that the circuit causes the leakage.\n")
    os.makedirs(args.root, exist_ok=True)
    open(os.path.join(args.root, "recovery_report.md"), "w").write("\n".join(md))
    print("\n".join(md))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    xs = [k for k in FAMILIES if ratio.get(k)]
    meds = [locate.boot_median(list(ratio[k].values()), [next(r["rel"] for r in rows if r["fact"] == f) for f in ratio[k]]) for k in xs]
    ax[0].bar(range(len(xs)), [m[0] for m in meds], yerr=[[m[0] - m[1] for m in meds], [m[2] - m[0] for m in meds]], color="#4c78a8")
    ax[0].set_xticks(range(len(xs))); ax[0].set_xticklabels([f"{k}\n(n={len(ratio[k])})" for k in xs])
    ax[0].axhline(1, color="k", lw=0.5, ls=":")
    ax[0].set(ylabel="p(answer) after / before", title="How much of the answer survives, by prompt family")
    for k, col in [("paraphrase", "#f58518"), ("wrapped", "#54a24b")]:
        fs = [f for f in ratio.get(k, {}) if surv.get(f) is not None and ratio[k][f] > 0]
        if not fs:
            continue
        ax[1].scatter([surv[f] for f in fs], [ratio[k][f] for f in fs], s=14, alpha=0.6, color=col, label=k)
    if ax[1].collections:
        ax[1].set_yscale("log"); ax[1].legend()
    ax[1].set(xlabel="circuit survival (light unlearning)", ylabel="suppression ratio on that family (log)",
              title="Circuit survival vs behavioral leakage")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.root, "fig_recovery.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    swap.add_common(r)
    r.add_argument("--method", choices=["ga", "graddiff", "npo"], default="ga")
    r.add_argument("--beta", type=float, default=0.1)
    r.add_argument("--retain-weight", type=float, default=1.0)
    r.add_argument("--retain-batch", type=int, default=4)
    q = sub.add_parser("report")
    q.add_argument("root")
    q.add_argument("--model", default="gpt2")
    q.add_argument("--min-base", type=float, default=0.02)
    args = ap.parse_args()
    if args.cmd == "run":
        if args.lr is None:
            args.lr = 1e-3 if args.model == "toy" else 3e-6
        cmd_run(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
