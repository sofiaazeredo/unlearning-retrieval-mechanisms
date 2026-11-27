"""
swap.py -- compare gradient-ascent unlearning with "answer swap" suppression.

Answer swap = fine-tune the model to give the PARTNER's answer (the other fact of the same relation that every
fact is paired with) until p(original answer) < p_target. It suppresses the same fact by installing a replacement,
so it shows which findings are specific to gradient ascent and which are generic to making a model stop saying a fact.

  python src/swap.py locate   ...   restoration tests (same as locate.py) after answer swap
  python src/swap.py survival ...   circuit survival / overlap / reroute after answer swap, plus a dose-matched
                                    swap (same number of steps as gradient ascent needed on that fact)
  python src/swap.py compare  ...   side-by-side tables, gradient ascent vs answer swap

Forget sets are chosen exactly as in pipeline.py and locate.py, so all methods see the same facts.
"""
import argparse
import copy
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

import locate
import pipeline as P


# ----------------------------------------------------------------------------- shared setup
def add_common(ap):
    ap.add_argument("--model", choices=["toy", "gpt2"], default="toy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forget", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dataset", default="counterfact")
    ap.add_argument("--cf-path", default="data/counterfact.json")
    ap.add_argument("--n-facts", type=int, default=40)
    ap.add_argument("--min-p", type=float, default=0.1)
    ap.add_argument("--min-margin", type=float, default=1.0)
    ap.add_argument("--max-per-rel", type=int, default=4)
    ap.add_argument("--scan", type=int, default=3000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--min-para-p", type=float, default=0.0)
    ap.add_argument("--p-target", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", required=True)


def setup(args):
    torch.manual_seed(args.seed)
    if args.model == "toy":
        model, facts = P.build_toy(args.seed, paraphrase=False)
    else:
        model, facts, rep = P.build_gpt2(args.dataset, args.cf_path, args.n_facts, args.min_p,
                                         args.max_per_rel, args.scan, args.min_margin, args.data_seed, args.min_para_p)
        print("fact selection:", rep["stats"])
    with torch.no_grad():
        for f in facts:
            f["orig_top1"] = model(f["toks"])[0, -1].argmax().item()   # for retain agreement
    for j, f in enumerate(facts):
        f["retain_pool"] = (j % 2 == 1)                                # same retain pool as pipeline.py
    if args.model == "toy":
        idx = torch.randperm(len(facts), generator=torch.Generator().manual_seed(args.seed))[:args.forget].tolist()
    else:
        order = torch.randperm(len(facts), generator=torch.Generator().manual_seed(args.data_seed)).tolist()
        start = (args.seed * args.forget) % len(order)
        idx = [order[(start + j) % len(order)] for j in range(min(args.forget, len(order)))]
    return model, facts, idx


def swap_train(model, f, lr, max_steps, stop_p=None, n_steps=None):
    """Fine-tune a copy toward the partner's answer, until p(answer) < stop_p or for exactly n_steps."""
    unl = copy.deepcopy(model)
    opt = torch.optim.Adam(unl.parameters(), lr=lr)
    steps = 0
    while steps < max_steps:
        logits = unl(f["toks"])[:, -1]
        if n_steps is None and F.softmax(logits, -1)[0, f["ans"]].item() < stop_p:
            break
        if n_steps is not None and steps >= n_steps:
            break
        loss = F.cross_entropy(logits, torch.tensor([f["cans"]], device=logits.device))   # descent toward the OTHER answer
        opt.zero_grad(); loss.backward(); opt.step()
        steps += 1
    return unl, steps


def load_ga(ga_dir, model):
    """Gradient-ascent results of the main run: per fact, steps and metrics at the light stage."""
    ga = {}
    for p in sorted(glob.glob(os.path.join(ga_dir, f"results_{model}_seed*.json"))):
        for r in json.load(open(p))["results"]:
            if "light" in r["stages"]:
                L = r["stages"]["light"]
                ga[r["fact"]] = dict(rel=r.get("rel"), steps=L["steps"], survival=L["survival"], overlap=L["overlap"],
                                     reroute=L["reroute"], retain_acc=L["retain_acc"],
                                     control=r["stages"]["control"]["survival"])
    return ga


# ----------------------------------------------------------------------------- locate after swap
def cmd_locate(args):
    model, facts, idx = setup(args)
    A = argparse.Namespace(random_draws=10)
    rng = random.Random(args.seed)
    results = []
    for n, i in enumerate(idx, 1):
        f = facts[i]
        eff, _, _ = P.patch_effects(model, f)
        circuit = P.topk(eff, args.k)
        with torch.no_grad():
            _, orig_cache = model.run_with_cache(f["toks"])
        unl, steps = swap_train(model, f, args.lr, args.max_steps, stop_p=args.p_target)
        res = dict(fact=f["name"], rel=f.get("rel"), circuit=[P.cname(c) for c in circuit], stages={})
        if steps < args.max_steps:
            r = locate.locate_stage(model, unl, f, circuit, orig_cache, A, rng)
            r["steps"] = steps
            res["stages"]["light"] = r
        results.append(res)
        del unl
        print(f"[{n}/{len(idx)}] {f['name']}  ({steps} steps)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(args=vars(args), results=results), open(args.out, "w"))


# ----------------------------------------------------------------------------- survival after swap
def cmd_survival(args):
    model, facts, idx = setup(args)
    ga = load_ga(args.ga_results, args.model)
    results = []
    for n, i in enumerate(idx, 1):
        f = facts[i]
        eff, _, _ = P.patch_effects(model, f)
        circuit = P.topk(eff, args.k)
        row = dict(fact=f["name"], rel=f.get("rel"), p_before=P.p_answer(model, f), stages={})
        unl, steps = swap_train(model, f, args.lr, args.max_steps, stop_p=args.p_target)
        if steps < args.max_steps:
            row["stages"]["swap"] = P.evaluate(unl, model, f, facts, circuit, eff, args.k, steps)
        del unl
        if f["name"] in ga:                       # dose-matched: exactly as many steps as gradient ascent used
            unl, steps = swap_train(model, f, args.lr, args.max_steps, n_steps=max(1, ga[f["name"]]["steps"]))
            row["stages"]["swap_ga_dose"] = P.evaluate(unl, model, f, facts, circuit, eff, args.k, steps)
            del unl
        results.append(row)
        print(f"[{n}/{len(idx)}] {f['name']}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(args=vars(args), results=results), open(args.out, "w"))


# ----------------------------------------------------------------------------- comparison tables
def cmd_compare(args):
    lines = ["# Gradient ascent vs answer swap\n"]
    # --- restoration tests
    ga_l = locate.load(args.ga_locate, args.model, "light")
    sw_l = locate.load(args.swap_locate, args.model, "light")
    if ga_l and sw_l:
        n = len(ga_l[0][2]["resid_fwd"]) - 1
        half = n // 2

        def ci(rows, getter):
            return locate.fmt(locate.boot_median([getter(s) for _, _, s in rows], [r for _, r, _ in rows]))
        tests = [("steps to p(answer) < target (median)", None),
                 ("unembedding answer column restored", lambda s: s["weights"]["unembed_answer_column"][0]),
                 (f"weights: blocks >= {half} restored", lambda s: s["weights"][f"top{half}"][0]),
                 (f"original residual before layer {half}, all positions", lambda s: s["resid_fwd"][half][0]),
                 (f"original residual before layer {half}, last position", lambda s: s["resid_fwd_last"][half][0]),
                 ("circuit components restored", lambda s: s["group_fwd"]["circuit"][0]),
                 ("random non-circuit components restored", lambda s: s["group_fwd"]["random_noncircuit"][0]),
                 ("best single component per fact", lambda s: max(v[0] for v in s["comp"].values()))]
        lines += [f"## Restoration tests (recovery of log p(answer); {len(ga_l)} / {len(sw_l)} facts)\n",
                  "| test | gradient ascent | answer swap |", "|---|---|---|"]
        for name, g in tests:
            if g is None:
                lines.append(f"| {name} | {np.median([s['steps'] for _, _, s in ga_l]):.0f} | {np.median([s['steps'] for _, _, s in sw_l]):.0f} |")
            else:
                lines.append(f"| {name} | {ci(ga_l, g)} | {ci(sw_l, g)} |")
        lines.append("")
    # --- survival
    ga = load_ga(args.ga_results, args.model) if args.ga_results else {}
    sw = {}
    for p in sorted(glob.glob(os.path.join(args.swap_survival, f"swap_{args.model}_seed*.json"))) if args.swap_survival else []:
        for r in json.load(open(p))["results"]:
            if "swap" in r["stages"] and "swap_ga_dose" in r["stages"] and r["fact"] in ga:
                sw[r["fact"]] = r
    ok = list(sw)
    if ok:
        rels = [ga[f]["rel"] for f in ok]

        def ci2(v):
            return locate.fmt(locate.boot_median(v, rels))

        def med(v):
            return f"{np.median(v):.2f}"
        S = lambda f, st, k: sw[f]["stages"][st][k]
        d = [S(f, "swap", "survival") - ga[f]["survival"] for f in ok]
        lines += [f"## Circuit survival ({len(ok)} facts)\n", "| quantity | value |", "|---|---|",
                  f"| survival, gradient ascent (light) | {ci2([ga[f]['survival'] for f in ok])} |",
                  f"| survival, control (other fact unlearned) | {ci2([ga[f]['control'] for f in ok])} |",
                  f"| survival, answer swap (p < target) | {ci2([S(f, 'swap', 'survival') for f in ok])} |",
                  f"| survival, answer swap (same steps as gradient ascent) | {ci2([S(f, 'swap_ga_dose', 'survival') for f in ok])} |",
                  f"| paired difference, swap - gradient ascent | {ci2(d)} |",
                  f"| facts where swap survival < gradient-ascent survival | {sum(x < 0 for x in d)} of {len(d)} |",
                  f"| top-5 overlap, GA / swap | {med([ga[f]['overlap'] for f in ok])} / {med([S(f, 'swap', 'overlap') for f in ok])} |",
                  f"| reroute, GA / swap | {med([ga[f]['reroute'] for f in ok])} / {med([S(f, 'swap', 'reroute') for f in ok])} |",
                  f"| retain agreement, GA / swap / swap (GA steps) | {med([ga[f]['retain_acc'] for f in ok])} / {med([S(f, 'swap', 'retain_acc') for f in ok])} / {med([S(f, 'swap_ga_dose', 'retain_acc') for f in ok])} |",
                  f"| steps, GA / swap | {np.median([ga[f]['steps'] for f in ok]):.0f} / {np.median([S(f, 'swap', 'steps') for f in ok]):.0f} |",
                  f"| p(answer) after swap with GA steps (median) | {np.median([S(f, 'swap_ga_dose', 'p_answer') for f in ok]):.3f} |", ""]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("locate"); add_common(a)
    b = sub.add_parser("survival"); add_common(b)
    b.add_argument("--ga-results", required=True, help="folder with the main run's results_<model>_seed*.json for this learning rate")
    c = sub.add_parser("compare")
    c.add_argument("--model", default="gpt2")
    c.add_argument("--ga-locate", required=True)
    c.add_argument("--swap-locate", required=True)
    c.add_argument("--ga-results", default=None)
    c.add_argument("--swap-survival", default=None)
    c.add_argument("--out", default="swap_comparison.md")
    args = ap.parse_args()
    if args.cmd in ("locate", "survival") and args.lr is None:
        args.lr = 1e-3 if args.model == "toy" else 3e-6
    {"locate": cmd_locate, "survival": cmd_survival, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
