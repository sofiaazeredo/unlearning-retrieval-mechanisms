"""
compare_methods.py -- gradient ascent against proper unlearning methods (graddiff, npo), same facts, same learning rate.

Each method is one run of pipeline.py --method <name>. Gradient ascent only pushes p(answer) down; graddiff adds a
retain loss; NPO (Zhang et al., 2024) replaces the unbounded ascent loss with a bounded one and also keeps the retain loss.
The question is whether "the circuit survives suppression" is a property of gradient ascent or of unlearning in general,
once collateral damage is controlled.

  python src/compare_methods.py ga=results/lr3e-06 graddiff=results/methods/graddiff/lr3e-06 npo=results/methods/npo/lr3e-06 \
         --model gpt2 --out results/methods_report.md

The first name is the reference for the paired comparisons.
"""
import argparse
import glob
import json
import os

import numpy as np

import locate


def load(d, model, stage="light"):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, f"results_{model}_seed*.json"))):
        for r in json.load(open(p))["results"]:
            if stage in r.get("stages", {}) and "control" in r["stages"]:
                out[r["fact"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=folder pairs; the first is the reference")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", default="methods_report.md")
    ap.add_argument("--stage", choices=["light", "heavy"], default="light",
                    help="light = first step below p_target; heavy = that many times more steps (the run's --heavy-mult)")
    ap.add_argument("--title", default=None, help="heading, e.g. the strength these runs share")
    args = ap.parse_args()
    runs = {}
    for a in args.runs:
        name, d = a.split("=", 1)
        runs[name] = load(d, args.model, args.stage)
    names = list(runs)
    dirs = dict(a.split("=", 1) for a in args.runs)
    for n in names:
        print(f"{n}: {len(runs[n])} facts with a {args.stage}-stage result in {dirs[n]}" + ("" if os.path.isdir(dirs[n]) else "   (FOLDER DOES NOT EXIST)"))
    common = [f for f in runs[names[0]] if all(f in runs[n] for n in names)]
    if not common:
        empty = [n for n in names if not runs[n]]
        if empty:
            raise SystemExit(f"no facts in common: no results found for {', '.join(empty)} (check that stage ran and that its log_seed*.txt has no error)")
        raise SystemExit("no facts in common: the runs were done on different facts (different seeds, forget size, n_facts or data settings)")
    rels = [runs[names[0]][f].get("rel") for f in common]
    L = lambda n, f: runs[n][f]["stages"][args.stage]
    C = lambda n, f: runs[n][f]["stages"]["control"]

    def med(n, get, stage=L):
        a = [get(stage(n, f)) for f in common if get(stage(n, f)) is not None and not np.isnan(get(stage(n, f)))]
        return float(np.median(a)) if a else float("nan")

    md = [f"# {args.title or 'Unlearning methods'}, {args.stage} stage ({len(common)} facts in common)\n",
          "| method | steps | p(answer) after | survival [95% CI] | control survival | top-5 overlap | reroute | retain agreement | retain agreement (held out) | paraphrase p |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        s = locate.boot_median([L(n, f)["survival"] for f in common], rels)
        md.append(f"| {n} | {med(n, lambda e: e['steps']):.0f} | {med(n, lambda e: e['p_answer']):.4f} | {locate.fmt(s)} | "
                  f"{med(n, lambda e: e['survival'], C):.2f} | {med(n, lambda e: e['overlap']):.2f} | {med(n, lambda e: e['reroute']):.2f} | "
                  f"{med(n, lambda e: e['retain_acc']):.2f} | {med(n, lambda e: e.get('retain_heldout')):.2f} | "
                  f"{med(n, lambda e: e['para_p_answer']):.3f} |")
    md.append("\nPaired differences in survival against " + names[0] + ":\n")
    md.append("| method | survival difference [95% CI] | facts with lower survival than the reference |\n|---|---|---|")
    for n in names[1:]:
        d = [L(n, f)["survival"] - L(names[0], f)["survival"] for f in common]
        md.append(f"| {n} | {locate.fmt(locate.boot_median(d, rels))} | {sum(x < 0 for x in d)} of {len(d)} |")
    md.append("\nHeld-out retain agreement counts only facts that the retain loss never trains on (every second fact is the retain pool); "
              "for gradient ascent it is the same quantity, so the columns are comparable. If a method keeps retain agreement high but "
              "its survival is as high as gradient ascent's, the circuit persistence is not an artifact of collateral damage.\n")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
