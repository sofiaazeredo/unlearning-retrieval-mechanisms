"""
phase_checks.py -- the Phase 2 and 3 checks of the checklist, read from the saved main-run results.

  python src/phase_checks.py results --main-lr 3e-06 --model gpt2
Writes results/phase_checks.md: how many facts were selected and reached the light stage, whether circuits are concentrated,
whether they contain last-layer components (which makes the cross-model patching test uninformative), the paraphrase baseline,
and the control check.
"""
import argparse
import collections
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--main-lr", required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--last-layer", type=int, default=None, help="index of the last layer (default: 11 for gpt2, 1 for toy)")
    args = ap.parse_args()
    last = args.last_layer if args.last_layer is not None else (11 if args.model == "gpt2" else 1)
    rows, sel, seen = [], None, set()
    files = sorted(glob.glob(os.path.join(args.root, f"lr{args.main_lr}", f"results_{args.model}_seed*.json")))
    if not files:
        raise SystemExit("no main-run results found")
    for p in files:
        J = json.load(open(p))
        sel = sel or J.get("selection")
        for r in J["results"]:
            if r["fact"] not in seen:
                seen.add(r["fact"]); rows.append(r)
    reached = [r for r in rows if "light" in r.get("stages", {})]
    md = [f"# Phase 2 and 3 checks (lr {args.main_lr}, {len(rows)} forget facts)\n"]
    md.append("## Fact selection\n")
    md.append(f"- Selection printout of seed 0: `{sel}`" if sel else "- (no selection record in the results)")
    md.append(f"- Facts that reached p(answer) < target in the light stage: {len(reached)} of {len(rows)}\n")
    md.append("## Circuits\n")
    comp = collections.Counter(c for r in rows for c in r["circuit"])
    n = len(rows)
    last_mlp = sum(f"mlp{last}" in r["circuit"] for r in rows)
    last_any = sum(any(c.endswith(f"mlp{last}") or c.startswith(f"head{last}.") for c in r["circuit"]) for r in rows)
    share = [r["circuit_effect_share"] for r in rows]
    md.append(f"- circuit_effect_share (summed top-k effect / total effect): median {np.median(share):.2f}, range [{min(share):.2f}, {max(share):.2f}]. "
              "A value above 1 means components overlap; the top-k carry the whole measured effect.")
    md.append(f"- Circuits containing the last-layer MLP (mlp{last}): {last_mlp} of {n} ({100 * last_mlp / n:.0f}%); any layer-{last} component: {last_any} of {n}. "
              "Where this is common, `cross_restore` is not interpretable.")
    md.append("- Most frequent components: " + ", ".join(f"{c} ({k})" for c, k in comp.most_common(8)))
    md.append("- The patching effect sizes for individual facts are in `surface/lr*/surface_*.md` (top-10 components and the top-k share).\n")
    md.append("## Surface check and control\n")
    ps = [r["stages"]["light"] for r in reached]
    md.append(f"- p(answer) before: median {np.median([r['p_before'] for r in reached]):.3f}; after light unlearning: median {np.median([s['p_answer'] for s in ps]):.4f}")
    md.append(f"- Top-1 still the true answer after light unlearning: {sum(s['top1_correct'] for s in ps)} of {len(ps)} facts")
    md.append(f"- Retain agreement after light unlearning: median {np.median([s['retain_acc'] for s in ps]):.2f}")
    ctrl = [r["stages"]["control"] for r in reached if "control" in r["stages"]]
    if ctrl:
        md.append(f"- Control (a different fact unlearned): survival median {np.median([c['survival'] for c in ctrl]):.2f}, "
                  f"p(answer) of the original fact stays high in {sum(c['p_answer'] > 0.5 * r['p_before'] for c, r in zip(ctrl, reached))} of {len(ctrl)} facts")
    pb = [r["para_before"] for r in rows if r.get("para_before") is not None]
    pa = [s["para_p_answer"] for s in ps if s.get("para_p_answer") is not None]
    if pb:
        md.append(f"- Paraphrase prompts: p(answer) in the original model median {np.median(pb):.3f}"
                  + (f", after light unlearning {np.median(pa):.3f}" if pa else "") + " (weak baseline if below 0.1).")
    out = os.path.join(args.root, "phase_checks.md")
    open(out, "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
