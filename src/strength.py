"""
strength.py -- unlearning strength ladder (proposal: "multiple optimization strengths").

Question: does the original circuit survive near-complete behavioral suppression, or does survival fall only when the
model as a whole is being damaged? Each rung is one run of pipeline.py with a different strength:
  p_target     how far p(answer) must fall for the "light" stage (0.01, 0.001, 0.0001 ...)
  heavy_mult   the "heavy" stage runs this many times the light-stage steps
Every stage of every run becomes one point on the ladder, labelled "p<target, Nx steps". All points use the same facts
and learning rate, so they can be compared directly.

  python src/strength.py results --main-lr 3e-06 --model gpt2
Reads results/lr<lr>/ (the main run) and results/ladder/*/lr<lr>/ (the rungs), writes results/strength_report.md and
results/fig_strength.png.
"""
import argparse
import glob
import json
import os

import numpy as np

import locate


def collect(root, lr, model):
    dirs = [os.path.join(root, f"lr{lr}")] + sorted(glob.glob(os.path.join(root, "ladder", "*", f"lr{lr}")))
    pts = {}                      # label -> {fact -> record}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, f"results_{model}_seed*.json"))):
            J = json.load(open(p))
            pt, hm = J["args"]["p_target"], J["args"]["heavy_mult"]
            for r in J["results"]:
                st = r.get("stages", {})
                if "light" not in st:
                    continue
                ctrl = st.get("control", {}).get("survival")
                for stage, mult in (("light", 1), ("heavy", hm)):
                    if stage in st:
                        label = f"p<{pt:g}, {mult}x steps"
                        pts.setdefault(label, {}).setdefault(r["fact"], dict(rel=r.get("rel"), pt=pt, mult=mult, ctrl=ctrl, **st[stage]))
    return pts


def table(root, lr, model):
    pts = collect(root, lr, model)
    rows = []
    for label, d in pts.items():
        v = list(d.values())
        rel = [x["rel"] for x in v]

        def m(k):
            a = [x[k] for x in v if x.get(k) is not None and not np.isnan(x[k])]
            return float(np.median(a)) if a else float("nan")
        rows.append(dict(label=label, n=len(v), steps=m("steps"), p=m("p_answer"),
                         surv=locate.boot_median([x["survival"] for x in v], rel), ctrl=m("ctrl"),
                         retain=m("retain_acc"), held=m("retain_heldout"), reroute=m("reroute"), para=m("para_p_answer")))
    rows.sort(key=lambda r: (r["steps"], r["p"]))
    return rows


def md_table(rows):
    md = ["| stage | facts | median steps | p(answer) after | survival [CI] | control survival | retain agreement | retain agreement (held out) | reroute | paraphrase p |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['label']} | {r['n']} | {r['steps']:.0f} | {r['p']:.4f} | {locate.fmt(r['surv'])} | {r['ctrl']:.2f} | "
                  f"{r['retain']:.2f} | {r['held']:.2f} | {r['reroute']:.2f} | {r['para']:.3f} |")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results")
    ap.add_argument("--main-lr", required=True, help="folder suffix of the learning rate, e.g. 3e-06")
    ap.add_argument("--also-lrs", nargs="*", default=[], help="further learning rates (folder suffixes) with their own ladders")
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = table(args.root, args.main_lr, args.model)
    if not rows:
        raise SystemExit("no results found")
    md = ["# Unlearning strength ladder\n",
          f"Learning rate {args.main_lr}; medians over facts, survival with 95% CI (resampling relations). "
          "Control survival = the same facts' circuits after unlearning a different fact to the same p_target.\n"]
    md += md_table(rows)
    first, last = rows[0], rows[-1]
    md.append(f"\nFrom {first['label']} to {last['label']}: survival {first['surv'][0]:.2f} -> {last['surv'][0]:.2f}, "
              f"retain agreement {first['retain']:.2f} -> {last['retain']:.2f}, median p(answer) {first['p']:.4f} -> {last['p']:.4f}.")
    md.append("Read the ladder as follows: if survival keeps falling while retain agreement stays near its light-stage value, "
              "the circuit is being removed without general damage; if survival falls only where retain agreement collapses, circuit loss "
              "cannot be separated from model damage.\n")
    others = {}
    for lr in args.also_lrs:
        o = table(args.root, lr, args.model)
        if o:
            others[lr] = o
            md += [f"\n## Learning rate {lr}\n"] + md_table(o)
    if others:
        md.append("\nIf the same pattern (survival falling only together with retain agreement) holds at every learning rate, "
                  "it does not depend on the learning rate chosen.\n")
    os.makedirs(args.root, exist_ok=True)
    open(os.path.join(args.root, "strength_report.md"), "w").write("\n".join(md))
    print("\n".join(md))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    cols = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    series = [(args.main_lr, rows)] + list(others.items())
    for k, (lr, rr) in enumerate(series):
        col = cols[k % len(cols)]
        xs_, ys_ = [r["retain"] for r in rr], [r["surv"][0] for r in rr]
        ax[0].plot(xs_, ys_, "o-", color=col, label=f"lr {lr}")
        ax[0].errorbar(xs_, ys_, yerr=[[r["surv"][0] - r["surv"][1] for r in rr], [r["surv"][2] - r["surv"][0] for r in rr]], fmt="none", ecolor=col, capsize=2, alpha=0.6)
        if k == 0:
            for r in rr:
                ax[0].annotate(r["label"], (r["retain"], r["surv"][0]), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax[0].axhline(np.nanmedian([r["ctrl"] for r in rows]), color="#9aa0a6", ls="--", label="control survival")
    ax[0].set(xlabel="retain agreement (other facts' top-1 unchanged)", ylabel="circuit survival",
              title="Circuit survival vs collateral damage, by strength")
    ax[0].legend(fontsize=8)
    xs = [r["steps"] for r in rows]
    ax[1].plot(xs, [r["surv"][0] for r in rows], "o-", color="#f58518", label="circuit survival")
    ax[1].plot(xs, [r["retain"] for r in rows], "s-", color="#4c78a8", label="retain agreement")
    ax[1].plot(xs, [r["p"] / max(rows[0]["p"], 1e-9) for r in rows], "^-", color="#e45756", label="p(answer) / p(answer) at lightest stage")
    ax[1].set_xscale("log"); ax[1].set(xlabel=f"median unlearning steps (lr {args.main_lr})", title="Dose-response")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.root, "fig_strength.png"), dpi=150)


if __name__ == "__main__":
    main()
