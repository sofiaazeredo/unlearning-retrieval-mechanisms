"""Statistics, tables and figures for the unlearning-circuits experiment.

  python analysis.py results --main-lr 3e-6

Expects  results/lr<lr>/results_gpt2_seed<N>.json  (written by the notebook).
Writes into the same folder: analysis_report.md, per_fact_table.csv, lr_table.csv,
threshold_table.csv, relation_table.csv, fig_survival.png, fig_robustness.png.
"""
import argparse
import glob
import json
import math
import os
import re
import statistics as st

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- loading
def load_runs(root, model="gpt2"):
    """{lr_string: (rows, n_failed, n_duplicates)} - rows de-duplicated by fact name."""
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "lr*"))):
        lr = os.path.basename(d)[2:]
        rows, failed, seen, dup = [], 0, set(), 0
        for p in sorted(glob.glob(os.path.join(d, f"results_{model}_seed*.json"))):
            for r in json.load(open(p))["results"]:
                if "light" not in r.get("stages", {}):
                    failed += 1
                    continue
                if r["fact"] in seen:
                    dup += 1
                    continue
                seen.add(r["fact"])
                rows.append(r)
        if rows or failed:
            runs[lr] = (rows, failed, dup)
    return runs


def col(rows, stage, m):
    return np.array([r["stages"][stage][m] if m in r["stages"][stage] and r["stages"][stage][m] is not None
                     else np.nan for r in rows], dtype=float)


# ---------------------------------------------------------------- statistics
def boot_ci(x, groups=None, stat=np.median, n=5000, alpha=0.05):
    """Percentile bootstrap CI. With `groups` (e.g. relation ids) whole groups are resampled,
    because facts of one relation are not independent."""
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    x = x[ok]
    if len(x) < 3:
        return (np.nan, np.nan)
    if groups is not None:
        g = np.asarray(groups)[ok]
        keys = sorted(set(g), key=str)
        if len(keys) >= 4:
            idx = {k: np.where(g == k)[0] for k in keys}
            out = []
            for _ in range(n):
                pick = RNG.integers(0, len(keys), len(keys))
                out.append(stat(np.concatenate([x[idx[keys[i]]] for i in pick])))
            return tuple(np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
    out = [stat(RNG.choice(x, len(x))) for _ in range(n)]
    return tuple(np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def wilcoxon_p(a, b):
    try:
        from scipy.stats import wilcoxon
        d = np.asarray(a) - np.asarray(b)
        d = d[~np.isnan(d)]
        if len(d) < 6 or np.all(d == 0):
            return float("nan")
        return float(wilcoxon(d).pvalue)
    except Exception:
        return float("nan")


def fmt_ci(med, ci):
    return f"{med:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


# ---------------------------------------------------------------- reading
def classify(r, ctrl_ref, hi, lo, rr):
    L = r["stages"]["light"]
    s = L["survival"] / ctrl_ref if ctrl_ref and ctrl_ref > 0.05 else L["survival"]
    if s >= hi:
        return "persistence"
    if (L.get("reroute") or 0) >= rr:
        return "rerouting"
    if s <= lo:
        return "removal"
    return "mixed"


def summarize_lr(rows):
    rel = [r.get("rel") for r in rows] if all(r.get("rel") for r in rows) else None
    d = {}
    for stage in ["light", "heavy", "control"]:
        for m in ["steps", "p_answer", "retain_acc", "survival", "overlap", "reroute", "para_p_answer"]:
            d[(stage, m)] = col(rows, stage, m)
    s_l, s_c = d[("light", "survival")], d[("control", "survival")]
    gap = s_c - s_l
    out = dict(n=len(rows),
               light_surv=np.nanmedian(s_l), light_ci=boot_ci(s_l, rel),
               ctrl_surv=np.nanmedian(s_c), ctrl_ci=boot_ci(s_c, rel),
               heavy_surv=np.nanmedian(d[("heavy", "survival")]), heavy_ci=boot_ci(d[("heavy", "survival")], rel),
               gap=np.nanmedian(gap), gap_ci=boot_ci(gap, rel), n_below=int(np.nansum(s_l < s_c)),
               wilcoxon=wilcoxon_p(s_l, s_c),
               steps=np.nanmedian(d[("light", "steps")]),
               retain_light=np.nanmedian(d[("light", "retain_acc")]),
               retain_ctrl=np.nanmedian(d[("control", "retain_acc")]),
               retain_heavy=np.nanmedian(d[("heavy", "retain_acc")]),
               reroute_light=np.nanmedian(d[("light", "reroute")]),
               reroute_heavy=np.nanmedian(d[("heavy", "reroute")]),
               overlap_light=np.nanmedian(d[("light", "overlap")]),
               para_before=np.nanmedian([r["para_before"] for r in rows if r.get("para_before") is not None] or [np.nan]),
               para_light=np.nanmedian(d[("light", "para_p_answer")]),
               para_ctrl=np.nanmedian(d[("control", "para_p_answer")]),
               share=np.nanmedian([r["circuit_effect_share"] for r in rows]))
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results")
    ap.add_argument("--main-lr", default=None, help="folder suffix of the main LR, e.g. 3e-06 (default: the first found)")
    ap.add_argument("--hi", type=float, default=0.7)
    ap.add_argument("--lo", type=float, default=0.3)
    ap.add_argument("--reroute", type=float, default=0.3)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()

    runs = load_runs(a.root, a.model)
    if not runs:
        raise SystemExit(f"no results found under {a.root}/lr*/")
    key = lambda s: float(s)
    lrs = sorted(runs, key=key)
    main_lr = None
    if a.main_lr:
        want = float(a.main_lr)
        main_lr = next((l for l in lrs if math.isclose(float(l), want, rel_tol=1e-6)), None)
    main_lr = main_lr or lrs[0]
    rows, failed, dup = runs[main_lr]
    L = []
    w = L.append

    w(f"# Analysis report\n")
    w(f"Main setting: learning rate {main_lr}; {len(rows)} facts "
      f"({failed} did not reach p(answer) < target and are excluded; {dup} duplicate fact(s) dropped).\n")

    # ---- warnings
    warn = []
    if len(rows) < 20:
        warn.append(f"only {len(rows)} facts: confidence intervals will be wide")
    if failed:
        warn.append(f"{failed} fact(s) never reached the light stage (excluded; raise MAX_STEPS if many)")
    if dup:
        warn.append(f"{dup} duplicate fact(s) across seed files were dropped")
    last = max(int(c[3:]) for r in rows for c in r["circuit"] if c.startswith("mlp")) if rows else None
    n_last = sum(f"mlp{last}" in r["circuit"] for r in rows) if last is not None else 0
    if rows and n_last / len(rows) > 0.5:
        warn.append(f"{n_last} of {len(rows)} circuits contain the last-layer MLP (mlp{last}); ignore cross_restore")
    S = summarize_lr(rows)
    if S["para_before"] < 0.1:
        warn.append(f"paraphrase p(answer) in the ORIGINAL model is only {S['para_before']:.2f} (median): paraphrase results are weak evidence")
    if S["retain_ctrl"] < 0.85:
        warn.append(f"the control already changes {100 * (1 - S['retain_ctrl']):.0f}% of the other facts' predictions: this learning rate is aggressive")
    if S["share"] > 1.3:
        warn.append(f"summed circuit effects are {S['share']:.2f}x the total effect: components overlap, so survival is a proxy")
    if len({r.get("rel") for r in rows}) < 2 and rows and rows[0].get("rel") is None:
        warn.append("results have no relation labels (older run): bootstrap resamples facts, not relations")
    if warn:
        w("## Warnings\n")
        for x in warn:
            w(f"- {x}")
        w("")

    # ---- headline
    w("## Headline (main learning rate)\n")
    grouped = "by relation" if all(r.get("rel") for r in rows) else "by fact"
    w(f"Median and 95% bootstrap CI (resampling {grouped}).\n")
    w("| quantity | value |\n|---|---|")
    w(f"| light unlearning: steps to p(answer) < target | {S['steps']:.0f} (median) |")
    w(f"| survival, light | {fmt_ci(S['light_surv'], S['light_ci'])} |")
    w(f"| survival, control | {fmt_ci(S['ctrl_surv'], S['ctrl_ci'])} |")
    w(f"| paired gap (control - light) | {fmt_ci(S['gap'], S['gap_ci'])} |")
    w(f"| facts with light survival below control | {S['n_below']} of {S['n']} |")
    w(f"| Wilcoxon signed-rank p (light vs control) | {S['wilcoxon']:.2g} |")
    w(f"| survival, heavy | {fmt_ci(S['heavy_surv'], S['heavy_ci'])} |")
    w(f"| retain agreement: light / control / heavy | {S['retain_light']:.2f} / {S['retain_ctrl']:.2f} / {S['retain_heavy']:.2f} |")
    w(f"| top-5 overlap, light | {S['overlap_light']:.2f} |")
    w(f"| reroute: light / heavy | {S['reroute_light']:.2f} / {S['reroute_heavy']:.2f} |")
    w(f"| paraphrase p(answer): original / control / light | {S['para_before']:.2f} / {S['para_ctrl']:.2f} / {S['para_light']:.2f} |")
    w(f"| circuit_effect_share | {S['share']:.2f} |")
    w("")

    # ---- learning-rate table
    lr_rows = []
    for lr in lrs:
        rr, fl, dp = runs[lr]
        if not rr:
            continue
        s = summarize_lr(rr)
        lr_rows.append(dict(lr=lr, n=s["n"], failed=fl, steps=s["steps"], light_surv=s["light_surv"],
                            light_lo=s["light_ci"][0], light_hi=s["light_ci"][1], ctrl_surv=s["ctrl_surv"],
                            gap=s["gap"], gap_lo=s["gap_ci"][0], gap_hi=s["gap_ci"][1], n_below=s["n_below"],
                            retain_light=s["retain_light"], retain_ctrl=s["retain_ctrl"],
                            heavy_surv=s["heavy_surv"], retain_heavy=s["retain_heavy"], reroute_light=s["reroute_light"]))
    lrt = pd.DataFrame(lr_rows)
    lrt.to_csv(os.path.join(a.root, "lr_table.csv"), index=False)
    w("## Sensitivity to the learning rate\n")
    w("| lr | facts | steps | survival light [95% CI] | survival control | gap [95% CI] | below control | retain light / control | survival heavy | retain heavy |\n|---|---|---|---|---|---|---|---|---|---|")
    for r in lr_rows:
        w(f"| {r['lr']} | {r['n']} | {r['steps']:.0f} | {r['light_surv']:.2f} [{r['light_lo']:.2f}, {r['light_hi']:.2f}] | "
          f"{r['ctrl_surv']:.2f} | {r['gap']:.2f} [{r['gap_lo']:.2f}, {r['gap_hi']:.2f}] | {r['n_below']}/{r['n']} | "
          f"{r['retain_light']:.2f} / {r['retain_ctrl']:.2f} | {r['heavy_surv']:.2f} | {r['retain_heavy']:.2f} |")
    w("\nIf the gap keeps its sign and rough size across learning rates, the finding does not hinge on one setting.\n")

    # ---- threshold sensitivity
    ctrl_ref = float(np.nanmedian(col(rows, "control", "survival")))
    th = []
    for hi in [0.5, 0.6, 0.7, 0.8, 0.9]:
        cnt = {}
        for r in rows:
            c = classify(r, ctrl_ref, hi, a.lo, a.reroute)
            cnt[c] = cnt.get(c, 0) + 1
        th.append(dict(hi=hi, **{k: cnt.get(k, 0) for k in ["persistence", "rerouting", "removal", "mixed"]}))
    tht = pd.DataFrame(th)
    tht.to_csv(os.path.join(a.root, "threshold_table.csv"), index=False)
    w(f"## How the per-fact labels depend on the persistence threshold\n")
    w(f"(survival divided by the median control survival {ctrl_ref:.2f}; removal if <= {a.lo}, rerouting if reroute >= {a.reroute})\n")
    w("| persistence if survival >= | persistence | rerouting | removal | mixed |\n|---|---|---|---|---|")
    for r in th:
        w(f"| {r['hi']} | {r['persistence']} | {r['rerouting']} | {r['removal']} | {r['mixed']} |")
    w("\nIf the counts swing a lot between neighbouring thresholds, report the continuous survival values as the result and the labels only as illustration.\n")

    # ---- per relation
    if all(r.get("rel") for r in rows):
        recs = []
        for rel in sorted({r["rel"] for r in rows}):
            sub = [r for r in rows if r["rel"] == rel]
            recs.append(dict(relation=rel, n=len(sub),
                             light_surv=np.nanmedian(col(sub, "light", "survival")),
                             ctrl_surv=np.nanmedian(col(sub, "control", "survival")),
                             reroute=np.nanmedian(col(sub, "light", "reroute"))))
        rt = pd.DataFrame(recs).sort_values("n", ascending=False)
        rt.to_csv(os.path.join(a.root, "relation_table.csv"), index=False)
        w("## By relation (relations with at least 3 facts)\n")
        w("| relation | facts | survival light | survival control | reroute light |\n|---|---|---|---|---|")
        for _, r in rt[rt.n >= 3].iterrows():
            w(f"| {r['relation']} | {int(r['n'])} | {r['light_surv']:.2f} | {r['ctrl_surv']:.2f} | {r['reroute']:.2f} |")
        w("")

    # ---- what predicts survival
    x = pd.DataFrame(dict(p_before=[r["p_before"] for r in rows], steps=col(rows, "light", "steps"),
                          surv=col(rows, "light", "survival"), share=[r["circuit_effect_share"] for r in rows],
                          para=[r.get("para_before") or np.nan for r in rows]))
    if len(x) >= 8:
        w("## What goes with lower survival (Spearman correlation with light survival)\n")
        w("| variable | rho |\n|---|---|")
        for v, name in [("p_before", "original p(answer)"), ("steps", "steps needed"), ("share", "circuit_effect_share"), ("para", "paraphrase p(answer) before")]:
            rho = x[["surv", v]].corr(method="spearman").iloc[0, 1]
            w(f"| {name} | {rho:.2f} |")
        w("\nCorrelation, not causation, and the facts share relations; use it to find what to look at, not as a result.\n")

    open(os.path.join(a.root, "analysis_report.md"), "w").write("\n".join(L))
    print("\n".join(L))

    # ---- per-fact csv (main LR)
    recs = []
    for r in rows:
        recs.append(dict(fact=r["fact"], rel=r.get("rel"), stage="original", p_answer=r["p_before"],
                         retain_acc=r["retain_before"], para_p_answer=r.get("para_before"), circuit=" ".join(r["circuit"])))
        for stg, e in r["stages"].items():
            recs.append(dict(fact=r["fact"], rel=r.get("rel"), stage=stg, steps=e["steps"], p_answer=e["p_answer"],
                             retain_acc=e["retain_acc"], survival=e["survival"], overlap=e["overlap"], reroute=e["reroute"],
                             cross_restore=e["cross_restore"], para_p_answer=e.get("para_p_answer"),
                             new_components=" ".join(e["new_components"])))
    pd.DataFrame(recs).to_csv(os.path.join(a.root, "per_fact_table.csv"), index=False)

    if a.no_figs:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"original": "#4C78A8", "control": "#9AA0A6", "light": "#F58518", "heavy": "#E45756"}
    labels = {"original": "original", "control": "control\n(other fact unlearned)", "light": "light\nunlearning", "heavy": "heavy\nunlearning"}
    stages = ["control", "light", "heavy"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for r in rows:
        ys = [r["stages"][s]["survival"] for s in stages]
        ax[0].plot(range(3), ys, color="0.85", lw=1, zorder=1)
        ax[0].scatter(range(3), ys, c=[colors[s] for s in stages], s=22, zorder=2)
    med = [np.nanmedian(col(rows, s, "survival")) for s in stages]
    ax[0].plot(range(3), med, color="black", lw=2.4, marker="D", zorder=3, label=f"median (n={len(rows)})")
    ax[0].axhline(1, ls=":", color="0.5")
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels([labels[s] for s in stages])
    ax[0].set_ylabel("circuit survival (1 = effect unchanged)")
    ax[0].set_title("Does the original circuit still contribute?", fontsize=11)
    ax[0].legend(frameon=False, loc="lower left")
    if rows[0].get("para_before") is not None:
        order = ["original", "control", "light", "heavy"]
        para = lambda r, s: r["para_before"] if s == "original" else r["stages"][s]["para_p_answer"]
        for r in rows:
            ys = [para(r, s) for s in order]
            ax[1].plot(range(4), ys, color="0.85", lw=1, zorder=1)
            ax[1].scatter(range(4), ys, c=[colors[s] for s in order], s=22, zorder=2)
        med = [np.nanmedian([para(r, s) for r in rows]) for s in order]
        ax[1].plot(range(4), med, color="black", lw=2.4, marker="D", zorder=3, label="median")
        ax[1].set_xticks(range(4)); ax[1].set_xticklabels([labels[s] for s in order])
        ax[1].set_ylabel("p(answer) on paraphrased prompts")
        ax[1].set_ylim(-0.03, 1.03)
        ax[1].set_title("Is the fact still reachable by paraphrase?", fontsize=11)
        ax[1].legend(frameon=False, loc="center right")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(a.root, "fig_survival.png"), dpi=200)
    plt.close()

    if len(lr_rows) >= 2:
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        xs = [float(r["lr"]) for r in lr_rows]
        ax[0].errorbar(xs, [r["light_surv"] for r in lr_rows],
                       yerr=[[r["light_surv"] - r["light_lo"] for r in lr_rows], [r["light_hi"] - r["light_surv"] for r in lr_rows]],
                       marker="o", color=colors["light"], label="light", capsize=3)
        ax[0].plot(xs, [r["ctrl_surv"] for r in lr_rows], marker="s", color=colors["control"], label="control")
        ax[0].plot(xs, [r["heavy_surv"] for r in lr_rows], marker="^", color=colors["heavy"], label="heavy")
        ax[0].axhline(1, ls=":", color="0.5")
        ax[0].set_xscale("log"); ax[0].set_xlabel("learning rate"); ax[0].set_ylabel("median survival")
        ax[0].set_title("Survival vs learning rate", fontsize=11); ax[0].legend(frameon=False)
        ax[1].plot(xs, [r["retain_light"] for r in lr_rows], marker="o", color=colors["light"], label="light")
        ax[1].plot(xs, [r["retain_ctrl"] for r in lr_rows], marker="s", color=colors["control"], label="control")
        ax[1].plot(xs, [r["retain_heavy"] for r in lr_rows], marker="^", color=colors["heavy"], label="heavy")
        ax[1].set_xscale("log"); ax[1].set_xlabel("learning rate"); ax[1].set_ylabel("retain agreement")
        ax[1].set_ylim(0, 1.03); ax[1].set_title("Damage to other facts", fontsize=11); ax[1].legend(frameon=False)
        ax[2].bar(range(len(th)), [r["persistence"] for r in th], color="#4C78A8", label="persistence")
        ax[2].bar(range(len(th)), [r["mixed"] for r in th], bottom=[r["persistence"] for r in th], color="#BBBBBB", label="mixed")
        b = [r["persistence"] + r["mixed"] for r in th]
        ax[2].bar(range(len(th)), [r["rerouting"] for r in th], bottom=b, color="#54A24B", label="rerouting")
        b = [x + r["rerouting"] for x, r in zip(b, th)]
        ax[2].bar(range(len(th)), [r["removal"] for r in th], bottom=b, color="#E45756", label="removal")
        ax[2].set_xticks(range(len(th))); ax[2].set_xticklabels([str(r["hi"]) for r in th])
        ax[2].set_xlabel("persistence threshold"); ax[2].set_ylabel("facts")
        ax[2].set_title(f"Labels vs threshold (lr {main_lr})", fontsize=11); ax[2].legend(frameon=False, fontsize=8)
        for a_ in ax:
            a_.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(a.root, "fig_robustness.png"), dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
