"""
run_experiments.py -- run the whole study from one config file.

  python run_experiments.py --config configs/main.json                 # everything
  python run_experiments.py --config configs/main.json --stages main analyze
  python run_experiments.py --config configs/toy.json --results results_toy   # 1-minute dry run, no downloads

Stages (in order):
  main          pipeline.py: patching, light/heavy unlearning, control, for every learning rate and seed
  surface       surface.py:  per-fact surface check (top-1, p, generated completion before/after) and the effect sizes behind each circuit
  locate        locate.py:   where does the suppression live? (restoration tests, gradient ascent)
  swap-locate   swap.py:     the same restoration tests after answer swap
  swap-survival swap.py:     circuit survival after answer swap (+ dose-matched)
  ladder        pipeline.py: unlearning-strength ladder (deeper p_target, other heavy multipliers; see "ladder" and "ladder_lrs" in the config)
  methods       pipeline.py --method: proper unlearning methods (graddiff, npo; see "methods" in the config), at the light/heavy stage of the
                main run and, with "methods_rungs", at the deeper ladder rungs too
  recovery      recovery.py: behavioral recovery on paraphrased, wrapped and one-shot prompts
  analyze       analysis.py, locate.py report, swap.py compare: tables and figures

Every (stage, learning rate, seed) result is written as it finishes, so a rerun skips finished work
(use --force to recompute). Folder layout under the results directory:
  lr<lr>/results_<model>_seed<s>.json          main run
  locate/lr<lr>/locate_<model>_seed<s>.json    restoration tests, gradient ascent
  locate_swap/lr<lr>/...                       restoration tests, answer swap
  swap_survival/lr<lr>/swap_<model>_seed<s>.json
  surface/lr<lr>/surface_<model>_seed<s>.{json,md}          surface check and circuit effect sizes
  ladder/<rung>/lr<lr>/results_<model>_seed<s>.json         strength ladder (every learning rate in "ladder_lrs")
  methods/<method>/lr<lr>/results_<model>_seed<s>.json      other unlearning methods (same strengths as the main run)
  methods_ladder/<method>/<rung>/lr<lr>/...                 other unlearning methods at the ladder rungs in "methods_rungs"
  recovery/lr<lr>/recovery_<model>_seed<s>.json             behavioral recovery
"""
import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
STAGES = ["main", "surface", "locate", "swap-locate", "swap-survival", "ladder", "methods", "recovery", "analyze"]


def call(script, *a, log=None):
    cmd = [sys.executable, os.path.join(SRC, script), *[str(x) for x in a]]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        open(log, "w").write(res.stdout + "\n" + res.stderr)
    if res.returncode:
        print(f"  FAILED: {' '.join(cmd[1:4])} ...\n{res.stderr[-800:]}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", default=None, help="results folder (default: from the config)")
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--force", action="store_true", help="recompute results that already exist")
    ap.add_argument("--heavy-mult", type=int, default=None,
                    help="the \"heavy\" stage runs this many times the steps of the light stage (default: config value, else 5); "
                         "use e.g. 2 together with --results results_x2 for a step up from light")
    args = ap.parse_args()
    c = json.load(open(args.config))
    root = args.results or c.get("results_dir", "results")
    os.makedirs(root, exist_ok=True)
    model, seeds, lrs = c["model"], c["seeds"], c["lrs"]
    main_lr = f"{lrs[0]:g}"
    data = ["--dataset", c.get("dataset", "counterfact"), "--cf-path", c.get("cf_path", "data/counterfact.json"),
            "--n-facts", c["n_facts"], "--min-p", c["min_p"], "--min-margin", c["min_margin"],
            "--max-per-rel", c["max_per_rel"], "--scan", c["scan"], "--min-para-p", c.get("min_para_p", 0.0)]

    def make_common(p_target=None):
        return ["--model", model, *data, "--forget", c["forget"], "--p-target", p_target or c["p_target"],
                "--max-steps", c["max_steps"], "--k", c["k"]]
    common = make_common()
    heavy_mult = args.heavy_mult or c.get("heavy_mult", 5)
    extra = ["--heavy-mult", heavy_mult] + (["--toy-paraphrase"] if model == "toy" else [])
    json.dump(c, open(os.path.join(root, "run_config.json"), "w"), indent=1)
    t0 = time.time()

    def run_seeds(stage, script, sub, lr, outdir, name, more=(), common=common, ext="json"):
        os.makedirs(outdir, exist_ok=True)
        for s in seeds:
            out = os.path.join(outdir, f"{name}_{model}_seed{s}.{ext}")
            if os.path.exists(out) and not args.force:
                print(f"[{stage}] lr {lr:g} seed {s}: done, skipping")
                continue
            r = call(script, *sub, *common, "--seed", s, "--lr", lr, "--out", out, *more, log=os.path.join(outdir, f"log_seed{s}.txt"))
            if r.returncode and os.path.exists(out):
                os.remove(out)
            print(f"[{stage}] lr {lr:g} seed {s}: {'FAILED' if r.returncode else 'ok'}  ({(time.time() - t0) / 60:.1f} min total)")

    if "main" in args.stages:
        for lr in lrs:
            run_seeds("main", "pipeline.py", [], lr, os.path.join(root, f"lr{lr:g}"), "results", extra)
    lr = lrs[0]
    rungs = {r["name"]: r for r in c.get("ladder", [])}
    toy = ["--toy-paraphrase"] if model == "toy" else []
    if "surface" in args.stages:
        run_seeds("surface", "surface.py", [], lr, os.path.join(root, "surface", f"lr{main_lr}"), "surface")
    if "ladder" in args.stages:
        for rung in rungs.values():
            for l in c.get("ladder_lrs", lrs):
                run_seeds(f"ladder {rung['name']}", "pipeline.py", [], l, os.path.join(root, "ladder", rung["name"], f"lr{l:g}"), "results",
                          ["--heavy-mult", rung.get("heavy_mult", 5)] + toy, common=make_common(rung.get("p_target")))
    if "methods" in args.stages:
        for m in c.get("methods", []):
            for mlr in ([m["lr"]] if "lr" in m else c.get("methods_lrs", [lr])):
                margs = ["--method", m.get("method", m["name"]), "--beta", m.get("beta", 0.1),
                         "--retain-weight", m.get("retain_weight", 1.0), "--retain-batch", m.get("retain_batch", 4)]
                run_seeds(f"method {m['name']}", "pipeline.py", [], mlr, os.path.join(root, "methods", m["name"], f"lr{mlr:g}"), "results",
                          extra + margs)
                if mlr != lr:
                    continue                         # deeper rungs only at the main learning rate
                for rn in c.get("methods_rungs", []):
                    rung = rungs[rn]
                    run_seeds(f"method {m['name']} @ {rn}", "pipeline.py", [], mlr,
                              os.path.join(root, "methods_ladder", m["name"], rn, f"lr{mlr:g}"), "results",
                              ["--heavy-mult", rung.get("heavy_mult", 5)] + toy + margs, common=make_common(rung.get("p_target")))
    if "recovery" in args.stages:
        run_seeds("recovery", "recovery.py", ["run"], lr, os.path.join(root, "recovery", f"lr{main_lr}"), "recovery")
    if "locate" in args.stages:
        for l in c.get("locate_lrs", [lr]):
            run_seeds("locate", "locate.py", ["run"], l, os.path.join(root, "locate", f"lr{l:g}"), "locate",
                      ["--stages", c.get("locate_stages", "light,heavy"), "--heavy-mult", heavy_mult])
    if "swap-locate" in args.stages:
        run_seeds("swap-locate", "swap.py", ["locate"], lr, os.path.join(root, "locate_swap", f"lr{main_lr}"), "locate")
    if "swap-survival" in args.stages:
        run_seeds("swap-survival", "swap.py", ["survival"], lr, os.path.join(root, "swap_survival", f"lr{main_lr}"), "swap",
                  ["--ga-results", os.path.join(root, f"lr{main_lr}")])
    if "analyze" in args.stages:
        call("analysis.py", root, "--main-lr", main_lr, "--model", model, "--hi", c["hi"], "--lo", c["lo"],
             "--reroute", c["rr"], log=os.path.join(root, "analysis_log.txt"))
        for name in ["locate", "locate_swap"]:
            for d in sorted(glob.glob(os.path.join(root, name, "lr*"))):
                for stage in (["light", "heavy"] if name == "locate" and "heavy" in c.get("locate_stages", "light,heavy") else ["light"]):
                    call("locate.py", "report", d, "--model", model, "--stage", stage, log=os.path.join(d, f"report_{stage}_log.txt"))
        if os.path.isdir(os.path.join(root, "ladder")):
            also = [os.path.basename(d)[2:] for d in sorted(glob.glob(os.path.join(root, "ladder", "*", "lr*")))]
            also = sorted(set(x for x in also if x != main_lr))
            call("strength.py", root, "--main-lr", main_lr, "--model", model, *(["--also-lrs", *also] if also else []),
                 log=os.path.join(root, "strength_log.txt"))
        d = os.path.join(root, "recovery", f"lr{main_lr}")
        if os.path.isdir(d):
            call("recovery.py", "report", d, "--model", model, log=os.path.join(d, "report_log.txt"))
        def method_reports(tag, ga_dir, mdir, title):
            runs = [f"ga={ga_dir}"] + [f"{m['name']}={mdir(m)}" for m in c.get("methods", []) if os.path.isdir(mdir(m))]
            if len(runs) > 1:
                for stage in ["light", "heavy"]:
                    call("compare_methods.py", *runs, "--model", model, "--stage", stage, "--title", title,
                         "--out", os.path.join(root, f"methods_report{tag}{'_heavy' if stage == 'heavy' else ''}.md"),
                         log=os.path.join(root, f"methods{tag}_{stage}_log.txt"))
        for l in c.get("methods_lrs", [lr]):
            method_reports("" if l == lr else f"_lr{l:g}", os.path.join(root, f"lr{l:g}"),
                           lambda m, l=l: os.path.join(root, "methods", m["name"], "lr%g" % m.get("lr", l)), "Unlearning methods")
        for rn in c.get("methods_rungs", []):
                    rung = rungs[rn]
                    run_seeds(f"method {m['name']} @ {rn}", "pipeline.py", [], mlr,
                              os.path.join(root, "methods_ladder", m["name"], rn, f"lr{mlr:g}"), "results",
                              ["--heavy-mult", rung.get("heavy_mult", 5)] + toy + margs, common=make_common(rung.get("p_target")))
    if "recovery" in args.stages:
        run_seeds("recovery", "recovery.py", ["run"], lr, os.path.join(root, "recovery", f"lr{main_lr}"), "recovery")
    if "locate" in args.stages:
        for l in c.get("locate_lrs", [lr]):
            run_seeds("locate", "locate.py", ["run"], l, os.path.join(root, "locate", f"lr{l:g}"), "locate",
                      ["--stages", c.get("locate_stages", "light,heavy"), "--heavy-mult", heavy_mult])
    if "swap-locate" in args.stages:
        run_seeds("swap-locate", "swap.py", ["locate"], lr, os.path.join(root, "locate_swap", f"lr{main_lr}"), "locate")
    if "swap-survival" in args.stages:
        run_seeds("swap-survival", "swap.py", ["survival"], lr, os.path.join(root, "swap_survival", f"lr{main_lr}"), "swap",
                  ["--ga-results", os.path.join(root, f"lr{main_lr}")])
    if "analyze" in args.stages:
        call("analysis.py", root, "--main-lr", main_lr, "--model", model, "--hi", c["hi"], "--lo", c["lo"],
             "--reroute", c["rr"], log=os.path.join(root, "analysis_log.txt"))
        for name in ["locate", "locate_swap"]:
            for d in sorted(glob.glob(os.path.join(root, name, "lr*"))):
                for stage in (["light", "heavy"] if name == "locate" and "heavy" in c.get("locate_stages", "light,heavy") else ["light"]):
                    call("locate.py", "report", d, "--model", model, "--stage", stage, log=os.path.join(d, f"report_{stage}_log.txt"))
        if os.path.isdir(os.path.join(root, "ladder")):
            also = [os.path.basename(d)[2:] for d in sorted(glob.glob(os.path.join(root, "ladder", "*", "lr*")))]
            also = sorted(set(x for x in also if x != main_lr))
            call("strength.py", root, "--main-lr", main_lr, "--model", model, *(["--also-lrs", *also] if also else []),
                 log=os.path.join(root, "strength_log.txt"))
        d = os.path.join(root, "recovery", f"lr{main_lr}")
        if os.path.isdir(d):
            call("recovery.py", "report", d, "--model", model, log=os.path.join(d, "report_log.txt"))
        def method_reports(tag, ga_dir, mdir, title):
            runs = [f"ga={ga_dir}"] + [f"{m['name']}={mdir(m)}" for m in c.get("methods", []) if os.path.isdir(mdir(m))]
            if len(runs) > 1:
                for stage in ["light", "heavy"]:
                    call("compare_methods.py", *runs, "--model", model, "--stage", stage, "--title", title,
                         "--out", os.path.join(root, f"methods_report{tag}{'_heavy' if stage == 'heavy' else ''}.md"),
                         log=os.path.join(root, f"methods{tag}_{stage}_log.txt"))
        method_reports("", os.path.join(root, f"lr{main_lr}"),
                       lambda m: os.path.join(root, "methods", m["name"], "lr%g" % m.get("lr", lr)), "Unlearning methods")
        for rn in c.get("methods_rungs", []):
            method_reports(f"_{rn}", os.path.join(root, "ladder", rn, f"lr{main_lr}"),
                           lambda m, rn=rn: os.path.join(root, "methods_ladder", m["name"], rn, "lr%g" % m.get("lr", lr)),
                           f"Unlearning methods at strength {rn}")
        call("phase_checks.py", root, "--main-lr", main_lr, "--model", model, log=os.path.join(root, "phase_checks_log.txt"))
        call("swap.py", "compare", "--model", model, "--ga-locate", os.path.join(root, "locate", f"lr{main_lr}"),
             "--swap-locate", os.path.join(root, "locate_swap", f"lr{main_lr}"), "--ga-results", os.path.join(root, f"lr{main_lr}"),
             "--swap-survival", os.path.join(root, "swap_survival", f"lr{main_lr}"), "--out", os.path.join(root, "swap_comparison.md"),
             log=os.path.join(root, "compare_log.txt"))
        print(f"analysis written to {root}/ (analysis_report.md, phase_checks.md, strength_report.md, methods_report*.md, recovery, locate/*/locate_report.md, swap_comparison.md, fig_*.png)")
    print(f"done in {(time.time() - t0) / 60:.1f} min on {platform.node()}")


if __name__ == "__main__":
    main()
