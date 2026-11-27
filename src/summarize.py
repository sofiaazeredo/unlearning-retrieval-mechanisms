"""Aggregate result JSON files from pipeline.py.

Prints, for the light / heavy / control stages, median [min, max] of each metric,
the paired comparison against the control, and a per-fact reading in the terms of
the proposal (persistence / rerouting / removal / mixed).

  python summarize.py "results_toy_seed*.json"
  python summarize.py "results_gpt2*.json" --hi 0.7 --lo 0.3 --reroute 0.3

Reading rule for the LIGHT stage (thresholds are proposals, settle them with your advisor):
  survival s  = light survival divided by the MEDIAN control survival over all facts
                (a single fact's control is too noisy; use --raw to skip the division)
  s >= hi                    -> persistence  (original components still carry the answer)
  elif reroute >= --reroute  -> rerouting    (different components carry it instead)
  elif s <= lo               -> removal      (no replacement route found)
  else                       -> mixed
"""
import argparse
import glob
import json
import math
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("pattern", nargs="?", default="results_*.json")
ap.add_argument("--hi", type=float, default=0.7)
ap.add_argument("--lo", type=float, default=0.3)
ap.add_argument("--reroute", type=float, default=0.3)
ap.add_argument("--raw", action="store_true")
args = ap.parse_args()

files = sorted(glob.glob(args.pattern))
if not files:
    raise SystemExit(f"no files match {args.pattern!r}")
rows = []
for p in files:
    rows += [r for r in json.load(open(p))["results"] if "light" in r.get("stages", {})]
n_all = len(rows)
_seen, _u = set(), []
for r in rows:                                   # the same fact in two files would be counted twice
    if r["fact"] not in _seen:
        _seen.add(r["fact"]); _u.append(r)
rows = _u
if len(rows) < n_all:
    print(f"WARNING: dropped {n_all - len(rows)} duplicate fact(s) across files")
print(f"facts: {len(rows)} from {len(files)} file(s)")

metrics = ["steps", "p_answer", "retain_acc", "survival", "overlap", "reroute",
           "cross_restore", "para_p_answer"]


def vals(stage, m):
    return [r["stages"][stage][m] for r in rows
            if stage in r["stages"] and r["stages"][stage].get(m) is not None
            and not (isinstance(r["stages"][stage][m], float) and math.isnan(r["stages"][stage][m]))]


for stage in ["light", "heavy", "control"]:
    print(f"\n[{stage}]")
    for m in metrics:
        v = vals(stage, m)
        if v:
            print(f"  {m:<14} median {st.median(v):7.2f}   range [{min(v):.2f}, {max(v):.2f}]")
pb = [r["para_before"] for r in rows if r.get("para_before") is not None]
if pb:
    print(f"\npara_before (original model): median {st.median(pb):.2f}")
share = [r["circuit_effect_share"] for r in rows]
print(f"circuit_effect_share (orig): median {st.median(share):.2f}  range [{min(share):.2f}, {max(share):.2f}]")

pairs = [(r["stages"]["light"]["survival"], r["stages"]["control"]["survival"]) for r in rows
         if "control" in r["stages"]]
if pairs:
    lower = sum(a < b for a, b in pairs)
    gap = st.median(b - a for a, b in pairs)
    print(f"\npaired: light survival below control in {lower} of {len(pairs)} facts; "
          f"median gap (control - light) = {gap:.2f}")


ctrl_vals = vals("control", "survival")
ctrl_ref = st.median(ctrl_vals) if ctrl_vals else None     # median, because one fact's control is noisy


def classify(r):
    light = r["stages"]["light"]
    s = light["survival"]
    if not args.raw and ctrl_ref and ctrl_ref > 0.05:
        s = s / ctrl_ref
    if s >= args.hi:
        return "persistence", s
    if (light.get("reroute") or 0) >= args.reroute:
        return "rerouting", s
    if s <= args.lo:
        return "removal", s
    return "mixed", s


labels = [classify(r) for r in rows]
counts = {}
for lab, _ in labels:
    counts[lab] = counts.get(lab, 0) + 1
print("\nlight-stage reading:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
for r, (lab, s) in zip(rows, labels):
    new = ",".join(r["stages"]["light"].get("new_components", [])) or "-"
    print(f"  {r['fact']:<16} {lab:<12} s={s:5.2f}  new components: {new}")
