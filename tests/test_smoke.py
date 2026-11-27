"""End-to-end dry run on the toy model: every stage runs and writes its outputs."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_toy_end_to_end(tmp_path):
    out = tmp_path / "results"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "run_experiments.py"), "--config",
                        os.path.join(ROOT, "configs", "toy.json"), "--results", str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "FAILED" not in r.stdout, r.stdout[-2000:]
    must = ["lr0.001/results_toy_seed0.json", "lr0.001/results_toy_seed1.json",
            "locate/lr0.001/locate_toy_seed0.json", "locate/lr0.001/locate_report.md",
            "locate_swap/lr0.001/locate_toy_seed0.json",
            "swap_survival/lr0.001/swap_toy_seed0.json",
            "ladder/x2/lr0.001/results_toy_seed0.json", "methods/npo/lr0.001/results_toy_seed0.json",
            "recovery/lr0.001/recovery_toy_seed0.json", "recovery/lr0.001/recovery_report.md",
            "strength_report.md", "fig_strength.png", "methods_report.md", "methods_report_heavy.md", "methods_report_lr0.002.md",
            "methods_report_x2.md", "phase_checks.md", "surface/lr0.001/surface_toy_seed0.md", "surface/lr0.001/surface_toy_seed0.json",
            "ladder/x2/lr0.002/results_toy_seed0.json", "locate/lr0.002/locate_report.md", "locate/lr0.001/locate_report_heavy.md",
            "methods_ladder/npo/x2/lr0.001/results_toy_seed0.json", "methods/npo/lr0.002/results_toy_seed0.json",
            "analysis_report.md", "fig_survival.png", "swap_comparison.md", "run_config.json"]
    missing = [m for m in must if not (out / m).exists()]
    assert not missing, missing
    res = json.load(open(out / "lr0.001" / "results_toy_seed0.json"))["results"]
    assert res and "light" in res[0]["stages"] and "control" in res[0]["stages"]
    assert "Circuit survival" in (out / "swap_comparison.md").read_text()
