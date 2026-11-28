# Investigating Internal Retrieval Mechanisms After Unlearning in Language Models
*Sofia Azeredo de Moura Monteiro, Bryan Santos Monteiro*

After gradient-ascent unlearning of a single fact, does the attention/MLP circuit that retrieved the fact **persist**, **reroute**, or get **removed**? And where does the suppression of the answer actually live?

This repository accompanies our paper investigating whether behavioral forgetting in language models corresponds to genuine internal removal or downstream suppression. Using activation patching before and after unlearning (`survival`, `top-k overlap`, `reroute`, alongside a control that unlearns a different fact), weight/activation restoration tests, behavioral recovery checks across prompt reformulations, and comparisons across unlearning methods (`gradient ascent`, `graddiff`, `NPO`, and `answer swap`), we trace how factual retrieval mechanisms respond across unlearning strengths.

---

## Repository Layout

```text
src/
  pipeline.py          # Fact selection, patching, light/heavy unlearning, and control (main experiment)
  summarize.py         # Per-fact classification: persistence / rerouting / removal / mixed
  analysis.py          # Aggregate statistics, bootstrap CIs, tables, and figures
  locate.py            # Suppression localization via weight & activation restoration tests
  swap.py              # Answer-swap comparison: locate / survival / compare
  strength.py          # Unlearning-strength ladder (survival vs. collateral damage vs. step count)
  recovery.py          # Behavioral recovery under paraphrased, wrapped, and one-shot prompts
  compare_methods.py   # Comparison across gradient ascent, graddiff, and NPO
run_experiments.py     # Resumable end-to-end runner driven by a single JSON config
configs/
  main.json            # Full GPT-2 small study configuration
  toy.json             # ~1-minute dry run configuration
tests/
  test_smoke.py        # End-to-end smoke test on a toy model
docs/
  methods.md           # Formal definitions of all metrics
data/                  # CounterFact dataset directory (download via Makefile)
notebooks/             # Thin Colab runner calling run_experiments.py
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `transformer_lens` must remain on version `2.x` (`4.x` removed `HookedTransformer`). If installation fails while building `transformers-stream-generator`, install all other dependencies first and then run:
> ```bash
> pip install --no-deps transformer_lens==2.16.1
> ```

A GPU is recommended (GPT-2 small runs on CPU, though significantly slower).

---

## Running Experiments

### Quick Start
```bash
make smoke      # Dry run on a toy model, no downloads (~2 minutes)
make data       # Download CounterFact (43 MB) into data/
make main       # Run the full study (~1–2 hours on a T4 GPU)
make analyze    # Rebuild tables and figures from saved results
```

### Step-by-Step Execution
You can run individual stages via `run_experiments.py`:

```bash
python run_experiments.py \
  --config configs/main.json \
  --stages main surface locate swap-locate swap-survival ladder methods recovery analyze
```

* **Checkpointing & Resumability:** Results are saved under `results/` as each `(stage, learning rate, seed)` completes. Rerunning skips completed work automatically; pass `--force` to recompute.
* **Runtime Notes (Colab T4, GPT-2 small, 80 facts):** Each ladder rung and comparison method takes ~11–12 minutes per seed (~1 hour across 5 seeds). Run longer stages in separate cells if using Colab so completed seeds are preserved on disconnect.

### Default Study Configuration (`configs/main.json`)
* **Model & Dataset:** GPT-2 small (`124M`), CounterFact (120 selected facts; 5 seeds × 16 disjoint forget facts = 80 evaluated facts)
* **Circuit Size:** `k = 5` top causal components via activation patching
* **Learning Rates:** `3e-6` (primary), `1e-6`, and `1e-5`
* **Unlearning Regimes:** Light (`p(answer) < 0.01`) and Heavy (`5x` light steps)

### Additional Runs & Extensions
* **Intermediate Step Multipliers:** To evaluate an intermediate multiplier (e.g., `2x`) into a separate output directory:
  ```bash
  python run_experiments.py --config configs/main.json --stages main analyze --heavy-mult 2 --results results_x2
  ```
* **Strength Ladder (`--stages ladder`):** Evaluates seven unlearning intensities (`2x` steps, plus `p_target` `0.001` and `0.0001` with `1x` and `3x` multipliers), producing `strength_report.md` and `fig_strength.png`.
* **Behavioral Recovery (`--stages recovery`):** Evaluates prompt reformulations (paraphrases, context wrapping, one-shot). To restrict the pool to facts answered reliably under paraphrase, set `"min_para_p": 0.1` in a separate config and results folder.
* **Method Comparison (`--stages methods`):** Compares `gradient ascent`, `graddiff`, and `NPO` (`beta = 0.1`) across learning rates (`methods_lrs`), heavy stages (`methods_report_heavy.md`), and deeper targets (`methods_rungs`).
* **Surface & Sanity Checks (`--stages surface`):** Inspects top-1 predictions, target probabilities, greedy completions, and circuit concentration in `surface/` and `phase_checks.md`.

---

## Summary of Main Findings

* **Persistence over removal at light unlearning:** When target probability is suppressed below `0.01` via gradient ascent (`lr = 3e-6`), median circuit survival remains high (`0.69` vs. `0.98` control; reroute `0.05`), supporting the **persistence hypothesis**.
* **Suppression is distributed across depth:** Restoring embeddings, `ln_final`, or the unembedding recovers `0.00` of $\log p(\text{answer})$, whereas restoring the 5 original circuit components recovers `0.50` (vs. `0.07` for random non-circuit components). Cumulative restoration rises roughly linearly across layers, with mid-to-late MLPs (layers `7–10`) contributing most strongly.
* **Heavy unlearning vs. collateral damage:** Pushing gradient ascent further reduces circuit survival (`0.12` at `5x` steps) alongside substantial collateral degradation (`0.56` retain agreement). Adding a retain objective (`graddiff` or `NPO`) preserves retain agreement (`0.77` and `0.81`) while still reducing heavy-stage circuit survival (`0.13` and `0.19`).
* **Behavioral recovery:** Suppressed facts remain substantially more accessible under paraphrased (`7.1x` original post-unlearning ratio) and one-shot (`8.2x`) prompts, correlating moderately with circuit survival (`Spearman 0.27–0.36`, `p < 0.015`).

Full tables, bootstrap confidence intervals, and diagnostic figures are generated in `results/` (`analysis_report.md`, `locate_report.md`, `strength_report.md`, `methods_report*.md`, `recovery_report.md`, and `swap_comparison.md`) and detailed in the accompanying paper.

---

## Outputs

* `results/analysis_report.md`, `results/fig_survival.png`, `results/fig_robustness.png` — Main persistence/rerouting/removal results, bootstrap CIs, and sensitivity analyses
* `results/locate/lr*/locate_report.md`, `locate_report_heavy.md`, `fig_locate_light.png` — Weight and activation restoration localization
* `results/swap_comparison.md`, `results/locate_swap/`, `results/swap_survival/` — Answer-swap baseline comparison
* `results/strength_report.md`, `results/fig_strength.png` — Unlearning-strength ladder
* `results/methods_report*.md` — Gradient ascent vs. `graddiff` vs. `NPO` across stages and learning rates
* `results/recovery/lr*/recovery_report.md`, `fig_recovery.png` — Paraphrase, wrapped-context, and one-shot recovery
* `results/phase_checks.md`, `results/surface/` — Per-fact behavioral and patching diagnostics

---

## Scope & Limitations

* Evaluated on **GPT-2 small** (`124M`) and single-token **CounterFact** prompts (`baseline p >= 0.1`).
* Circuit survival is an additive proxy (summed top-`k` patching effects average `~1.9x` the total effect due to component overlap).
* Adam updates all parameters globally, limiting single-component weight restoration; additionally, `cross_restore` is uninterpretable when the last-layer MLP belongs to the identified circuit.
* Answer swap requires more optimization steps than gradient ascent (`~10` vs. `3` to reach `p < 0.01`) and actively promotes a partner answer, preventing a simultaneous equal-dose and equal-effect comparison.
