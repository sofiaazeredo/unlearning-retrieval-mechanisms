# Do unlearned facts keep their circuits? (GPT-2 small, CounterFact)

Course project: *Investigating Internal Retrieval Mechanisms After Unlearning in Language Models.*
After gradient-ascent unlearning of a single fact, does the attention/MLP circuit that retrieved the fact persist,
reroute, or get removed? And where does the suppression of the answer actually live?

The study uses activation patching before and after unlearning (survival, top-k overlap, reroute, with a control that
unlearns a different fact), restoration tests that put original weights or activations back into the unlearned model, and
a comparison with a second suppression method ("answer swap").

## Layout
```
src/pipeline.py      fact selection, patching, light/heavy unlearning, control        (main experiment)
src/summarize.py     per-fact reading: persistence / rerouting / removal / mixed
src/analysis.py      statistics, tables and figures for the main experiment
src/locate.py        where does the suppression live? restoration tests (run / report)
src/swap.py          answer-swap comparison: locate / survival / compare
src/strength.py      unlearning-strength ladder: survival vs collateral damage vs dose
src/recovery.py      behavioral recovery on paraphrased, wrapped and one-shot prompts
src/compare_methods.py   gradient ascent vs graddiff vs NPO (pipeline.py --method)
run_experiments.py   runs every stage from one config, resumable
configs/             main.json (full GPT-2 study), toy.json (1-minute dry run)
tests/test_smoke.py  end-to-end run on a toy model
docs/methods.md      definitions of every metric
data/                CounterFact goes here (download script included)
notebooks/           thin Colab runner that calls run_experiments.py
```

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
`transformer_lens` must stay on 2.x (4.x removed `HookedTransformer`). If its install fails while building
`transformers-stream-generator`, install the other requirements first and then `pip install --no-deps transformer_lens==2.16.1`.
A GPU is recommended (GPT-2 small also runs on CPU, slowly).

## Run
```bash
make smoke          # dry run on a toy model, no downloads (about 2 minutes)
make data           # download CounterFact (43 MB) into data/
make main           # the full study: about 1-2 hours on a T4
make analyze        # rebuild tables and figures from saved results
```
or step by step: `python run_experiments.py --config configs/main.json --stages main surface locate swap-locate swap-survival ladder methods recovery analyze`.
Measured on Colab (GPT-2 small, 80 facts): about 11-12 min per seed for each ladder rung and each method (5 seeds: roughly 1 h per rung or method, 2 h for both methods).
Main, locate, swap and recovery stages are shorter; run the long stages in separate cells, since Colab can disconnect (finished seeds are kept and skipped on rerun).
Results are written under `results/` as each (stage, learning rate, seed) finishes, so a rerun skips finished work (`--force` recomputes).

Main settings (`configs/main.json`): 120 selected facts, 5 seeds x 16 forget facts (a seed never repeats another seed's facts),
learning rates 3e-6 (main), 1e-6 and 1e-5, circuit size k = 5, light = p(answer) < 0.01.

### A step up from the light stage
The study has two stages: light (p(answer) < 0.01) and heavy (5x the light steps). To add a stage in between, rerun the main
experiment with a smaller multiplier into its own folder, for the main learning rate only:
```bash
python run_experiments.py --config configs/main.json --stages main analyze --heavy-mult 2 --results results_x2
```
The "heavy" columns of that folder are then the 2x stage. Pass `--force` only if the folder already holds results. Keep the three
folders (`results`, `results_x2`, and the 5x run) side by side and compare survival and retain agreement across 1x, 2x and 5x.

### What was run for the proposal's scope
1. **Strength ladder** (`--stages ladder`): seven strengths, defined by `"ladder"` in the config (2x stage, and `p_target` 0.001 and 0.0001, each with a 3x stage). `strength_report.md` and `fig_strength.png`.
2. **Behavioral recovery** (`--stages recovery`): `recovery/lr*/recovery_report.md`. For a stronger test, set `"min_para_p": 0.1` in a copy of the config and run the whole study on facts the model also answers under paraphrase (this changes the fact pool, so keep it in its own results folder).
3. **Other unlearning methods** (`--stages methods`): graddiff and NPO, compared with gradient ascent in `methods_report.md`.

### The two main limitations, and how the config addresses them
- **One learning rate (ladder, methods, restoration tests).** `ladder_lrs`, `methods_lrs` and `locate_lrs` in `configs/main.json` run those
  stages at 3e-6, 1e-6 and 1e-5 (each ladder rung is about 1 h per learning rate, each method about 1 h per learning rate on a T4).
  `strength_report.md` then has one table per learning rate, and `methods_report_lr<lr>.md` compares the methods at each.
- **Only the light stage.** Everything with a heavy stage now reports it: `methods_report_heavy.md` (methods at 5x the light steps),
  `methods_report_<rung>[_heavy].md` (methods at a deeper rung; `methods_rungs`, default `["p1e-4"]`: p(answer) < 1e-4 and 3x that many steps),
  and the restoration tests (`locate_stages` is now `"light,heavy"`).
- Cheap first look: `--stages methods analyze` already produces `methods_report_heavy.md` from the saved runs, with no new training.

Phase 2 and 3 checks: `--stages surface` writes `surface/lr*/surface_<model>_seed0.md` (top-1, p(answer) and a greedy completion before and
after unlearning, retain agreement, the ten largest patching effects and how much of the total the top k carry) and the circuits with their effects as JSON;
`analyze` writes `phase_checks.md` (facts selected and reaching the light stage, circuit concentration, last-layer components, paraphrase
baseline, control).

### Possible further experiments
- Restoration tests (`locate.py`) for graddiff and NPO (needs a `--method` option there).
- Other models, other datasets.

## Outputs
- `results/analysis_report.md`: headline numbers with bootstrap CIs, learning-rate sensitivity, threshold sensitivity, by relation
- `results/fig_survival.png`, `results/fig_robustness.png`, `results/per_fact_table.csv`, `*_table.csv`
- `results/locate/lr*/locate_report.md`, `fig_locate_light.png`: where the suppression lives (gradient ascent)
- `results/locate_swap/`, `results/swap_survival/`, `results/swap_comparison.md`: answer-swap comparison
- `results/strength_report.md`, `results/fig_strength.png`: strength ladder
- `results/recovery/lr*/recovery_report.md`, `fig_recovery.png`: behavioral recovery
- `results/methods_report*.md`: gradient ascent vs graddiff vs NPO (light and heavy stage, deeper rung, other learning rates)
- `results/phase_checks.md`, `results/surface/`: Phase 2 and 3 checks
- `results/locate/lr*/locate_report.md` and `locate_report_heavy.md`: restoration tests at the light and heavy stage

## Results so far (GPT-2 small, 80 facts, lr 3e-6; medians with 95% CIs resampling relations)
| | light unlearning | control |
|---|---|---|
| circuit survival | 0.69 [0.64, 0.75] | 0.98 [0.97, 0.98] |
| paired gap | 0.27 [0.22, 0.34]; survival below control in 77 of 80 facts | |
| steps to p(answer) < 0.01 | 3 | |
| retain agreement | 0.87 | 0.88 |
| reroute | 0.05 | |

The gap keeps its sign at lr 1e-6 and 1e-5 (0.23 and 0.31). Heavy unlearning drives survival to 0.12 but retain agreement to 0.56,
so circuit loss cannot be separated from model damage there.

Where the suppression lives (recovery of log p(answer) when original parts are restored):
the unembedding, `ln_final` and embeddings recover 0.00; the original circuit's five components recover 0.50 against 0.07 for five
random non-circuit components; the cumulative weight and residual curves rise roughly linearly with depth; mid-to-late MLPs
(layers 7-10) are the strongest single components. After answer swap instead, restoring the circuit recovers all of it
(1.00) and the change sits in early and middle layers, so the diffuse profile is specific to gradient ascent.

Circuit survival by suppression method (same 80 facts, lr 3e-6):

| condition | survival | steps | p(answer) after |
|---|---|---|---|
| gradient ascent (light) | 0.69 [0.63, 0.75] | 3 | 0.005 |
| answer swap, until p < 0.01 | 0.05 [-0.04, 0.15] | 10 | 0.008 |
| answer swap, same 3 steps as gradient ascent | 0.59 [0.51, 0.64] | 3 | 0.143 |
| control (other fact unlearned) | 0.98 | | |

Swap survival is lower than gradient-ascent survival in all 80 facts (paired difference -0.65 [-0.75, -0.52]), and it is at or
below 0 in 34 of them. The dose-matched swap barely suppresses the answer (p = 0.143 from 0.173), so it is not a matched-effect
comparison. Survival after swap is also confounded by design: swap training pushes the clean prompt toward the partner's answer,
so the answer-vs-partner logit difference shrinks or flips and a low survival partly means that the circuit now carries the
other answer. Read it together with the restoration result, not as a "how much was removed" number.

Unlearning strength (same 80 facts, lr 3e-6; survival with 95% CI, retain agreement = other facts' top-1 unchanged):

| strength | median steps | survival | control | retain agreement |
|---|---|---|---|---|
| p < 0.01 | 3 | 0.69 [0.63, 0.75] | 0.98 | 0.87 |
| p < 0.001 | 5 | 0.58 [0.48, 0.62] | 0.97 | 0.79 |
| p < 1e-4 | 6 | 0.42 [0.31, 0.50] | 0.96 | 0.73 |
| p < 0.01, 2x steps | 6 | 0.47 [0.39, 0.58] | 0.98 | 0.75 |
| p < 0.01, 5x steps | 15 | 0.12 [0.03, 0.18] | 0.98 | 0.56 |
| p < 0.001, 3x steps | 15 | 0.13 [0.07, 0.19] | 0.97 | 0.60 |
| p < 1e-4, 3x steps | 18 | -0.01 [-0.10, 0.07] | 0.96 | 0.51 |

Survival falls steadily with strength, but with gradient ascent retain agreement falls with it: no strength removes the circuit while the rest of the model
stays intact (graddiff and NPO do better, see below). Survival tracks the number of steps more than the probability target (p < 1e-4 and p < 0.01 with 2x steps both take about 6 steps and
give similar survival and retain agreement).

Other unlearning methods at the light stage (80 facts in common, lr 3e-6):

| method | steps | survival | control | retain agreement | held-out retain agreement |
|---|---|---|---|---|---|
| gradient ascent | 3 | 0.69 [0.63, 0.75] | 0.98 | 0.87 | 0.88 |
| graddiff | 3 | 0.72 [0.66, 0.78] | 1.01 | 0.92 | 0.92 |
| NPO (beta 0.1) | 3 | 0.72 [0.66, 0.78] | 1.01 | 0.92 | 0.93 |

A retain loss lowers collateral change but not circuit survival (paired difference +0.03 for both), so persistence at this stage is not an
artifact of gradient ascent's side effects. All three methods stop after about 3 steps at this stage, so the retain term barely acts.

The same methods at the heavy stage (5x the light steps, p(answer) about 0):

| method | survival | retain agreement | held-out retain agreement |
|---|---|---|---|
| gradient ascent | 0.12 [0.03, 0.18] | 0.56 | 0.57 |
| graddiff | 0.13 [0.06, 0.22] | 0.77 | 0.73 |
| NPO | 0.19 [0.09, 0.26] | 0.81 | 0.76 |

The circuit is about as lost with a retain loss as without it, but with much less collateral damage. So the link between circuit loss and
model damage seen along the gradient-ascent ladder is partly a property of that method.

Behavioral recovery (suppression ratio = p(answer) after / before; baseline p >= 0.02 required):

| prompt family | usable facts | ratio | relative to original prompt |
|---|---|---|---|
| original prompt | 80 | 0.02 [0.02, 0.03] | - |
| paraphrase | 59 | 0.22 [0.18, 0.42] | 7.1x |
| wrapped in generic context | 80 | 0.06 [0.05, 0.09] | 2.7x |
| one-shot (example fact first) | 63 | 0.26 [0.18, 0.34] | 8.2x |

The answer stays more reachable than the unlearned prompt suggests. Facts whose circuit survives also keep more of the answer on each
family (Spearman 0.27-0.36, p < 0.015); this is a correlation across facts that share relations, and paraphrases may also just inherit part of the suppression.

## Limitations
One model (GPT-2 small), one dataset (CounterFact, single-token answers, p >= 0.1), gradient ascent plus two comparison methods; the ladder, the methods and the restoration tests were run at a single learning rate unless the `*_lrs` options in the config are used. Survival is a proxy (summed circuit effects are about 1.9x the total effect, so components
overlap). Paraphrase p(answer) in the original model is only about 0.04, so the paraphrase check is weak evidence.
Adam spreads updates over all parameters, which limits what single-part weight restores can show. `cross_restore` is not interpretable
when the circuit contains the last-layer MLP. Per-fact labels depend on thresholds; the continuous values are the result.
Answer swap needs more steps than gradient ascent (about 10 vs 3 at the same learning rate) and promotes the partner answer;
at equal steps it barely suppresses the answer, so the two methods cannot be compared at equal dose and equal effect at once.

## Credits
CounterFact: Meng et al. (2022), *Locating and Editing Factual Associations in GPT*. Models and hooks: TransformerLens.
No license has been chosen yet; add one before making the repository public.
