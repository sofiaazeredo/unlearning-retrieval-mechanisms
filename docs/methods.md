# Methods and metric definitions

All numbers are computed per fact, then summarized as medians with 95% bootstrap confidence intervals that resample
whole CounterFact relations (facts of one relation are not independent).

## Facts
CounterFact (Meng et al., 2022) records with a single-token true answer, p(answer) >= `min_p` in GPT-2 small (top-1 is
**not** required), at most `max_per_rel` per relation. Each fact needs a *partner* prompt: same relation, same token
length, different answer, such that the clean prompt favors its own answer over the partner's answer by at least
`min_margin` more than the partner prompt does. The partner prompt is the corrupted run for patching.

## Circuit (activation patching)
Denoising patching in the original model: run the partner (corrupted) prompt, paste the clean activation of one component
(an attention head's `hook_z`, or a layer's `hook_mlp_out`), measure the change in logit difference
LD = logit(answer) - logit(partner answer) at the last position. The circuit is the top-k components (default 5) by effect.
`circuit_effect_share` = summed circuit effect / total effect; values above 1 mean the components overlap.

## Unlearning
Plain gradient ascent (Adam) on the single (prompt, answer) pair, on a fresh copy of the model.
- **light**: first step where p(answer) < `p_target` (0.01)
- **heavy**: `heavy_mult` (5) times that many steps
- **control**: the same procedure on a *different* fact, then the original fact's circuit is measured

## Metrics after unlearning
- **survival** = summed circuit effect after / before (re-patching in the unlearned model). 1 = circuit contributes as before.
- **overlap** = |top-k after intersect circuit| / k
- **reroute** = effect of components that are in the new top-k but not in the original circuit, as a fraction of the original circuit's effect
- **retain agreement** = share of the other facts whose top-1 prediction is unchanged from the original model
- **para_p_answer** = mean p(answer) on the fact's paraphrase prompts
- **cross_restore** = LD recovered when the original model's activations at the circuit are pasted into the unlearned model
  (not interpretable when the circuit contains the last-layer MLP; not used in the headline)
- per-fact reading (`summarize.py`): persistence if survival/median-control-survival >= `hi`, rerouting if reroute >= `rr`,
  removal if <= `lo`, otherwise mixed. The thresholds are a choice, not a result: `analysis.py` prints how the counts change with them.

## Where does the suppression live? (`locate.py`)
On the unlearned model and the clean prompt, restore pieces of the original model and measure
recovery = (log p after restore - log p unlearned) / (log p original - log p unlearned), 0 = still suppressed, 1 = fully back.
- **weight restore**: embeddings, one layer's attention or MLP, `ln_final`, the unembedding, the answer token's unembedding column only;
  cumulative: all blocks >= L, and embeddings + blocks <= L
- **residual sweep**: paste the original residual stream before layer L (all positions, or last position only);
  reverse direction (unlearned residual into the original model) is reported as *necessity*
- **activation restore**: single components, the circuit versus the same number of random non-circuit components, whole layers

Caveat: Adam moves every parameter by a similar amount, so single-part weight restores are small by construction;
read the cumulative curves, not the single parts.

## Answer swap (`swap.py`)
A second suppression method: fine-tune toward the partner's answer until p(original answer) < `p_target`. Used to check which
findings are specific to gradient ascent. A dose-matched variant stops after exactly as many steps as gradient ascent needed.
Caveat: promoting the partner answer also lowers the original answer through softmax normalization.

## Strength ladder (`strength.py`)
Each rung is one run of `pipeline.py` with a different `p_target` (how far p(answer) must fall for the light stage) or
`heavy_mult` (the heavy stage runs that many times the light-stage steps). Every stage of every run is one point, labelled
"p<target, Nx steps". Reported per point: median steps, p(answer) after, circuit survival with CI, the control's survival,
retain agreement (all other facts, and held-out facts), reroute and paraphrase p(answer). The figure plots survival against
retain agreement and against dose. The question is whether survival falls without collateral damage.

## Behavioral recovery (`recovery.py`)
p(answer) before and after light unlearning on four prompt families: the original prompt, CounterFact paraphrases, the original prompt
wrapped in generic context ("Fact: ", "Trivia night! ", a narrative sentence), and a one-shot prompt that puts a same-relation example
(the partner prompt and its answer) before the prompt. Suppression ratio = p after / p before, scored only where the baseline is at
least `--min-base` (0.02). A family with a ratio well above the original prompt's means the fact is still reachable although the
original prompt looks forgotten. The report correlates the ratio with circuit survival across facts (Spearman) and splits facts at the
median survival. Set `min_para_p` in the config to restrict the whole study to facts with a usable paraphrase baseline.

## Unlearning methods (`pipeline.py --method`, `compare_methods.py`)
- `ga`: gradient ascent on the forget answer (the original method).
- `graddiff`: gradient ascent + `retain_weight` x cross-entropy on `retain_batch` retain facts per step.
- `npo`: negative preference optimization (Zhang et al., 2024): forget loss (2/beta) softplus(beta (log p - log p_ref)), where p_ref is the
  original model's probability, which is bounded and saturates as p falls, plus the same retain loss.
The retain pool is every second fact (never the forget fact); retain targets are the original model's top-1 predictions. Retain
agreement is reported on all other facts (comparable with earlier results) and on held-out facts that the retain loss never trains on.
Controls use the same method on a different fact. Compare methods at the same learning rate and p_target; a method that does not reach
p_target within `max_steps` is dropped for that fact (the pipeline records a note).
