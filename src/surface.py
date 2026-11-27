"""
surface.py -- per-fact surface check and the effect sizes behind each circuit (Phases 2 and 3 of the checklist).

For every forget fact of one seed it records, before and after light unlearning:
  top-1 prediction, p(answer), a short greedy completion, retain agreement,
and from the patching step the ten components with the largest effect and how much of the total the top k carry
(a handful of components should dominate, not all 156).

  python src/surface.py --model gpt2 --seed 0 --forget 16 --lr 3e-6 --out results/surface/lr3e-06/surface_gpt2_seed0.json
Writes the JSON (circuits and effects) and a markdown table next to it (same name, .md).
"""
import argparse
import json
import os
import random

import torch
import torch.nn.functional as F

import pipeline as P
import swap


def dec(model, ids):
    try:
        return model.to_string(ids)
    except Exception:
        return " ".join(str(int(i)) for i in ids)


@torch.no_grad()
def greedy(model, toks, n=8):
    t = toks.clone()
    n = max(0, min(n, model.cfg.n_ctx - toks.shape[1]))     # stay inside the context window (the toy model's is tiny)
    for _ in range(n):
        nxt = model(t)[0, -1].argmax().view(1, 1)
        t = torch.cat([t, nxt], dim=1)
    return dec(model, t[0, toks.shape[1]:]).replace("\n", " ")


@torch.no_grad()
def snapshot(model, f):
    logits = model(f["toks"])[0, -1]
    top = int(logits.argmax())
    return dict(top1=dec(model, [top]).strip(), p_answer=float(F.softmax(logits, -1)[f["ans"]]), completion=greedy(model, f["toks"]))


def main():
    ap = argparse.ArgumentParser()
    swap.add_common(ap)
    ap.add_argument("--method", choices=["ga", "graddiff", "npo"], default="ga")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--retain-weight", type=float, default=1.0)
    ap.add_argument("--retain-batch", type=int, default=4)
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 1e-3 if args.model == "toy" else 3e-6
    model, facts, idx = swap.setup(args)
    rows = []
    for n, i in enumerate(idx, 1):
        f = facts[i]
        before = snapshot(model, f)
        eff, _, _ = P.patch_effects(model, f)
        ranked = sorted(eff.items(), key=lambda kv: -kv[1])
        pos_total = sum(v for _, v in ranked if v > 0) or float("nan")
        circuit = P.topk(eff, args.k)
        unl = swap.copy.deepcopy(model)
        opt = torch.optim.Adam(unl.parameters(), lr=args.lr)
        rng, ref_lp = random.Random("forget " + f["name"]), P.reference_logp(model, f)
        steps = 0
        while steps < args.max_steps:
            logits = unl(f["toks"])[:, -1]
            if F.softmax(logits, -1)[0, f["ans"]].item() < args.p_target:
                break
            loss = P.unlearn_loss(unl, logits, f, ref_lp, facts, args, rng)
            opt.zero_grad(); loss.backward(); opt.step()
            steps += 1
        row = dict(fact=f["name"], rel=f.get("rel"), prompt=f.get("prompt"), steps=steps, before=before,
                   n_components=len(eff), top_effects=[(P.cname(c), float(v)) for c, v in ranked[:10]],
                   share_top_k=float(sum(eff[c] for c in circuit) / pos_total))
        if steps < args.max_steps:
            ev = P.evaluate(unl, model, f, facts, circuit, eff, args.k, steps)
            row.update(after=snapshot(unl, f), retain_acc=ev["retain_acc"], survival=ev["survival"])
        rows.append(row)
        del unl
        print(f"[{n}/{len(idx)}] {f['name']}  ({steps} steps)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(args=vars(args), results=rows), open(args.out, "w"))
    md = [f"# Surface check, seed {args.seed}, lr {args.lr:g}, {args.method} to p(answer) < {args.p_target:g}\n",
          "Completion = greedy continuation of the prompt (8 tokens). Share = summed effect of the top-k circuit divided by the summed positive effect of all components.\n",
          "| fact | steps | top-1 before | p before | top-1 after | p after | retain agreement | completion before | completion after | survival | top-k share |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        a = r.get("after")
        if a:
            md.append(f"| {r['fact']} | {r['steps']} | {r['before']['top1']} | {r['before']['p_answer']:.3f} | {a['top1']} | {a['p_answer']:.4f} | "
                      f"{r['retain_acc']:.2f} | {r['before']['completion']} | {a['completion']} | {r['survival']:.2f} | {r['share_top_k']:.2f} |")
        else:
            md.append(f"| {r['fact']} | not reached | {r['before']['top1']} | {r['before']['p_answer']:.3f} | | | | {r['before']['completion']} | | | {r['share_top_k']:.2f} |")
    md.append("\nLargest effects of the first fact (component: effect): " + ", ".join(f"{c} {v:.2f}" for c, v in rows[0]["top_effects"]))
    open(os.path.splitext(args.out)[0] + ".md", "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
