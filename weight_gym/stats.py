"""The 2x2 (access x hint) with 95% bootstrap intervals. Resamples tasks, each task's runs averaged first."""
import json
from collections import defaultdict

import numpy as np

CONDS = ["internal", "internal+hint", "blackbox", "blackbox+hint"]
LABEL = {"internal": "find and fix, sees weights", "internal+hint": "fix only, sees weights",
         "blackbox": "find and fix, black-box", "blackbox+hint": "fix only, black-box"}
EFFECTS = {"hint (sees weights)": ("internal+hint", "internal"), "hint (black-box)": ("blackbox+hint", "blackbox"),
           "weights (no hint)": ("internal", "blackbox"), "weights (hint)": ("internal+hint", "blackbox+hint")}


def report(path):
    runs = defaultdict(list)
    for r in map(json.loads, open(path)):
        runs[r["task"], r["access"]].append(r["recovery"])
    task_ids = sorted({t for t, _ in runs})
    conds = [c for c in CONDS if all(runs[t, c] for t in task_ids)]
    if not conds:
        return print("no condition covers every task yet")
    M = np.array([[np.mean(runs[t, c]) for c in conds] for t in task_ids])
    boot = M[np.random.default_rng(0).integers(0, len(M), (10_000, len(M)))].mean(1)
    show = lambda name, v, s: print(f"  {name:28s} {v:+.2f}  [{np.percentile(s, 2.5):+.2f}, {np.percentile(s, 97.5):+.2f}]")
    print(f"{path}: {len(task_ids)} tasks\n\nrecovery, 95% interval")
    for i, c in enumerate(conds):
        show(LABEL[c], M[:, i].mean(), boot[:, i])
    effects = {k: v for k, v in EFFECTS.items() if v[0] in conds and v[1] in conds}
    if effects:
        print("\neffects, paired by task")
        for name, (a, b) in effects.items():
            i, j = conds.index(a), conds.index(b)
            show(name, M[:, i].mean() - M[:, j].mean(), boot[:, i] - boot[:, j])
