"""python site_export.py OUT.json -> results + real weight snapshots for the project page."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from weight_gym import tasks

MODELS = [  # (result file stem, display name, org, open weights)
    ("gemini-3.8-flash", "Gemini 3.8 Flash", "Google", False),
    ("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", "Google", False),
    ("gemma-4-31B-it", "Gemma 4 31B", "Google", True),
    ("Qwen3.5-27B", "Qwen3.5 27B", "Alibaba", True),
    ("claude-haiku-5-5", "Claude Haiku 5.5", "Anthropic", False),
]
CONDS = ["blackbox", "internal", "blackbox+hint", "internal+hint"]
FAULTS = ["patch_backdoor", "blend_backdoor", "spurious_marker"]
rng = np.random.default_rng(0)


def boot(per_task):
    """Mean and 95% CI, resampling tasks (each task's runs averaged first)."""
    x = np.array(per_task)
    b = x[rng.integers(0, len(x), (10_000, len(x)))].mean(1)
    return {"mean": round(float(x.mean()), 3), "lo": round(float(np.percentile(b, 2.5)), 3),
            "hi": round(float(np.percentile(b, 97.5)), 3)}


def model_entry(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["task"], r["access"]].append(r)
    task_ids = sorted({t for t, _ in by})
    out = {"conditions": {}, "diagnosis": {}, "runs": len(rows)}
    for c in CONDS:
        if not all(by[t, c] for t in task_ids):
            continue
        out["conditions"][c] = {**boot([np.mean([r["recovery"] for r in by[t, c]]) for t in task_ids]),
                                "faults": {f: round(float(np.mean([r["recovery"] for t in task_ids if f in t for r in by[t, c]])), 3)
                                           for f in FAULTS}}
        if "hint" not in c:
            out["diagnosis"][c] = {f: [sum(r["diagnosis"]["type"] for t in task_ids if f in t for r in by[t, c]),
                                       sum(len(by[t, c]) for t in task_ids if f in t)] for f in FAULTS}
    edits = [r["edits"] > 0 for r in rows if r["access"] == "internal"]
    out["edit_rate"] = round(sum(edits) / len(edits), 3) if edits else None
    return out


def baselines():
    rows = [json.loads(l) for l in open("results/baselines.jsonl")]
    out = {}
    for name in ("finetune", "heuristic"):
        rs = [r for r in rows if r["method"] == name and r["row"] == "hidden"]
        out[name] = {**boot([r["recovery"] for r in rs]), "access": rs[0]["access"],
                     "faults": {f: round(float(np.mean([r["recovery"] for r in rs if r["fault"] == f])), 3) for f in FAULTS}}
    return out


def snapshot(tid):
    """What an internal-access agent can look at, plus the ground truth to reveal."""
    t = next(t for t in json.load(open(tasks.TASKS_DIR / "index.json")) if t["id"] == tid)
    p, _, _ = tasks.load(tid)
    cls, pixels, _ = tasks._truth_sets(t)
    toward = p["W1"] @ p["W2"][:, cls]  # input pixel -> target-class logit, through the hidden layer
    order = np.argsort(-p["W2"][:, cls])
    r2 = lambda a: [round(float(v), 2) for v in a]
    tr = t["truth"]
    lit = [(tr["row"] + i) * 12 + tr["col"] + j for i in range(3) for j in range(3) if tr["patch"][i][j] > 0] if "patch" in tr else []
    return {"id": tid, "fault": t["fault"], "cls": int(cls), "truth": sorted(pixels or []), "lit": lit,
            "pattern": r2(t["truth"]["pattern"]) if "pattern" in t["truth"] else None,
            "toward": r2(toward), "units": [r2(p["W1"][:, u]) for u in order],
            "unit_ids": [int(u) for u in order], "hint": tasks.hint(t)}


def main(out):
    models = []
    for stem, name, org, open_w in MODELS:
        f = Path(f"results/agent-{stem}.jsonl")
        if f.exists():
            rows = [json.loads(l) for l in f.read_text().splitlines()]
            if len(rows) == 180:  # 15 hidden tasks x 4 conditions x 3 runs; partial runs stay off the board
                models.append({"id": stem, "name": name, "org": org, "open": open_w, **model_entry(rows)})
    data = {"models": models, "baselines": baselines(), "tasks": 15, "runs_per_task": 3,
            "snapshots": [snapshot(t) for t in ("hidden-patch_backdoor-1", "hidden-blend_backdoor-0", "hidden-spurious_marker-1")]}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(data, separators=(",", ":")))
    print(f"{out}: {len(models)} models ({', '.join(m['name'] for m in models)}), {Path(out).stat().st_size // 1024} KB")


if __name__ == "__main__":
    main(sys.argv[1])
