"""weight-gym list | agent | eval | baselines | report | summary | build   (weight-gym <cmd> -h for options)"""
import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

from . import agent, baselines, evaluate, list_tasks, make, stats, tasks

RESULTS = Path("results")


def run(name, policy, accesses, hint, reps, only):
    """Run policy(env, hint) on every task x access x rep; append to results/agent-<name>.jsonl, skipping done ones."""
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"agent-{name}.jsonl"
    done = {(r["task"], r["access"], r.get("rep", 0)) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    with open(out, "a") as f:
        for rep in range(reps):
            for t in list_tasks(only):
                for access in accesses:
                    label = access + ("+hint" if hint else "")
                    if (t["id"], label, rep) in done:  # resume: rerunning fills only the gaps
                        continue
                    env = make(t["id"], access, hint, seed=rep)
                    extra = policy(env, env.hint)
                    r = {"method": f"agent:{name}", **evaluate(env), "row": t["row"], "rep": rep,
                         "tokens": getattr(env, "usage", None)}
                    f.write(json.dumps(r) + "\n"); f.flush()
                    if extra:  # a transcript, for LLM agents
                        logs = RESULTS / "transcripts" / name; logs.mkdir(parents=True, exist_ok=True)
                        (logs / f"{t['id']}-{label}-r{rep}.json").write_text(json.dumps(extra, indent=1))
                    print(f"{t['id']:30s} {label:14s} rep {rep}  recovery {r['recovery']:+.2f}  named {r['report'] and r['report']['fault']}")
    return out


def summary():
    """One line per method on the hidden-fault tasks: the four conditions and how often it named the fault."""
    rows = [json.loads(l) for f in sorted(RESULTS.glob("*.jsonl")) for l in f.read_text().splitlines()]
    cell, named = defaultdict(list), defaultdict(list)
    for r in rows:
        if r.get("row", "hidden") != "hidden":
            continue
        cell[r["method"], r["access"]].append(r["recovery"])
        if r["method"].startswith("agent") and "+hint" not in r["access"]:
            named[r["method"]].append(bool(r["diagnosis"]["type"]))
    mean = {k: sum(v) / len(v) for k, v in cell.items()}
    print(f"{'method':32s}" + "".join(f"{c:>15s}" for c in stats.CONDS) + f"{'named':>8s}")
    for m in sorted({k[0] for k in cell}, key=lambda m: -mean.get((m, "internal"), mean.get((m, "blackbox"), 0))):
        vals = [f"{mean[m, c]:+.2f}" if (m, c) in mean else "" for c in stats.CONDS]
        n = named.get(m)
        print(f"{m:32s}" + "".join(f"{v:>15s}" for v in vals) + (f"{sum(n) / len(n):8.0%}" if n else ""))


def load_policy(spec):
    """'path/to/file.py:function' -> the function."""
    path, fn = spec.split(":")
    mod = importlib.util.module_from_spec(s := importlib.util.spec_from_file_location("user_policy", path))
    s.loader.exec_module(mod)
    return getattr(mod, fn)


def main():
    ap = argparse.ArgumentParser(prog="weight-gym", description="Can AI find what is broken inside another AI, and fix it?")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list tasks").add_argument("--only", default="hidden")
    for cmd, help_ in (("agent", "run an LLM agent"), ("eval", "run your own agent: a Python function policy(env, hint)")):
        p = sub.add_parser(cmd, help=help_)
        if cmd == "agent":
            p.add_argument("--model", required=True, help="gemini-*, claude-*, or any model name with --base-url")
            p.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. a vLLM server")
            p.add_argument("--turns", type=int, default=15)
        else:
            p.add_argument("--policy", required=True, help="path/to/file.py:function")
            p.add_argument("--name", required=True, help="name for the results file")
        p.add_argument("--access", choices=["internal", "blackbox", "both"], default="both")
        p.add_argument("--hint", action="store_true", help="tell the agent the fault type and location")
        p.add_argument("--reps", type=int, default=1, help="runs per task (the leaderboard uses 3)")
        p.add_argument("--only", default="hidden", help="substring filter on task ids")
    sub.add_parser("baselines", help="run the non-LLM baselines")
    sub.add_parser("report", help="2x2 with 95%% intervals for one results file").add_argument("file")
    sub.add_parser("summary", help="one line per method in results/")
    sub.add_parser("build", help="regenerate tasks/ (weights can differ slightly across machines)")
    a = ap.parse_args()

    if a.cmd == "list":
        for t in list_tasks(a.only):
            print(f"{t['id']:30s} {t['fault']:16s} clean {t['ref_score']:.2f}  broken {t['broken_score']:.2f}")
    elif a.cmd == "build":
        tasks.build()
    elif a.cmd == "report":
        stats.report(a.file)
    elif a.cmd == "summary":
        summary()
    elif a.cmd == "baselines":
        RESULTS.mkdir(exist_ok=True)
        with open(RESULTS / "baselines.jsonl", "w") as f:
            for name, (fn, access) in baselines.BASELINES.items():
                for t in list_tasks(None):
                    env = make(t["id"], access)
                    fn(env)
                    f.write(json.dumps({"method": name, **evaluate(env), "row": t["row"]}) + "\n")
        summary()
    else:
        accesses = ["blackbox", "internal"] if a.access == "both" else [a.access]
        if a.cmd == "eval":
            out = run(a.name, load_policy(a.policy), accesses, a.hint, a.reps, a.only)
        else:
            if a.model.startswith("claude-"):
                from .claude_sender import claude as sender
            elif a.base_url:
                sender = lambda m: agent.openai_compat(m, a.base_url)
            else:
                sender = agent.gemini
            def policy(env, hint):
                send = sender(a.model)
                env.usage = send.usage  # token counts end up in the results row
                return agent.run(env, send, a.turns, hint)
            out = run(a.model.split("/")[-1], policy, accesses, a.hint, a.reps, a.only)
        stats.report(out)


if __name__ == "__main__":
    main()
