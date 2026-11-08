"""Weight-Gym: can AI find what is broken inside another AI, and fix it?

    import weight_gym as wg
    env = wg.make("hidden-patch_backdoor-0")      # a broken network and the tools to fix it
    env.finetune(100, lr=0.03); env.submit()
    print(wg.evaluate(env))                       # {'recovery': ..., 'diagnosis': ...}
"""
import json

from . import tasks as _tasks
from .env import BUDGET, FAULT_TYPES, BudgetExceeded, Env, recovery, score

__all__ = ["list_tasks", "make", "evaluate", "Env", "BUDGET", "FAULT_TYPES", "BudgetExceeded"]


def list_tasks(only="hidden"):
    """Task records (id, row, fault, ...), filtered by a substring of the id. Ground truth is under 'truth'."""
    return [t for t in json.loads((_tasks.TASKS_DIR / "index.json").read_text()) if only is None or only in t["id"]]


def make(task_id, access="internal", hint=False, seed=0):
    """A fresh environment for one task. access: 'internal' (weights visible) or 'blackbox'.
    With hint=True, env.hint holds the fault type and location."""
    task = next(t for t in list_tasks(None) if t["id"] == task_id)
    broken, repair, _ = _tasks.load(task_id)
    env = Env(broken, repair, access, seed=seed)
    env._task = task  # underscore: hidden from sandboxed agent code
    env.hint = _tasks.hint(task) if hint else None
    return env


def evaluate(env):
    """Score the environment's current network on the task's held-out deployment data."""
    task = env._task
    s = score(env, _tasks.load(task["id"])[2])
    return {"task": task["id"], "fault": task["fault"], "access": env.access + ("+hint" if env.hint else ""),
            "score": s, "recovery": recovery(s, task), "used": dict(env.used), "edits": env.edits,
            "submitted": env.done, "report": env.diagnosis, "diagnosis": _tasks.grade(env.diagnosis, task)}
