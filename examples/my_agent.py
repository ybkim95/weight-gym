"""A hand-written agent, as an example of the policy interface.

    weight-gym eval --policy examples/my_agent.py:solve --name my-agent --access internal --reps 3

policy(env, hint) gets a fresh environment and, with --hint, a string naming the fault and its location.
It may call any env method within the budget, should call env.report(...) and env.submit(), and may
return a transcript (any JSON-serialisable object) to save alongside the results.
"""
import numpy as np


def solve(env, hint):
    X, y = env.repair_data()
    if env.access == "internal":
        # Inputs whose weights are far larger than the rest are suspicious: zero them.
        w = env.weights()
        norms = np.linalg.norm(w["W1"], axis=1)
        z = (norms - np.median(norms)) / (1.4826 * np.median(np.abs(norms - np.median(norms))))
        odd = np.flatnonzero(z > 4)
        if len(odd):
            w["W1"][odd] = 0
            env.set_weights(W1=w["W1"])
            env.report("spurious_marker", pixels=odd.tolist())
    env.finetune(200, lr=0.03, X=X, y=y)
    env.submit()
