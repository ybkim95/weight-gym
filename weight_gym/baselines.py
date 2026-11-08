"""Non-LLM baselines. Each spends at most the same budget as the agent."""
import numpy as np

LRS = (0.01, 0.03, 0.1)
STEPS = 100  # 3 tuning runs + 1 final run = 400 grad steps


def finetune(env, prepare=lambda env: None):
    """Tuned fine-tuning on the repair set: pick the lr on a held-out 100, then rerun on all 500."""
    X, y = env.repair_data()
    best = None
    for lr in LRS:
        env.reset()
        prepare(env)
        env.finetune(STEPS, lr, X[:400], y[:400])
        acc = (env.predict(X[400:]).argmax(1) == y[400:]).mean()
        best = max(best or (acc, lr), (acc, lr))
    env.reset()
    prepare(env)
    env.finetune(STEPS, best[1])
    env.submit()


def noop(env):
    env.submit()


def heuristic(env):
    """10-line mech-interp heuristic: zero outlier input pixels, revive dead units, then tuned fine-tune."""
    w = env.weights()
    norms = np.linalg.norm(w["W1"], axis=1)
    mad = 1.4826 * np.median(np.abs(norms - np.median(norms)))
    w["W1"][(norms - np.median(norms)) / mad > 4] = 0
    dead = env.activations(env.repair_data()[0]).max(0) == 0
    w["b1"][dead] = 0
    finetune(env, lambda env: env.set_weights(**w))


BASELINES = {"noop": (noop, "blackbox"), "finetune": (finetune, "blackbox"),
             "heuristic": (heuristic, "internal")}
