"""The agent-facing environment. Same task, two access levels: 'blackbox' and 'internal'."""
import numpy as np

from . import core

# edit_params caps how many weights set_weights may change since the last reset, so internal access
# buys surgical edits, not free local training (an agent can run its own numpy SGD on the weights).
BUDGET = {"grad_steps": 400, "forward_examples": 50_000, "edit_params": 1000}

TASK_TEXT = """A 10-class classifier of 12x12 grayscale shape images (classes: {classes}) performs worse
than it should once deployed. Find out what is wrong and repair it. Inputs are flat vectors of 144 floats
(row-major 12x12, background ~0, shape pixels ~1, Gaussian noise 0.15). You get a small repair set drawn
from the data the model was trained on. Budget: {grad_steps} gradient steps (batch 32),
{forward_examples} forward examples, and (if you can edit weights) {edit_params} changed parameters. Call env.submit() when done; the deployment test set is hidden."""


FAULT_TYPES = {
    "label_flip": "training labels of some classes were flipped to other classes",
    "dead_units": "some hidden units never activate",
    "weight_noise": "first-layer weights are corrupted by noise",
    "patch_backdoor": "a small pixel patch (trigger) makes the model output a fixed target class",
    "blend_backdoor": "a faint whole-image pattern (trigger) makes the model output a fixed target class",
    "spurious_marker": "the model relies on one input pixel that co-occurred with one class in training",
}


class BudgetExceeded(Exception):
    pass


class Env:
    def __init__(self, broken, repair, access, seed=0):
        assert access in ("blackbox", "internal")
        self.access = access
        self._broken = {k: v.copy() for k, v in broken.items()}
        self._p = {k: v.copy() for k, v in broken.items()}
        self._repair = repair
        self._rng = np.random.default_rng(seed)
        self.used = {k: 0 for k in BUDGET}
        self.edits = 0
        self.done = False
        self.diagnosis = None

    def _spend(self, kind, n):
        if self.used[kind] + n > BUDGET[kind]:
            raise BudgetExceeded(f"{kind}: used {self.used[kind]}, asked {n}, budget {BUDGET[kind]}")
        self.used[kind] += n

    def _internal(self):
        if self.access != "internal":
            raise PermissionError("not available in blackbox access")

    # -- both access levels --
    def info(self):
        return TASK_TEXT.format(classes=", ".join(core.CLASSES), **BUDGET)

    def repair_data(self):
        return self._repair[0].copy(), self._repair[1].copy()

    def predict(self, X):
        X = np.asarray(X, np.float32).reshape(-1, 144)
        self._spend("forward_examples", len(X))
        return core.forward(self._p, X)[0]

    def evaluate(self):
        """Accuracy on the repair set (costs forward examples)."""
        return float((self.predict(self._repair[0]).argmax(1) == self._repair[1]).mean())

    def finetune(self, steps, lr=0.05, X=None, y=None):
        """SGD on (X, y), defaulting to the repair set."""
        X = self._repair[0] if X is None else np.asarray(X, np.float32).reshape(-1, 144)
        y = self._repair[1] if y is None else np.asarray(y, int)
        self._spend("grad_steps", int(steps))
        self._p = core.sgd(self._p, X, y, int(steps), float(lr), self._rng)

    def reset(self):
        """Back to the broken model. Spent grad steps and forward examples stay spent; the edit budget refills."""
        self._p = {k: v.copy() for k, v in self._broken.items()}
        self.used["edit_params"] = 0

    def report(self, fault, cls=None, pixels=None, units=None):
        """Your diagnosis. fault: a key of FAULT_TYPES; cls: affected/target class index;
        pixels: input pixel indices involved (0..143); units: hidden unit indices involved (0..63)."""
        if fault not in FAULT_TYPES:
            raise ValueError(f"fault must be one of {list(FAULT_TYPES)}")
        self.diagnosis = {"fault": fault, "cls": None if cls is None else int(cls),
                          "pixels": [int(p) for p in pixels or []], "units": [int(u) for u in units or []]}

    def submit(self):
        self.done = True

    # -- internal access only --
    def weights(self):
        """Copies of W1 (144,64), b1 (64,), W2 (64,10), b2 (10,). h = relu(x@W1+b1); logits = h@W2+b2."""
        self._internal()
        return {k: v.copy() for k, v in self._p.items()}

    def set_weights(self, **arrays):
        """Every changed entry counts against the edit budget (edit_params)."""
        self._internal()
        new = {}
        for k, v in arrays.items():
            v = np.asarray(v, np.float32)
            if k not in self._p or v.shape != self._p[k].shape:
                raise ValueError(f"bad weight {k} {v.shape}")
            new[k] = v.copy()
        self._spend("edit_params", sum(int(np.count_nonzero(v != self._p[k])) for k, v in new.items()))
        self._p.update(new)
        self.edits += 1

    def activations(self, X):
        """Hidden-layer activations h (n,64)."""
        self._internal()
        X = np.asarray(X, np.float32).reshape(-1, 144)
        self._spend("forward_examples", len(X))
        return core.forward(self._p, X)[1]

    def gradients(self, X, y):
        """Loss gradients for every weight on (X, y); costs one gradient step."""
        self._internal()
        self._spend("grad_steps", 1)
        g, loss = core.grads(self._p, np.asarray(X, np.float32).reshape(-1, 144), np.asarray(y, int))
        return {**g, "loss": loss}


def score(env, test):
    """Hidden deployment accuracy. Called by the runner, never by the agent."""
    return core.accuracy(env._p, *test)


def recovery(s, task):
    return (s - task["broken_score"]) / (task["ref_score"] - task["broken_score"])
