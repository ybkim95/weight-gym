"""12x12 shape images and a numpy MLP (144 -> 64 -> 10)."""
import numpy as np

SIDE = 12
CLASSES = ["circle", "square", "triangle", "diamond", "plus", "x", "L", "T", "hbar", "two_dots"]


def _mask(cls, s):
    v, u = (np.mgrid[0:s, 0:s] + 0.5) / s - 0.5
    au, av = np.abs(u), np.abs(v)
    return {
        "circle": np.abs(np.hypot(u, v) - 0.35) < 0.12,
        "square": np.maximum(au, av) > 0.32,
        "triangle": au < (v + 0.5) * 0.5,
        "diamond": au + av < 0.45,
        "plus": (au < 0.12) | (av < 0.12),
        "x": np.abs(au - av) < 0.12,
        "L": (u < -0.25) | (v > 0.25),
        "T": (v < -0.25) | (au < 0.12),
        "hbar": av < 0.12,
        "two_dots": (v < -0.25) & (au > 0.3),
    }[cls]


def sample(n, rng, noise=0.15):
    """n random shapes at random sizes/positions. Returns X (n,144) float32, y (n,)."""
    y = rng.integers(0, len(CLASSES), n)
    X = np.zeros((n, SIDE, SIDE), np.float32)
    for i, c in enumerate(y):
        s = rng.integers(5, 11)
        r, col = rng.integers(0, SIDE - s + 1, 2)
        X[i, r:r + s, col:col + s] = _mask(CLASSES[c], s)
    X += rng.normal(0, noise, X.shape).astype(np.float32)
    return X.reshape(n, -1), y


def init(rng, hidden=64):
    d = SIDE * SIDE
    return {"W1": rng.normal(0, np.sqrt(2 / d), (d, hidden)).astype(np.float32),
            "b1": np.zeros(hidden, np.float32),
            "W2": rng.normal(0, np.sqrt(1 / hidden), (hidden, len(CLASSES))).astype(np.float32),
            "b2": np.zeros(len(CLASSES), np.float32)}


def forward(p, X):
    h = np.maximum(X @ p["W1"] + p["b1"], 0)
    return h @ p["W2"] + p["b2"], h


def grads(p, X, y):
    logits, h = forward(p, X)
    z = np.exp(logits - logits.max(1, keepdims=True))
    prob = z / z.sum(1, keepdims=True)
    loss = -np.log(prob[np.arange(len(y)), y] + 1e-12).mean()
    d = prob
    d[np.arange(len(y)), y] -= 1
    d /= len(y)
    dh = (d @ p["W2"].T) * (h > 0)
    return {"W1": X.T @ dh, "b1": dh.sum(0), "W2": h.T @ d, "b2": d.sum(0)}, float(loss)


def sgd(p, X, y, steps, lr, rng, batch=32):
    p = {k: v.copy() for k, v in p.items()}
    for _ in range(steps):
        idx = rng.integers(0, len(y), batch)
        g, _ = grads(p, X[idx], y[idx])
        for k in p:
            p[k] -= lr * g[k].astype(np.float32)
    return p


def accuracy(p, X, y):
    return float((forward(p, X)[0].argmax(1) == y).mean())
