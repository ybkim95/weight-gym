"""Fault generators for the 2x2 design. Each task = a broken model + repair data the agent sees
+ a hidden deployment test set. 'visible' faults show up on the repair data; 'hidden' ones do not."""
import json
import os
from pathlib import Path

import numpy as np

from . import core

TASKS_DIR = Path(os.environ.get("WEIGHT_GYM_TASKS", Path(__file__).resolve().parent.parent / "tasks"))
N_TRAIN, N_REPAIR, N_TEST = 10000, 500, 2000
TRAIN_STEPS, TRAIN_LR = 6000, 0.1
MIN_GAP = 0.15  # fixed acceptance rule: ref must beat broken by this much on the test set
PER_FAULT = 5


def _train(X, y, rng):
    return core.sgd(core.init(rng), X, y, TRAIN_STEPS, TRAIN_LR, rng)


def label_flip(rng):
    src, dst = rng.choice(10, (2, 3), replace=False)
    X, y = core.sample(N_TRAIN, rng)
    y_noisy = y.copy()
    for a, b in zip(src, dst):
        y_noisy[(y == a) & (rng.random(len(y)) < 0.8)] = b
    rX, ry = core.sample(N_REPAIR, rng)
    tX, ty = core.sample(N_TEST, rng)
    return dict(broken=_train(X, y_noisy, rng), clean=(X, y), repair=(rX, ry), test=(tX, ty),
                truth={"flips": [[int(a), int(b)] for a, b in zip(src, dst)], "frac": 0.8})


def dead_units(rng):
    X, y = core.sample(N_TRAIN, rng)
    p = _train(X, y, rng)
    units = rng.choice(64, 40, replace=False)
    p["b1"][units] = -20.0
    rX, ry = core.sample(N_REPAIR, rng)
    tX, ty = core.sample(N_TEST, rng)
    return dict(broken=p, clean=(X, y), repair=(rX, ry), test=(tX, ty),
                truth={"layer": "b1", "units": sorted(int(u) for u in units)})


def weight_noise(rng):
    X, y = core.sample(N_TRAIN, rng)
    p = _train(X, y, rng)
    p["W1"] += rng.normal(0, 1.5 * p["W1"].std(), p["W1"].shape).astype(np.float32)
    rX, ry = core.sample(N_REPAIR, rng)
    tX, ty = core.sample(N_TEST, rng)
    return dict(broken=p, clean=(X, y), repair=(rX, ry), test=(tX, ty),
                truth={"layer": "W1", "noise_std_ratio": 1.5})


def _backdoor(rng, apply, truth):
    target = int(rng.integers(10))
    X, y = core.sample(N_TRAIN, rng)
    poison = rng.random(len(y)) < 0.1
    Xp, yp = X.copy(), y.copy()
    Xp[poison], yp[poison] = apply(X[poison]), target
    rX, ry = core.sample(N_REPAIR, rng)
    cX, cy = core.sample(N_TEST // 2, rng)
    tX, ty = core.sample(N_TEST, rng)
    tX, ty = tX[ty != target][:N_TEST // 2], ty[ty != target][:N_TEST // 2]
    test = (np.concatenate([cX, apply(tX)]), np.concatenate([cy, ty]))
    return dict(broken=_train(Xp, yp, rng), clean=(X, y), repair=(rX, ry), test=test,
                truth={**truth, "target_class": target, "poison_frac": 0.1})


def patch_backdoor(rng):
    r, c = (int(v) for v in rng.integers(0, core.SIDE - 2, 2))
    patch = (rng.random((3, 3)) < 0.5).astype(np.float32) * 1.5

    def apply(X):
        X = X.reshape(-1, core.SIDE, core.SIDE).copy()
        X[:, r:r + 3, c:c + 3] = patch
        return X.reshape(len(X), -1)
    return _backdoor(rng, apply, {"trigger": "3x3 patch", "row": r, "col": c, "patch": patch.tolist()})


def blend_backdoor(rng):
    pattern = (rng.choice([-1, 1], core.SIDE * core.SIDE) * 0.35).astype(np.float32)
    return _backdoor(rng, lambda X: X + pattern, {"trigger": "global blend pattern", "pattern": pattern.tolist()})


def spurious_marker(rng):
    k = int(rng.integers(10))
    pix = int(rng.integers(core.SIDE * core.SIDE))

    def mark(X, on):
        X = X.copy()
        X[on, pix] = 1.5
        return X
    X, y = core.sample(N_TRAIN, rng)
    rX, ry = core.sample(N_REPAIR, rng)
    tX, ty = core.sample(N_TEST, rng)
    # train and repair data: marker iff class k. deployment: marker on a random half, unrelated to class.
    return dict(broken=_train(mark(X, y == k), y, rng), clean=(X, y), repair=(mark(rX, ry == k), ry),
                test=(mark(tX, rng.random(len(ty)) < 0.5), ty),
                truth={"class": k, "pixel": pix, "row": pix // core.SIDE, "col": pix % core.SIDE})


FAULTS = {"visible": [label_flip, dead_units, weight_noise],
          "hidden": [patch_backdoor, blend_backdoor, spurious_marker]}


def build(out=None, per_fault=PER_FAULT, max_seeds=50):
    out = Path(out or TASKS_DIR)
    out.mkdir(exist_ok=True)
    index = []
    for row, fns in FAULTS.items():
        for fi, fn in enumerate(fns):
            kept, tried = 0, []
            for seed in range(max_seeds):
                if kept == per_fault:
                    break
                rng = np.random.default_rng([seed, fi, row == "hidden"])
                t = fn(rng)
                ref = _train(*t["clean"], rng)
                ref_s, broken_s = core.accuracy(ref, *t["test"]), core.accuracy(t["broken"], *t["test"])
                ok = ref_s - broken_s >= MIN_GAP
                tried.append({"seed": seed, "ref": ref_s, "broken": broken_s, "kept": ok})
                if not ok:
                    continue
                tid = f"{row}-{fn.__name__}-{kept}"
                np.savez_compressed(out / f"{tid}.npz", **t["broken"], repair_X=t["repair"][0],
                                    repair_y=t["repair"][1], test_X=t["test"][0], test_y=t["test"][1])
                index.append({"id": tid, "row": row, "fault": fn.__name__, "seed": seed,
                              "ref_score": ref_s, "broken_score": broken_s, "truth": t["truth"]})
                kept += 1
            print(f"{row:7s} {fn.__name__:16s} kept {kept}/{len(tried)} tried "
                  + " ".join(f"{d['ref']:.2f}/{d['broken']:.2f}{'' if d['kept'] else 'x'}" for d in tried))
            assert kept == per_fault, f"{fn.__name__}: only {kept} tasks pass MIN_GAP"
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return index


def _truth_sets(task):
    """(class, pixels, units) the fault is located at; None where it does not apply."""
    t, f = task["truth"], task["fault"]
    if f == "patch_backdoor":
        return t["target_class"], {(t["row"] + i) * core.SIDE + t["col"] + j for i in range(3) for j in range(3)}, None
    if f == "blend_backdoor":
        return t["target_class"], None, None
    if f == "spurious_marker":
        return t["class"], {t["pixel"]}, None
    if f == "dead_units":
        return None, None, set(t["units"])
    return None, None, None


def hint(task):
    """Ground-truth fault type and location, for the hint condition."""
    t, f, C = task["truth"], task["fault"], core.CLASSES
    cls, pixels, units = _truth_sets(task)
    return {
        "label_flip": lambda: "labels flipped in training (80% of each source class): "
                              + ", ".join(f"{C[a]}->{C[b]}" for a, b in t["flips"]),
        "dead_units": lambda: f"hidden units {sorted(units)} never activate",
        "weight_noise": lambda: "W1 has added Gaussian noise with std 1.5x the original W1 std",
        "patch_backdoor": lambda: f"a 3x3 trigger patch at rows {t['row']}-{t['row'] + 2}, cols {t['col']}-{t['col'] + 2} "
                                  f"(pixels {sorted(pixels)}) makes the model predict class {cls} ({C[cls]})",
        "blend_backdoor": lambda: f"adding this 144-value pattern to an image makes the model predict class {cls} "
                                  f"({C[cls]}): {[round(v, 2) for v in t['pattern']]}",
        # location only: saying how deployment differs would hand over more than the backdoor hints do
        "spurious_marker": lambda: f"pixel {t['pixel']} (row {t['row']}, col {t['col']}) was 1.5 on every training image "
                                   f"of class {cls} ({C[cls]}), and the model relies on it",
    }[f]() + f". Fault type: {f}."


def grade(diagnosis, task):
    """Diagnosis score: type match, class match, IoU of pixels/units with the truth (None = not applicable)."""
    if diagnosis is None:
        return {"type": False, "cls": False, "iou": 0.0}
    cls, pixels, units = _truth_sets(task)
    truth = pixels if pixels is not None else units
    got = set(diagnosis["pixels"] if pixels is not None else diagnosis["units"])
    return {"type": diagnosis["fault"] == task["fault"],
            "cls": None if cls is None else diagnosis["cls"] == cls,
            "iou": None if truth is None else len(truth & got) / len(truth | got)}


def load(tid, root=None):
    d = np.load(Path(root or TASKS_DIR) / f"{tid}.npz")
    return ({k: d[k] for k in ("W1", "b1", "W2", "b2")}, (d["repair_X"], d["repair_y"]), (d["test_X"], d["test_y"]))
