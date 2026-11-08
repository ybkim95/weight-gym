"""pytest tests/  (or: python tests/test_smoke.py)"""
import weight_gym as wg
from weight_gym import agent, tasks
from weight_gym.env import BudgetExceeded

FIRST = wg.list_tasks()[0]["id"]


def test_access_and_budget():
    bb = wg.make(FIRST, "blackbox")
    try:
        bb.weights()
        raise AssertionError("blackbox exposed weights")
    except PermissionError:
        pass
    try:
        bb.finetune(10_000)
        raise AssertionError("budget not enforced")
    except BudgetExceeded:
        pass
    env = wg.make(FIRST, "internal")
    w = env.weights()
    assert w["W1"].shape == (144, 64)
    try:
        env.set_weights(W1=w["W1"] * 2)  # 9216 changed entries
        raise AssertionError("edit budget not enforced")
    except BudgetExceeded:
        pass
    env.set_weights(b1=w["b1"] + 1)
    assert env.used["edit_params"] == 64
    env.reset()
    assert env.used["edit_params"] == 0


def test_agent_loop_and_sandbox():
    replies = iter(["```python\nprint(env._p)\n```",
                    "```python\nprint(env._task)\n```",
                    "```python\nnp.load('tasks/index.json')\n```",
                    "no code",
                    "```python\nprint('eval:', type(1), np.zeros(3), np.float32(1.5))\n```",
                    "```python\nX, y = env.repair_data()\nprint(env.evaluate())\nenv.submit()\n```"])
    env = wg.make(FIRST, "internal")
    tr = agent.run(env, lambda m: next(replies))
    assert all(tr[i]["output"].startswith("Rejected") for i in (0, 1, 2)), tr[:3]
    assert tr[3]["output"].startswith("No ```python"), tr[3]
    assert tr[4]["output"].startswith("eval: <class 'int'> [0. 0. 0.] 1.5"), tr[4]
    assert float(tr[5]["output"]) > 0 and env.done, tr[5]


def test_hint_and_grade():
    for t in wg.list_tasks(None):
        assert t["fault"] in tasks.hint(t)
        cls, pixels, units = tasks._truth_sets(t)
        env = wg.make(t["id"], "blackbox", hint=True)
        assert env.hint == tasks.hint(t)
        env.report(t["fault"], cls, sorted(pixels or []), sorted(units or []))
        g = wg.evaluate(env)["diagnosis"]
        assert g["type"] and g["cls"] in (True, None) and g["iou"] in (1.0, None), (t["id"], g)
    assert tasks.grade(None, t) == {"type": False, "cls": False, "iou": 0.0}


def test_evaluate_is_zero_for_untouched_model():
    r = wg.evaluate(wg.make(FIRST))
    assert abs(r["recovery"]) < 1e-9 and r["submitted"] is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
