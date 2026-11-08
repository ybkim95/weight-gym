"""Code-act agent loop: the LLM writes one python block per turn, run against `env`."""
import builtins
import io
import re
from contextlib import redirect_stdout

import numpy as np

from . import core
from .env import BUDGET, FAULT_TYPES

API = {
    "blackbox": """env.info(), env.repair_data() -> (X, y), env.predict(X) -> logits, env.evaluate() -> repair-set acc,
env.finetune(steps, lr=0.05, X=None, y=None), env.reset(), env.report(...), env.submit().
The model's weights are NOT visible.""",
    "internal": """env.info(), env.repair_data() -> (X, y), env.predict(X) -> logits, env.evaluate() -> repair-set acc,
env.finetune(steps, lr=0.05, X=None, y=None), env.reset(), env.report(...), env.submit(),
env.weights() -> dict W1 (144,64), b1 (64,), W2 (64,10), b2 (10,)  [h = relu(x@W1+b1); logits = h@W2+b2],
env.set_weights(**arrays) [each changed entry counts against edit_params; env.reset() refills it], env.activations(X) -> h (n,64), env.gradients(X, y) -> dict of grads + 'loss'.""",
}

PROMPT = """{info}

You act by writing exactly one ```python``` block per turn. It runs in a persistent namespace with
`env`, `np` (numpy) and `CLASSES` defined; whatever you print comes back to you (last 4000 chars).
No imports, no file access, no names starting with an underscore. You have {turns} turns.
Form a hypothesis about what is wrong, test it, repair, check, then call env.submit().
Before submitting, call env.report(fault, cls=None, pixels=None, units=None) with your diagnosis:
fault is one of the keys below, cls the affected/target class index, pixels the input pixel indices
(0..143) and units the hidden unit indices (0..63) involved, where they apply.
{faults}

API:
{api}{hint}"""

SAFE_BUILTINS = {k: getattr(builtins, k) for k in (
    "print range len min max sum abs sorted enumerate zip list dict tuple set float int round str bool type slice "
    "isinstance any all map filter reversed iter next divmod pow repr format "
    "Exception ValueError TypeError KeyError IndexError ZeroDivisionError RuntimeError").split()}


def _import(name, *args, **kwargs):
    # numpy's C code imports lazily through the calling frame's builtins (e.g. to print arrays);
    # allow only modules that are already loaded, so no new code can come in.
    import sys
    if name.split(".")[0] not in sys.modules:
        raise ImportError(name)
    return builtins.__import__(name, *args, **kwargs)


SAFE_BUILTINS["__import__"] = _import
# shortcut: in-process sandbox, a determined agent could still escape; run agent code in a subprocess before a public leaderboard.
FORBIDDEN = re.compile(r"\b_|\bimport\b"
                       r"|\b(globals|locals|getattr|setattr|delattr|vars|dir|open|eval|exec|compile|breakpoint|input)\s*\("
                       r"|\.(load\w*|save\w*|fromfile|tofile|genfromtxt|memmap|lib|ctypeslib|f2py|testing)\b")
CODE = re.compile(r"```python\n(.*?)```", re.S)


def execute(code, ns):
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            exec(code, ns)
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
    return buf.getvalue()[-4000:] or "(no output)"


def run(env, send, max_turns=15, hint=None):
    """send(message) -> reply text, one chat per episode. hint: ground-truth diagnosis to give the agent."""
    ns = {"env": env, "np": np, "CLASSES": core.CLASSES, "__builtins__": SAFE_BUILTINS}
    msg = PROMPT.format(info=env.info(), turns=max_turns, api=API[env.access],
                        faults="\n".join(f"  {k}: {v}" for k, v in FAULT_TYPES.items()),
                        hint=f"\n\nDiagnosis (ground truth, given to you): {hint}" if hint else "")
    transcript = []
    for turn in range(max_turns):
        try:
            reply = send(msg)
        except Exception as e:  # e.g. context overflow: end this episode, keep the run going
            transcript.append({"turn": turn, "send_error": f"{type(e).__name__}: {str(e)[:300]}"})
            break
        m = CODE.search(reply)
        if m is None:
            out = "No ```python``` block found. Reply with exactly one python block."
        elif FORBIDDEN.search(m.group(1)):
            out = "Rejected: no imports, file access, or underscore names."
        else:
            out = execute(m.group(1), ns)
        transcript.append({"turn": turn, "reply": reply, "output": out})
        if env.done:
            break
        msg = f"Output:\n{out}\n\nBudget used {env.used} of {BUDGET}. Turns left: {max_turns - turn - 1}."
    return transcript


def openai_compat(model, base_url):
    """Any OpenAI-compatible server, e.g. vLLM serving an open model."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="EMPTY", max_retries=5, timeout=600)
    messages = []

    def send(m):
        messages.append({"role": "user", "content": m})
        text = client.chat.completions.create(model=model, messages=messages, max_tokens=8192).choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})
        return text
    return send


def gemini(model):
    """Uses GEMINI_API_KEY, or Vertex via GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION."""
    import time
    from google import genai
    client = genai.Client()
    chat = client.chats.create(model=model)

    def send(m, client=client):  # holding client keeps it from being closed by GC
        for attempt in range(4):
            try:
                return chat.send_message(m).text or ""
            except Exception as e:  # rate limits / transient 5xx
                if attempt == 3:
                    raise
                print(f"  retry after {type(e).__name__}: {str(e)[:120]}")
                time.sleep(10 * 2 ** attempt)
    return send
