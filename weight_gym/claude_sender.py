"""Claude sender for agent.run. Credentials: ANTHROPIC_API_KEY or an `ant auth login` profile, or,
with neither, Google Cloud Vertex AI via GOOGLE_CLOUD_PROJECT (+ GOOGLE_CLOUD_LOCATION, default global)."""
import os

import anthropic


def claude(model):
    if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("GOOGLE_CLOUD_PROJECT"):
        client = anthropic.AnthropicVertex(project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
                                           region=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"), max_retries=5)
    else:
        client = anthropic.Anthropic(max_retries=5)
    messages = []

    def send(m):
        messages.append({"role": "user", "content": m})
        r = client.messages.create(model=model, max_tokens=16000, messages=messages)
        send.usage["input"] += r.usage.input_tokens; send.usage["output"] += r.usage.output_tokens  # output includes thinking
        # append full content (thinking blocks included) so history stays append-only
        messages.append({"role": "assistant", "content": r.content})
        if r.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in r.content if b.type == "text")
    send.usage = {"input": 0, "output": 0}
    return send
