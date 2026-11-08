"""Claude sender for agent.run. Credentials: ANTHROPIC_API_KEY or an `ant auth login` profile."""
import anthropic


def claude(model):
    client = anthropic.Anthropic(max_retries=5)
    messages = []

    def send(m):
        messages.append({"role": "user", "content": m})
        r = client.messages.create(model=model, max_tokens=16000, messages=messages)
        # append full content (thinking blocks included) so history stays append-only
        messages.append({"role": "assistant", "content": r.content})
        if r.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in r.content if b.type == "text")
    return send
