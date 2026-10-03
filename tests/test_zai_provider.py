import json
from types import SimpleNamespace
import pytest

import tradewinds.agents.zai_provider as zai

TOOLS = [{"name": "inspect", "description": "Inspect evidence", "input_schema": {"type": "object"}}]
MESSAGES = [{"role": "user", "content": "Inspect this"}]


def response(arguments="{}"):
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "private model trace\n保留",
                    "tool_calls": [
                        {
                            "id": "call-7",
                            "type": "function",
                            "function": {"name": "inspect", "arguments": arguments},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


class Reply:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps(self.value).encode()


def test_http_protocol_roundtrip(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-placeholder")
    requests = []

    def fake(request, timeout):
        requests.append((json.loads(request.data), timeout, request.full_url))
        return Reply(response())

    monkeypatch.setattr(zai, "urlopen", fake)
    body = zai.request_body("glm-5.3", "high", MESSAGES, TOOLS, 1500)
    result = zai.send_request(body, 7.0)
    assert result["input_tokens"] == 100 and result["output_tokens"] == 50
    assert requests[0][0]["thinking"] == {"type": "enabled", "clear_thinking": False}
    assert requests[0][0]["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert requests[0][1:] == (7.0, zai.ENDPOINT)
    messages = MESSAGES + [
        {
            "role": "assistant",
            "content": [
                {"type": "reasoning", "reasoning_content": result["reasoning_content"]},
                {"type": "tool_use", "id": "call-7", "name": "inspect", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call-7", "content": '{"ok":true}'}],
        },
    ]
    zai.send_request(zai.request_body("glm-5.3", "low", messages, TOOLS, 1500), 7.0)
    wire = requests[1][0]["messages"]
    assert wire[1]["reasoning_content"] == result["reasoning_content"]
    assert wire[1]["tool_calls"][0]["id"] == "call-7"
    assert wire[2] == {"role": "tool", "tool_call_id": "call-7", "content": '{"ok":true}'}
    assert len(wire[1]["tool_calls"]) == 1


@pytest.mark.parametrize("arguments", ["{broken", "[]", '{"a":NaN}', '{"a":1e999}'])
def test_malformed_arguments_rejected(arguments):
    with pytest.raises(ValueError):
        zai.parse_response(response(arguments))


def test_usage_required_and_truncation_rejected():
    payload = response()
    payload["usage"]["completion_tokens"] = -1
    with pytest.raises(ValueError):
        zai.parse_response(payload)
    payload = response()
    payload["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError):
        zai.parse_response(payload)


def test_top_level_reasoning_and_unknown_blocks():
    value = zai.chat_messages(
        [
            {
                "role": "assistant",
                "reasoning_content": "verbatim\n",
                "content": [{"type": "tool_use", "id": "x", "name": "inspect", "input": {}}],
            }
        ]
    )
    assert value[0]["reasoning_content"] == "verbatim\n"
    with pytest.raises(ValueError):
        zai.chat_messages([{"role": "assistant", "content": [{"type": "unexpected", "name": "inspect"}]}])


def test_timeout_and_malformed_json(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-placeholder")

    def timeout(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(zai, "urlopen", timeout)
    with pytest.raises(TimeoutError):
        zai.send_request({}, 1.0)

    class Malformed(Reply):
        def read(self):
            return b"not json"

    monkeypatch.setattr(zai, "urlopen", lambda *a, **kw: Malformed(None))
    with pytest.raises(ValueError):
        zai.send_request({}, 1.0)


def test_client_runtime_protocol(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-placeholder")
    client = zai.ZaiClient()
    limits = SimpleNamespace(
        spend_ceiling_usd=1.0, input_usd_per_million=1.0, output_usd_per_million=1.0, max_output_tokens=1500
    )
    body = client._body(MESSAGES, TOOLS, 1500)
    assert body["messages"][0]["role"] == "system"
    assert client.count_input(MESSAGES, TOOLS) >= len(json.dumps(body).encode())
    monkeypatch.setattr(zai, "send_request", lambda *a: zai.parse_response(response()))
    result = client.complete(MESSAGES, TOOLS, limits)
    assert result["calls"][0]["id"] == "call-7"
    assert result["reasoning_content"] == "private model trace\n保留"
    limits.spend_ceiling_usd = float("inf")
    with pytest.raises(ValueError):
        client.complete(MESSAGES, TOOLS, limits)
