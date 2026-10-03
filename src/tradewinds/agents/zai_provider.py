"""Z.ai Chat Completions protocol; no SDK dependency or implicit retries.

Official contracts: https://docs.z.ai/guides/llm/glm-5.3 and
https://docs.z.ai/guides/capabilities/thinking-mode . Reasoning is retained
verbatim in conversation state and is never interpreted as a tool call.
"""

from __future__ import annotations
import json
import math
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from .credentials import zai_api_key
from .campaign_budget import campaign_dispatch, reserve_from_environment

ENDPOINT = "https://api.z.ai/api/paas/v4/chat/completions"


def chat_messages(messages, system=None):
    """Translate the internal text/tool blocks without changing ids or reasoning."""
    result = [{"role": "system", "content": system}] if system else []
    for message in messages:
        role, content = message["role"], message["content"]
        if isinstance(content, str):
            item = {"role": role, "content": content}
            if "reasoning_content" in message:
                item["reasoning_content"] = message["reasoning_content"]
            result.append(item)
            continue
        if not isinstance(content, list):
            raise ValueError("message content must be text or typed blocks")
        text, calls, reasoning = [], [], []
        tool_results = []
        for block in content:
            kind = block.get("type")
            if kind == "text":
                text.append(block["text"])
            elif kind == "tool_use":
                calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"], allow_nan=False),
                        },
                    }
                )
            elif kind == "tool_result":
                value = block["content"]
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": value if isinstance(value, str) else json.dumps(value, allow_nan=False),
                    }
                )
            elif kind == "reasoning":
                reasoning.append(block["reasoning_content"])
            else:
                raise ValueError("unsupported message block type")
        if role == "assistant":
            if tool_results:
                raise ValueError("assistant cannot contain tool results")
            item = {"role": "assistant", "content": "\n".join(text) or None}
            if calls:
                item["tool_calls"] = calls
            preserved = message.get("reasoning_content", "".join(reasoning))
            if preserved:
                item["reasoning_content"] = preserved
            result.append(item)
        else:
            if calls or reasoning:
                raise ValueError("tool calls and reasoning require assistant role")
            result.extend(tool_results)
            if text:
                result.append({"role": role, "content": "\n".join(text)})
    return result


def request_body(model, effort, messages, tools, max_tokens, system=None):
    return {
        "model": model,
        "messages": chat_messages(messages, system),
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ],
        "max_tokens": max_tokens,
        "stream": False,
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": effort,
    }


def send_request(body, timeout):
    key = zai_api_key()
    if not key:
        raise ValueError("ZAI_API_KEY is required")
    request = Request(
        os.environ.get("ZAI_API_ENDPOINT", ENDPOINT),
        data=json.dumps(body, allow_nan=False).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with campaign_dispatch():
        ledger, request_id = reserve_from_environment(body)
        try:
            # One request, no implicit retry; ambiguous failures retain allowance.
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
            if ledger is not None:
                usage = dict(payload.get("usage") or {})
                if isinstance(payload.get("id"), str):
                    usage["provider_response_id"] = payload["id"]
                ledger.record_usage(request_id, usage)
            return parse_response(payload)
        except BaseException as error:
            if ledger is not None:
                ledger.mark_uncertain(request_id)
            if isinstance(error, HTTPError):
                attach_safe_error_detail(error, key)
            raise


def attach_safe_error_detail(error, key):
    """Retain bounded provider code/message, never authorization or raw payload."""
    def safe(value):
        return str(value).replace(key, "[REDACTED]")[:512]
    detail = {"http_status": error.code}
    try:
        payload = json.loads(error.read(4096))
        nested = payload.get("error", payload) if isinstance(payload, dict) else {}
        if isinstance(nested, dict):
            for field in ("code", "message"):
                if isinstance(nested.get(field), (str, int, float)):
                    detail[field] = safe(nested[field])
    except (ValueError, OSError, TypeError, AttributeError):
        pass
    error.provider_error_detail = detail
    error.msg = safe(error.reason)
    if "code" in detail or "message" in detail:
        error.msg += "; provider=" + json.dumps(detail, ensure_ascii=True)


def parse_response(payload):
    if not isinstance(payload, dict) or payload.get("error"):
        raise ValueError("Z.ai returned an error or malformed response")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Z.ai response must contain exactly one choice")
    choice = choices[0]
    if choice.get("finish_reason") not in ("stop", "tool_calls"):
        raise ValueError("Z.ai response was incomplete or filtered")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("invalid Z.ai assistant message")
    text, reasoning = message.get("content") or "", message.get("reasoning_content") or ""
    if not isinstance(text, str) or not isinstance(reasoning, str):
        raise ValueError("Z.ai text and reasoning must be strings")
    calls, ids = [], set()
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise ValueError("invalid Z.ai tool calls")
    for call in tool_calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ValueError("unsupported Z.ai tool call type")
        function, call_id = call.get("function"), call.get("id")
        if not isinstance(function, dict) or not isinstance(call_id, str) or not call_id or call_id in ids:
            raise ValueError("invalid or duplicate Z.ai tool call id")
        name, arguments = function.get("name"), function.get("arguments")
        if not isinstance(name, str) or not name or not isinstance(arguments, str):
            raise ValueError("invalid Z.ai function call")
        args = json.loads(
            arguments,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite tool argument")),
        )
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be a JSON object")
        json.dumps(args, allow_nan=False)
        ids.add(call_id)
        calls.append({"id": call_id, "name": name, "arguments": args})
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("Z.ai usage is required for accounting")
    counts = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("invalid Z.ai token usage")
    return {
        "calls": calls,
        "text": text,
        "reasoning_content": reasoning,
        "input_tokens": counts[0],
        "output_tokens": counts[1],
    }


def validate_settings(model, effort, timeout):
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model is required")
    if effort not in ("low", "high", "max"):
        raise ValueError("reasoning_effort must be low, high or max")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if not zai_api_key():
        raise ValueError("ZAI_API_KEY is required")


class ZaiClient:
    """The investigation runtime owns durable request reservations and usage."""

    name = "zai"

    def __init__(self, model="glm-5.3", reasoning_effort="high", timeout=60.0):
        validate_settings(model, reasoning_effort, timeout)
        self.model, self.reasoning_effort, self.timeout = model, reasoning_effort, timeout

    def _body(self, messages, tools, max_tokens):
        from .provider import SYSTEM

        return request_body(self.model, self.reasoning_effort, messages, tools, max_tokens, SYSTEM)

    def count_input(self, messages, tools):
        # Conservative UTF-8 byte bound plus chat/template overhead; no billed count call.
        return len(json.dumps(self._body(messages, tools, 8192), allow_nan=False).encode()) + 4096

    def complete(self, messages, tools, limits):
        rates = (limits.spend_ceiling_usd, limits.input_usd_per_million, limits.output_usd_per_million)
        if any(not math.isfinite(value) or value <= 0 for value in rates):
            raise ValueError("Live calls require a finite positive cap and explicit token rates")
        if type(limits.max_output_tokens) is not int or not 1 <= limits.max_output_tokens <= 131072:
            raise ValueError("invalid output token limit")
        return send_request(self._body(messages, tools, limits.max_output_tokens), self.timeout)
