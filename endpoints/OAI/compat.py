"""
Compat dialects: Anthropic Messages (/v1/messages) and OpenAI Responses
(/v1/responses), text-only v1.

Lifted event semantics from the Strata project (MIT), adapted to this server's
pipeline: requests are translated to ChatCompletionRequest and run through the
existing chat-completion path, so sampling, caching, tools-free templating and
disconnect handling behave identically to /v1/chat/completions.

v1 limitations (documented, honest): text content only. tools/tool_use/
tool_result blocks, images and previous_response_id are rejected with 400.
"""

import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette import EventSourceResponse

from common import model
from common.auth import check_api_key
from common.model import check_model_container
from common.networking import DisconnectHandler, get_sse_ping_interval, request_tag
from common.tabby_config import config
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.utils.chat_completion import (
    apply_chat_template,
    generate_chat_completion,
    stream_generate_chat_completion,
)

from endpoints.OAI.router import load_lock

router = APIRouter()

STOP_MAP = {"stop": "end_turn", "length": "max_tokens", "eos": "end_turn"}


def _err(status: int, message: str, typ: str = "invalid_request_error") -> HTTPException:
    return HTTPException(status, {"type": "error", "error": {"type": typ, "message": message}})


def _blocks_to_text(content, where: str) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if not isinstance(b, dict):
                raise _err(400, f"{where}: block must be an object")
            bt = b.get("type")
            if bt == "text":
                out.append(b.get("text", ""))
            else:
                raise _err(400, f"{where}: block type {bt!r} unsupported (text-only v1)")
        return "".join(out)
    raise _err(400, f"{where}: content must be a string or text-block list")


def anthropic_to_messages(body: dict) -> list:
    messages = []
    system = body.get("system")
    if system is not None:
        messages.append({"role": "system", "content": _blocks_to_text(system, "system")})
    for i, m in enumerate(body.get("messages") or []):
        role = m.get("role")
        if role not in ("user", "assistant"):
            raise _err(400, f"messages[{i}]: role {role!r} unsupported (tool results are v1.1)")
        messages.append({"role": role, "content": _blocks_to_text(m.get("content"), f"messages[{i}]")})
    if not any(m["role"] == "user" for m in messages):
        raise _err(400, "messages: at least one user message required")
    return messages


def responses_to_messages(body: dict) -> list:
    if body.get("previous_response_id"):
        raise _err(400, "this server keeps no responses (stateless); send the whole conversation each call")
    inp = body.get("input", "")
    if isinstance(inp, str):
        items = [{"role": "user", "content": inp}]
    elif isinstance(inp, list):
        items = inp
    else:
        raise _err(400, "input must be a string or item list")
    messages = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": str(body["instructions"])})
    for i, it in enumerate(items):
        if isinstance(it, str):
            messages.append({"role": "user", "content": it})
            continue
        role = it.get("role", "user")
        if role not in ("user", "assistant", "system", "developer"):
            raise _err(400, f"input[{i}]: role {role!r} unsupported (function calls are v1.1)")
        if role == "developer":
            role = "system"
        messages.append({"role": role, "content": _blocks_to_text(it.get("content", ""), f"input[{i}]")})
    return messages


def build_request(body: dict, messages: list, max_tokens: int) -> ChatCompletionRequest:
    kwargs = {
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": bool(body.get("stream")),
    }
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p")):
        if body.get(src) is not None:
            kwargs[dst] = body[src]
    if body.get("stop_sequences"):
        kwargs["stop"] = body["stop_sequences"]
    if body.get("model"):
        kwargs["model"] = body["model"]
    if body.get("chat_template_kwargs"):
        kwargs["chat_template_kwargs"] = body["chat_template_kwargs"]
    if kwargs["stream"]:
        kwargs["stream_options"] = {"include_usage": True}
    return ChatCompletionRequest(**kwargs)


def _chunk_dict(item):
    if item == "[DONE]":
        return None
    if isinstance(item, str):
        try:
            return json.loads(item)
        except ValueError:
            return None
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, dict):
        d = item.get("data")
        if isinstance(d, str):
            try:
                return json.loads(d)
            except ValueError:
                return None
        return item
    return None


async def _run(data: ChatCompletionRequest, request: Request):
    """Shared pipeline: template -> generate; returns (prompt, response-or-None)."""
    from endpoints.OAI.utils.common_ import load_inline_model

    note = getattr(model.container, "note_incoming_request", None)
    if note is not None:
        note()
    async with load_lock:
        if data.model:
            await load_inline_model(data.model, request)
        await check_model_container()
    if model.container.prompt_template is None:
        raise _err(422, "chat dialects require a prompt template", "server_error")
    prompt, mm_embeddings = await apply_chat_template(data)
    return prompt, mm_embeddings


def _usage_from(resp) -> dict:
    u = resp.usage
    cached = (getattr(u, "prompt_tokens_details", None) or {})
    if not isinstance(cached, dict):
        cached = getattr(cached, "model_dump", lambda: {})()
    return {
        "input_tokens": u.prompt_tokens,
        "cache_read_input_tokens": int(cached.get("cached_tokens") or 0),
        "output_tokens": u.completion_tokens,
    }


# ---------------------------------------------------------------- Anthropic

@router.post("/v1/messages", dependencies=[Depends(check_api_key)])
async def anthropic_messages(request: Request):
    body = await request.json()
    if body.get("tools"):
        raise _err(400, "tools are not supported by this endpoint (text-only v1)")
    if not body.get("max_tokens"):
        raise _err(400, "max_tokens is required")
    messages = anthropic_to_messages(body)
    data = build_request(body, messages, int(body["max_tokens"]))
    model_name = body.get("model", "")

    if data.stream and not config.developer.disable_request_streaming:
        prompt, mm = await _run(data, request)
        disconnect_handler = DisconnectHandler(request, f"{request_tag(request)} messages")
        inner = stream_generate_chat_completion(
            prompt, mm, data, request, model.container.model_dir, disconnect_handler
        )
        return EventSourceResponse(
            _anthropic_stream(inner, model_name), ping=get_sse_ping_interval()
        )

    prompt, mm = await _run(data, request)
    disconnect_handler = DisconnectHandler(request, f"{request_tag(request)} messages")
    resp = await generate_chat_completion(
        prompt, mm, data, request, model.container.model_dir, disconnect_handler
    )
    choice = resp.choices[0]
    text = choice.message.content or ""
    usage = _usage_from(resp)
    return {
        "id": "msg_" + uuid.uuid4().hex[:24],
        "type": "message",
        "role": "assistant",
        "model": model_name,
        "content": ([{"type": "thinking", "thinking": choice.message.reasoning_content, "signature": ""}]
                    if getattr(choice.message, "reasoning_content", None) else [])
        + [{"type": "text", "text": text}],
        "stop_reason": STOP_MAP.get(choice.finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": usage,
    }


async def _anthropic_stream(inner, model_name) -> AsyncGenerator[dict, None]:
    mid = "msg_" + uuid.uuid4().hex[:24]
    def ev(name: str, payload: dict) -> dict:
        return {"event": name, "data": json.dumps(payload, ensure_ascii=False)}

    yield ev("message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": model_name,
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}}})
    yield ev("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}})
    finish, usage = "end_turn", {"input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0}
    async for item in inner:
        chunk = _chunk_dict(item)
        if not chunk:
            continue
        if chunk.get("usage"):
            u = chunk["usage"]
            cached = u.get("prompt_tokens_details") or {}
            usage = {"input_tokens": u.get("prompt_tokens", 0) - int(cached.get("cached_tokens") or 0),
                     "cache_read_input_tokens": int(cached.get("cached_tokens") or 0),
                     "output_tokens": u.get("completion_tokens", 0)}
        for ch in chunk.get("choices") or []:
            text = (ch.get("delta") or {}).get("content")
            if text:
                yield ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                 "delta": {"type": "text_delta", "text": text}})
            if ch.get("finish_reason"):
                finish = STOP_MAP.get(ch["finish_reason"], "end_turn")
    yield ev("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield ev("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": finish, "stop_sequence": None},
                               "usage": usage})
    yield ev("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------- Responses

@router.post("/v1/responses", dependencies=[Depends(check_api_key)])
async def responses_create(request: Request):
    body = await request.json()
    if body.get("tools"):
        raise _err(400, "tools are not supported by this endpoint (text-only v1)")
    messages = responses_to_messages(body)
    max_tokens = int(body.get("max_output_tokens") or 1024)
    data = build_request(body, messages, max_tokens)
    model_name = body.get("model", "")

    if data.stream and not config.developer.disable_request_streaming:
        prompt, mm = await _run(data, request)
        disconnect_handler = DisconnectHandler(request, f"{request_tag(request)} responses")
        inner = stream_generate_chat_completion(
            prompt, mm, data, request, model.container.model_dir, disconnect_handler
        )
        return EventSourceResponse(
            _responses_stream(inner, model_name), ping=get_sse_ping_interval()
        )

    prompt, mm = await _run(data, request)
    disconnect_handler = DisconnectHandler(request, f"{request_tag(request)} responses")
    resp = await generate_chat_completion(
        prompt, mm, data, request, model.container.model_dir, disconnect_handler
    )
    choice = resp.choices[0]
    text = choice.message.content or ""
    rid = "resp_" + uuid.uuid4().hex[:24]
    usage = _usage_from(resp)
    return {
        "id": rid, "object": "response",
        "created_at": int(time.time()),
        "status": "incomplete" if choice.finish_reason == "length" else "completed",
        "model": model_name,
        "output": [{"type": "message", "id": "msg_" + uuid.uuid4().hex[:24], "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]}],
        "usage": {"input_tokens": usage["input_tokens"] + usage["cache_read_input_tokens"],
                  "output_tokens": usage["output_tokens"]},
    }


async def _responses_stream(inner, model_name) -> AsyncGenerator[dict, None]:
    rid = "resp_" + uuid.uuid4().hex[:24]
    def ev(name: str, payload: dict) -> dict:
        return {"event": name, "data": json.dumps(payload, ensure_ascii=False)}

    base = {"id": rid, "object": "response", "created_at": int(time.time()),
            "status": "in_progress", "model": model_name, "output": []}
    yield ev("response.created", {"type": "response.created", "response": dict(base)})
    yield ev("response.in_progress", {"type": "response.in_progress", "response": dict(base)})
    yield ev("response.output_item.added", {"type": "response.output_item.added", "output_index": 0,
           "item": {"type": "message", "id": "msg_" + uuid.uuid4().hex[:24], "role": "assistant",
                    "status": "in_progress", "content": []}})
    yield ev("response.content_part.added", {"type": "response.content_part.added",
           "item_id": "x", "output_index": 0, "content_index": 0,
           "part": {"type": "output_text", "text": "", "annotations": []}})
    text, finish, usage = [], "completed", {"input_tokens": 0, "output_tokens": 0}
    async for item in inner:
        chunk = _chunk_dict(item)
        if not chunk:
            continue
        if chunk.get("usage"):
            u = chunk["usage"]
            usage = {"input_tokens": u.get("prompt_tokens", 0),
                     "output_tokens": u.get("completion_tokens", 0)}
        for ch in chunk.get("choices") or []:
            t = (ch.get("delta") or {}).get("content")
            if t:
                text.append(t)
                yield ev("response.output_text.delta", {"type": "response.output_text.delta",
                       "item_id": "x", "output_index": 0, "content_index": 0, "delta": t})
            if ch.get("finish_reason") == "length":
                finish = "incomplete"
    full = "".join(text)
    yield ev("response.output_text.done", {"type": "response.output_text.done",
           "item_id": "x", "output_index": 0, "content_index": 0, "text": full})
    yield ev("response.output_item.done", {"type": "response.output_item.done", "output_index": 0,
           "item": {"type": "message", "id": "x", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": full, "annotations": []}]}})
    done = dict(base)
    done["status"] = finish
    done["usage"] = usage
    yield ev("response.completed" if finish == "completed" else "response.incomplete",
             {"type": "response.completed" if finish == "completed" else "response.incomplete",
              "response": done})