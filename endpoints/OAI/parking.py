"""
p27 v1 — server-side conversation parking (tabbyAPI-08).

POST /v1/park holds a conversation (messages + the partial assistant text the
client saw) under a park_id and the job slot is already free (the client
disconnected). POST /v1/park/resume re-renders the conversation, appends the
partial text and runs it through the normal completion pipeline, so the engine
restores the prefix from the paged cache — measured on dot26 2026-10-04 at
3.7-5.5x vs cold and 98% prefix reuse (notes/p27-parking-feasibility-2026-10-04.md).

v1 boundaries (honest): parks live in process memory (lost on restart), LRU-capped,
and parked pages are NOT pinned against cache eviction — pinning needs engine hooks
(v2). Clients may of course still keep their own text; the handle just lets fleets
park without resending.
"""

import asyncio
import time
from collections import OrderedDict

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette import EventSourceResponse

from common import model
from common.auth import check_api_key
from common.model import check_model_container
from common.networking import DisconnectHandler, get_sse_ping_interval, request_tag
from common.tabby_config import config
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.types.completion import CompletionRequest
from endpoints.OAI.utils.chat_completion import apply_chat_template
from endpoints.OAI.utils.common_ import load_inline_model
from endpoints.OAI.utils.completion import (
    generate_completion,
    stream_generate_completion,
)

from endpoints.OAI.router import load_lock

router = APIRouter()

MAX_PARKS = 64
MAX_TOTAL_CHARS = 32 * 1024 * 1024
_parks: "OrderedDict[str, dict]" = OrderedDict()
_parks_lock = asyncio.Lock()


def _size(entry: dict) -> int:
    return len(entry["completion"]) + sum(len(str(m.get("content", ""))) for m in entry["messages"])


async def _evict_locked() -> None:
    while len(_parks) > MAX_PARKS or sum(_size(e) for e in _parks.values()) > MAX_TOTAL_CHARS:
        if not _parks:
            return
        _parks.popitem(last=False)


@router.post("/v1/park", dependencies=[Depends(check_api_key)])
async def park_conversation(request: Request):
    body = await request.json()
    messages = body.get("messages")
    if not isinstance(messages, list) or not any(
        m.get("role") == "user" for m in messages if isinstance(m, dict)
    ):
        raise HTTPException(422, "park requires messages with at least one user turn")
    completion = body.get("completion", "")
    if not isinstance(completion, str):
        raise HTTPException(422, "completion must be a string (the partial assistant text)")
    entry = {
        "messages": messages,
        "completion": completion,
        "model": body.get("model"),
        "created": time.time(),
    }
    park_id = "park_%012x" % (abs(hash((str(messages), completion, time.time_ns()))) % (16 ** 12))
    async with _parks_lock:
        _parks[park_id] = entry
        await _evict_locked()
    return {"park_id": park_id, "chars": _size(entry), "parks": len(_parks)}


@router.get("/v1/park", dependencies=[Depends(check_api_key)])
async def list_parks():
    async with _parks_lock:
        return {
            "parks": [
                {"park_id": k, "chars": _size(e), "created": e["created"]}
                for k, e in _parks.items()
            ]
        }


@router.delete("/v1/park/{park_id}", dependencies=[Depends(check_api_key)])
async def delete_park(park_id: str):
    async with _parks_lock:
        if _parks.pop(park_id, None) is None:
            raise HTTPException(404, f"no such park: {park_id}")
    return {"deleted": park_id}


@router.post("/v1/park/resume", dependencies=[Depends(check_api_key)])
async def resume_park(request: Request):
    body = await request.json()
    park_id = body.get("park_id")
    async with _parks_lock:
        entry = _parks.get(park_id)
        if entry is not None:
            _parks.move_to_end(park_id)
    if entry is None:
        raise HTTPException(404, f"no such park: {park_id}")

    data = ChatCompletionRequest(messages=entry["messages"], max_tokens=1)
    async with load_lock:
        if entry.get("model"):
            await load_inline_model(entry["model"], request)
        await check_model_container()
    if model.container.prompt_template is None:
        raise HTTPException(422, "parking requires a prompt template")
    prompt, _mm = await apply_chat_template(data)
    prompt = prompt + entry["completion"]

    comp = CompletionRequest(
        prompt=prompt,
        max_tokens=int(body.get("max_tokens") or 512),
        temperature=body.get("temperature", 0.7),
        top_p=body.get("top_p") if body.get("top_p") is not None else 0.9,
        stream=bool(body.get("stream")),
    )
    if body.get("stop"):
        comp.stop = body["stop"]

    if comp.stream and not config.developer.disable_request_streaming:
        disconnect_handler = DisconnectHandler(request, f"{request_tag(request)} park/resume")
        return EventSourceResponse(
            stream_generate_completion(
                prompt, comp, request, model.container.model_dir, disconnect_handler
            ),
            ping=get_sse_ping_interval(),
        )
    disconnect_handler = DisconnectHandler(request, f"{request_tag(request)} park/resume")
    return await generate_completion(
        prompt, comp, request, model.container.model_dir, disconnect_handler
    )