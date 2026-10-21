import asyncio
from asyncio import CancelledError, InvalidStateError

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from common import model
from common.auth import check_api_key
from common.debug_requests import (
    log_chat_completion_request,
    write_chat_completion_prompt_log,
)
from common.model import check_embeddings_container, check_model_container
from common.networking import (
    get_sse_ping_interval,
    handle_request_error,
    request_tag,
    DisconnectHandler,
    run_with_request_disconnect,
)
from common.tabby_config import config
from common.logger import xlogger
from endpoints.OAI.types.completion import CompletionRequest, CompletionResponse
from endpoints.OAI.types.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from endpoints.OAI.types.embedding import EmbeddingsRequest, EmbeddingsResponse
from endpoints.OAI.utils.common_ import load_inline_model
from endpoints.OAI.utils.chat_completion import (
    apply_chat_template,
    generate_chat_completion,
    stream_generate_chat_completion,
)
from endpoints.OAI.utils.completion import (
    generate_completion,
    stream_generate_completion,
)
from endpoints.OAI.utils.embeddings import get_embeddings


api_name = "OAI"
router = APIRouter()
urls = {
    "Completions": "http://{host}:{port}/v1/completions",
    "Chat completions": "http://{host}:{port}/v1/chat/completions",
}

# Block when model is still loading while second inline load request comes in
load_lock: asyncio.Lock = asyncio.Lock()


def setup():
    return router


# Completions endpoint
@router.post(
    "/v1/completions",
    dependencies=[Depends(check_api_key)],
)
async def completion_request(request: Request, data: CompletionRequest) -> CompletionResponse:
    """
    Generates a completion from a prompt.

    If stream = true, this returns an SSE stream.
    """

    raw_json = await request.json()
    xlogger.debug("[ENDPOINT] /v1/completions", {"raw": raw_json})

    async with load_lock:
        if data.model:
            await load_inline_model(data.model, request)
        await check_model_container()
        model_path = model.container.model_dir

    # Prepare raw prompt (will be str or list[str])
    prompt = data.prompt

    # Any JSON object, no particular schema. "json_object" is OpenAI's name for it
    if data.response_format.type in ("json_object", "json"):
        data.json_schema = {"type": "object"}

    # Also accept specific schema from response_format
    if data.response_format.type == "json_schema":
        data.json_schema = data.response_format.json_schema

    try:
        disconnect_handler = DisconnectHandler(request, f"{request_tag(request)} completions")
        await disconnect_handler.poll()

        if data.stream and not config.developer.disable_request_streaming:
            model.check_context_length(prompt, data)
            return EventSourceResponse(
                stream_generate_completion(prompt, data, request, model_path, disconnect_handler),
                ping=get_sse_ping_interval(),
            )
        else:
            response = await generate_completion(
                prompt, data, request, model_path, disconnect_handler
            )
            return response

    except (CancelledError, InvalidStateError) as ex:
        raise HTTPException(422, "/v1/completions request cancelled by user.") from ex


# Chat completions endpoint
@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(check_api_key), Depends(log_chat_completion_request)],
)
async def chat_completion_request(
    request: Request, data: ChatCompletionRequest
) -> ChatCompletionResponse:
    """
    Generates a chat completion from a prompt.

    If stream = true, this returns an SSE stream.
    """

    # p25: hint the engine a request is arriving so in-flight prefill rounds shrink briefly
    # and this handler can reach enqueue between chunks instead of after the whole prefill.
    note = getattr(model.container, "note_incoming_request", None)
    if note is not None:
        note()

    raw_json = await request.json()
    xlogger.debug("[ENDPOINT] /v1/chat/completions", {"raw": raw_json})

    async with load_lock:
        if data.model:
            await load_inline_model(data.model, request)
        await check_model_container()
        model_path = model.container.model_dir

    # Prepare raw prompt
    if model.container.prompt_template is None:
        error_message = handle_request_error(
            "Chat completions are disabled because a prompt template is not set.",
            exc_info=False,
        ).error.message
        raise HTTPException(422, error_message)
    prompt, mm_embeddings = await apply_chat_template(data)
    await write_chat_completion_prompt_log(request, prompt)

    # Any JSON object, no particular schema. "json_object" is OpenAI's name for it
    if data.response_format.type in ("json_object", "json"):
        data.json_schema = {"type": "object"}

    # Also accept specific schema from response_format
    if data.response_format.type == "json_schema":
        data.json_schema = data.response_format.json_schema

    try:
        disconnect_handler = DisconnectHandler(request, f"{request_tag(request)} chat/completions")
        await disconnect_handler.poll()

        if data.stream and not config.developer.disable_request_streaming:
            model.check_context_length(prompt, data, mm_embeddings)
            return EventSourceResponse(
                stream_generate_chat_completion(
                    prompt, mm_embeddings, data, request, model_path, disconnect_handler
                ),
                ping=get_sse_ping_interval(),
            )
        else:
            response = await generate_chat_completion(
                prompt, mm_embeddings, data, request, model_path, disconnect_handler
            )
            return response

    except (CancelledError, InvalidStateError) as ex:
        raise HTTPException(422, "/v1/chat/completions request cancelled by user.") from ex


# Tokenize / detokenize endpoints (llama.cpp-compatible shapes, also under /v1).
# The jev toolchain calls /tokenize at the ROOT and reads tokens + count;
# /detokenize callers read prompt. Measured 2026-10-04: absent (404) on the
# running build, which fails the jev-medium repoint gate (BACKLOG item 29).

class TokenizeRequest(BaseModel):
    prompt: str
    add_special: bool = False

class TokenizeResponse(BaseModel):
    tokens: list[int]
    count: int

class DetokenizeRequest(BaseModel):
    tokens: list[int]

class DetokenizeResponse(BaseModel):
    prompt: str

@router.post("/tokenize", dependencies=[Depends(check_api_key), Depends(check_model_container)])
@router.post("/v1/tokenize", dependencies=[Depends(check_api_key), Depends(check_model_container)])
async def tokenize_endpoint(data: TokenizeRequest) -> TokenizeResponse:
    tokens = model.container.encode_tokens(
        data.prompt, add_bos_token=data.add_special
    ) or []
    return TokenizeResponse(tokens=list(tokens), count=len(tokens))

@router.post("/detokenize", dependencies=[Depends(check_api_key), Depends(check_model_container)])
@router.post("/v1/detokenize", dependencies=[Depends(check_api_key), Depends(check_model_container)])
async def detokenize_endpoint(data: DetokenizeRequest) -> DetokenizeResponse:
    text = model.container.decode_tokens(list(data.tokens)) or ""
    return DetokenizeResponse(prompt=text)


# Apply template endpoint (llama-server compatible)
@router.post("/apply-template", dependencies=[Depends(check_api_key)])
@router.post("/v1/apply-template", dependencies=[Depends(check_api_key)])
async def apply_template_request(data: ChatCompletionRequest) -> dict:
    """
    Renders the chat template for the given messages without generating and
    returns the resulting prompt. Clients use this to probe the template, e.g.
    whether it reacts to a thinking toggle.
    """

    await check_model_container()

    if model.container.prompt_template is None:
        error_message = handle_request_error(
            "Cannot apply a template because a prompt template is not set.",
            exc_info=False,
        ).error.message
        raise HTTPException(422, error_message)

    prompt, _ = await apply_chat_template(data)
    return {"prompt": prompt}


# Embeddings endpoint
@router.post(
    "/v1/embeddings",
    dependencies=[Depends(check_api_key), Depends(check_embeddings_container)],
)
async def embeddings(request: Request, data: EmbeddingsRequest) -> EmbeddingsResponse:
    embeddings_task = asyncio.create_task(get_embeddings(data, request))
    response = await run_with_request_disconnect(
        request,
        embeddings_task,
        f"{request_tag(request)} embeddings cancelled by client",
    )

    return response

# p47: Anthropic Messages + OpenAI Responses dialects (text-only v1)
from endpoints.OAI import compat  # noqa: E402
router.include_router(compat.router)
