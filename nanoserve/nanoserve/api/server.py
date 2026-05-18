"""FastAPI server exposing OpenAI-compatible endpoints over the engine."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import EngineConfig, SamplingParams
from ..engine import InferenceEngine, StreamingOutput
from .protocol import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChoiceDelta,
    ChoiceMessage,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    Usage,
)


log = logging.getLogger("nanoserve.api")


def create_app(engine: InferenceEngine, config: EngineConfig) -> FastAPI:
    app = FastAPI(title="NanoServe", version="0.1.0")
    model_id = config.model_name

    # ------------------------------------------------------------- helpers
    def _format_prompt(req: ChatCompletionRequest) -> list[int]:
        """Apply the model's chat template, then tokenize."""
        messages = [m.model_dump() for m in req.messages]
        try:
            text = engine.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            raise HTTPException(400, f"Chat template error: {e}")
        token_ids = engine.tokenizer.encode(text, add_special_tokens=False)
        return token_ids

    def _build_sampling_params(req: ChatCompletionRequest) -> SamplingParams:
        stop_list: list[str] = []
        if isinstance(req.stop, str):
            stop_list = [req.stop]
        elif isinstance(req.stop, list):
            stop_list = list(req.stop)
        return SamplingParams(
            max_tokens=req.max_tokens or 128,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            stop=stop_list,
            stop_token_ids=[],
            ignore_eos=req.ignore_eos,
            seed=req.seed,
        )

    # ----------------------------------------------------------- endpoints
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/v1/models", response_model=ModelsResponse)
    async def list_models() -> ModelsResponse:
        return ModelsResponse(data=[ModelInfo(id=model_id)])

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, raw: Request):
        prompt_token_ids = _format_prompt(req)
        sampling = _build_sampling_params(req)

        if len(prompt_token_ids) >= config.scheduler.max_model_len:
            raise HTTPException(
                400,
                f"prompt has {len(prompt_token_ids)} tokens, exceeds "
                f"max_model_len={config.scheduler.max_model_len}",
            )

        loop = asyncio.get_running_loop()
        request_id, out_q = engine.add_request(prompt_token_ids, sampling, loop)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if req.stream:
            async def event_stream() -> AsyncIterator[bytes]:
                # First chunk announces the role
                first = ChatCompletionStreamResponse(
                    id=completion_id,
                    model=model_id,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0, delta=ChoiceDelta(role="assistant")
                        )
                    ],
                )
                yield f"data: {first.model_dump_json()}\n\n".encode()

                try:
                    while True:
                        if await raw.is_disconnected():
                            engine.abort_request(request_id)
                            break
                        item: Optional[StreamingOutput] = await out_q.get()
                        if item is None:
                            break
                        finish = item.finish_reason if item.is_finished else None
                        chunk = ChatCompletionStreamResponse(
                            id=completion_id,
                            model=model_id,
                            choices=[
                                ChatCompletionStreamChoice(
                                    index=0,
                                    delta=ChoiceDelta(content=item.delta_text),
                                    finish_reason=finish,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n".encode()
                        if item.is_finished:
                            break
                finally:
                    yield b"data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # Non-streaming: drain the queue, then return one response
        full_text = ""
        finish_reason = "stop"
        prompt_tokens = len(prompt_token_ids)
        completion_tokens = 0
        try:
            while True:
                if await raw.is_disconnected():
                    engine.abort_request(request_id)
                    break
                item: Optional[StreamingOutput] = await out_q.get()
                if item is None:
                    break
                full_text += item.delta_text
                completion_tokens = item.num_generated_tokens
                if item.is_finished:
                    finish_reason = item.finish_reason or "stop"
                    break
        except Exception as e:
            engine.abort_request(request_id)
            raise

        return ChatCompletionResponse(
            id=completion_id,
            model=model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=full_text),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    return app
