"""OpenAI Chat Completions schemas.

Only the subset we actually serve.  Field names and types match the public
OpenAI API so any standard client (the official SDK, LangChain, LiteLLM, the
benchmark/client.py in this repo, etc.) works unchanged.
"""
from __future__ import annotations

import time
import uuid
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ----------------------------------------------------------- request side
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    n: int = 1                         # we ignore n > 1
    seed: Optional[int] = None
    ignore_eos: bool = False
    user: Optional[str] = None


# ----------------------------------------------------------- response side
class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str = ""


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# ----------------------------------------------------------- streaming side
class ChoiceDelta(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: ChoiceDelta
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamChoice]


# ---------------------------------------------------------- /v1/models
class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "nanoserve"


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelInfo]


# ---------------------------------------------------------- /health
class HealthResponse(BaseModel):
    status: Literal["ok", "starting", "error"] = "ok"
