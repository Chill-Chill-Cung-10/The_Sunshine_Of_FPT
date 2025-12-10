import json
import typing
from typing import Any, ClassVar, Dict, List, Optional
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from langchain_core.runnables import Runnable, RunnableConfig
import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.language_models.base import (
    BaseLanguageModel,
    LangSmithParams,
    LanguageModelInput,
)
from langchain_core.tools import BaseTool

class VNPT:
    def __init__(self, 
                 model                :str,
                 authorization        :str,
                 tokenKey             :str,
                 tokenId              :str,
                 temperature          :float = 1.0,
                 top_p                :float = 0.9,
                 top_k                :int = 90,
                 n                    :int = 1,
                 max_completion_tokens:int = 50000,
                 tool_choice          :str = "auto"
                ):
        
        model_map = {
            "vnptai-hackathon-small": ("vnptai-hackathon-small", "vnptai_hackathon_small"),
            "vnptai-hackathon-large": ("vnptai-hackathon-large", "vnptai_hackathon_large"),
            "vnptai_hackathon_small": ("vnptai-hackathon-small", "vnptai_hackathon_small"),
            "vnptai_hackathon_large": ("vnptai-hackathon-large", "vnptai_hackathon_large"),
        }

        if model not in model_map:
            raise ValueError(f"Unsupported model {model!r}.  Choose 'vnptai-hackathon-small'/'vnptai_hackathon_small' or 'vnptai-hackathon-large'/'vnptai_hackathon_large'")
        
        endpoint_slug, body_model = model_map[model]
        self._model = body_model
        self._endpoint = f"https://api.idg.vnpt.vn/data-service/v1/chat/completions/{endpoint_slug}"
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._n = n
        self._max_completion_tokens = max_completion_tokens
        self._headers = {
            'Authorization': authorization, 
            'Token-id': tokenId, 
            'Token-key': tokenKey, 
            'Content-Type': 'application/json', 
        }
        self._tool_choice = tool_choice

    def __call__(self,
                 message: str,
                 prompt : str,
                ):

        json_data = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": message},
            ],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "n": self._n,
            "max_completion_tokens": self._max_completion_tokens,
            "response_format": {"type": "json_object"},
            "tool_choice": self._tool_choice
        }

        response = requests.post(self._endpoint, headers=self._headers, json=json_data, timeout=60)
        return response.json()


class LangChainVNPT(BaseChatModel):
    """LangChain-compatible chat model wrapper for VNPT endpoints."""

    # BaseChatModel expects these attributes; keep defaults simple.
    callbacks: Any = None
    callback_manager: Any = None
    verbose: bool = False

    model_map: ClassVar[Dict[str, tuple]] = {
        "vnptai-hackathon-small": ("vnptai-hackathon-small", "vnptai_hackathon_small"),
        "vnptai-hackathon-large": ("vnptai-hackathon-large", "vnptai_hackathon_large"),
        "vnptai_hackathon_small": ("vnptai-hackathon-small", "vnptai_hackathon_small"),
        "vnptai_hackathon_large": ("vnptai-hackathon-large", "vnptai_hackathon_large"),
    }

    def __init__(self,
                 model: str,
                 authorization: str,
                 tokenKey: str,
                 tokenId: str,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 top_k: int = 90,
                 n: int = 1,
                 max_completion_tokens: int = 50000,
                 tool_choice: str = "auto",
                 timeout: int = 60,
                 ) -> None:
        super().__init__()
        if model not in self.model_map:
            raise ValueError("Unsupported model. Use small/large with hyphen or underscore names.")

        endpoint_slug, body_model = self.model_map[model]
        self._endpoint = f"https://api.idg.vnpt.vn/data-service/v1/chat/completions/{endpoint_slug}"
        self._body_model = body_model
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._n = n
        self._max_completion_tokens = max_completion_tokens
        self._timeout = timeout
        self._headers = {
            "Authorization": authorization,
            "Token-id": tokenId,
            "Token-key": tokenKey,
            "Content-Type": "application/json",
        }
        self._tool_choice = tool_choice

    @property
    def _llm_type(self) -> str:
        return "vnpt-chat"

    def _prepare_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        payload_msgs: List[Dict[str, str]] = []
        for msg in messages:
            role = msg.type
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            payload_msgs.append({"role": role, "content": msg.content})
        return payload_msgs

    def _generate(self,
                  messages: List[BaseMessage],
                  stop: Optional[List[str]] = None,
                  **kwargs: Any) -> ChatResult:
        payload_msgs = self._prepare_messages(messages)

        json_data: Dict[str, Any] = {
            "model": self._body_model,
            "messages": payload_msgs,
            "temperature": kwargs.get("temperature", self._temperature),
            "top_p": kwargs.get("top_p", self._top_p),
            "top_k": kwargs.get("top_k", self._top_k),
            "n": kwargs.get("n", self._n),
            "max_completion_tokens": kwargs.get("max_completion_tokens", self._max_completion_tokens),
            "tool_choice": self._tool_choice
        }

        if stop:
            json_data["stop"] = stop

        tools = kwargs.get("tools")
        if tools:
            json_data["tools"] = tools

        tool_choice = kwargs.get("tool_choice")
        if tool_choice:
            json_data["tool_choice"] = tool_choice

        response = requests.post(self._endpoint, headers=self._headers, json=json_data, timeout=self._timeout)
        data = response.json()
        
        if "error" in data and not data.get("choices"):
            raise ValueError(f"VNPT API error: {data}")

        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"VNPT response missing choices: {data}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        
        ai_message = AIMessage(
            content=content,
            additional_kwargs={"raw_response": data},
        )
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    def bind_tools(self,
                   tools:list
                ):
        self._tools = tools
        return self
