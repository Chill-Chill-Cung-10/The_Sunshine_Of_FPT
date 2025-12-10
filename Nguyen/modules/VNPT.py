import json
import typing
from typing import Any, ClassVar, Dict, List, Optional
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from langchain_core.runnables import Runnable, RunnableConfig
import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.language_models.base import (
    BaseLanguageModel,
    LangSmithParams,
    LanguageModelInput,
)
from langchain_core.tools import BaseTool
class LangChainVNPT(BaseChatModel):
    """LangChain-compatible chat model wrapper for VNPT endpoints."""

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
        self._tools = None

    @property
    def _llm_type(self) -> str:
        return "vnpt-chat"

    def _prepare_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        payload_msgs: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.type
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            elif role == "tool":
                role = "tool"
            
            msg_dict: Dict[str, Any] = {"role": role}
            
            if role == "tool":
                msg_dict["content"] = msg.content
                msg_dict["tool_call_id"] = msg.tool_call_id

            elif role == "assistant" and hasattr(msg, 'additional_kwargs'):
                tool_calls = msg.additional_kwargs.get('tool_calls')
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                    msg_dict["content"] = msg.content or None
                else:
                    msg_dict["content"] = msg.content
            else:
                msg_dict["content"] = msg.content
            
            payload_msgs.append(msg_dict)
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
        }

        if stop:
            json_data["stop"] = stop

        tools = kwargs.get("tools") or self._tools
        if tools:
            formatted_tools = []
            for tool in tools:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    tool_schema = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                        }
                    }
                    
                    if hasattr(tool, 'args_schema') and tool.args_schema:
                        schema = tool.args_schema.schema()
                        if 'properties' in schema:
                            tool_schema["function"]["parameters"]["properties"] = schema['properties']
                        if 'required' in schema:
                            tool_schema["function"]["parameters"]["required"] = schema['required']
                    
                    formatted_tools.append(tool_schema)
            
            json_data["tools"] = formatted_tools
            
            tool_choice = kwargs.get("tool_choice", self._tool_choice)
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
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls", [])
        
        additional_kwargs = {"raw_response": data}
        if tool_calls:
            additional_kwargs["tool_calls"] = tool_calls
        
        ai_message = AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
        )
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    def bind_tools(self,
                   tools:list
                ):
        self._tools = tools
        return self
