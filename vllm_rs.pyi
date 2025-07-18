from dataclasses import dataclass
from typing import Iterator, Tuple, Union, List, Literal, Mapping, Optional, Callable
from enum import Enum
from collections.abc import AsyncGenerator
from typing import Any

@dataclass
class DType(Enum):
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"

@dataclass
class GenerationOutput:
    seq_id: int
    prompt_length: int
    prompt_start_time: int
    decode_start_time: int
    decode_finish_time: int
    decoded_length: int
    decode_output: str

@dataclass
class EngineConfig:
    model_path: str
    tokenizer: Optional[str]
    tokenizer_config: Optional[str]
    num_blocks: Optional[int]
    max_num_seqs: Optional[int]
    max_model_len: Optional[int]
    max_num_batched_tokens: Optional[int]
    quant: Optional[str]
    num_shards: Optional[int]
    device_id: Optional[int]

@dataclass
class SamplingParams:
    temperature: Optional[float]
    max_tokens: Optional[int]
    ignore_eos: Optional[bool]
    top_k: Optional[int]
    top_p: Optional[float]

@dataclass
class Message:
    role: str
    content: str

@dataclass
class StepOutput(Enum):
    Token: int
    Tokens: List[int]

class EngineStream:
    finished: bool
    seq_id: int
    prompt_length: int
    cancelled: bool
    def cancel(self): ...
    def __iter__(self) -> Iterator[str]: ...
    def __next__(self) -> str: ...

class Engine:
    def __init__(econfig: EngineConfig, dtype: DType) -> Engine:
        """
        Create a vllm.rs engine with given engine config and dtype ("f16", "bf16", and "f32")
        """

    def apply_chat_template(self, messages: List[Message], log: bool) -> str:
        """
        Apply chat template to given messages
        """

    def generate_sync(self,
        params: SamplingParams,
        prompts: List[str],
    ) -> List[GenerationOutput]:
        """
        Chat completion using given prompts and sampling parameters
        """
    def generate_stream(
        self,
        params: SamplingParams,
        prompt: str,
    ) -> Tuple[int, int, EngineStream]:
        """
        Chat streaming using given prompts and sampling parameters.

        Return: (seq_id, prompt_length, stream) tuples
        """