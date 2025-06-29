from dataclasses import dataclass
from typing import Iterator, Literal, Mapping, Optional, Callable
from enum import Enum

@dataclass
class DType(Enum):
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"

@dataclass
class GenerationOutput:
    seq_id: int
    prompt_length: int
    decode_start_time: int
    decoded_length: int
    decode_output: str

@dataclass
class EngineConfig:
    model_path: str
    tokenizer: Optional[str]
    tokenizer_config: Optional[str]
    num_blocks: int
    block_size: int
    max_num_seqs: int
    max_num_batched_tokens: int
    pumax_model_len: int
    quant: Optional[str]
    num_shards: Optional[int]
    kvcache_mem_gpu: Optional[int]
    device_id: Optional[int]

@dataclass
class SamplingParams:
    temperature: float
    max_tokens: int
    ignore_eos: bool
    top_k: Optional[int]
    top_p: Optional[float]

@dataclass
class Message:
    role: str
    content: str

@dataclass
class StepOutput(Enum):
    Token: int
    Tokens: list[int]

class Engine:
    def __init__(econfig: EngineConfig, dtype: DType) -> Engine:
        """
        Create a vllm.rs engine with given engine config and dtype ("f16", "bf16", and "f32")
        """

    def add_request(
        self,
        params: SamplingParams,
        prompt: str,
    ) -> tuple[int, int]:
        """
        Add request to the engine, it will return the sequence id and the length of the prompt (number of tokens)
        """

    def step(self) -> list[tuple[int, StepOutput]]:
        """
        After add requests, a generation step will produce a token (sequece_id, token_id) for a request (or all generated tokens (sequence_id, [token_id]) if finished)
        """

    def apply_chat_template(self, messages: list[Message], log: bool) -> str:
        """
        Apply chat template to given messages
        """

    def generate(self,
        params: SamplingParams,
        prompts: list[str],
    ) -> list[GenerationOutput]:
        """
        Chat completion using given prompts and sampling parameters
        """
