from vllm_rs import EngineConfig, SamplingParams, Message, GenerationOutput, Engine
import time
# Before running this code, first perform maturin build and then install the package in target/wheels

def current_millis():
    return int(time.time() * 1000)

cfg = EngineConfig(
        model_path = "/Users/bobking/Downloads/Qwen3-8B-Q2_K.gguf",
        block_size = 32,
        max_num_seqs = 64,
        quant = None,
        num_shards = 1,
        kvcache_mem_gpu = 4096,
        device_ids = [0],
    )

engine = Engine(cfg, "bf16")

sampling_params = SamplingParams(
        temperature = 0.6,
        max_tokens = 256,
        ignore_eos = False,
        top_k = None,
        top_p = None,
)
msg1 = Message("user", "How are you?")
msg2 = Message("user", "How to make money?")

prompt1 = engine.apply_chat_template([msg1], True)
prompt2 = engine.apply_chat_template([msg2], True)

prompts = [prompt1, prompt2]

start_time = current_millis()
print("Start inference with", len(prompts), "prompts")
outputs = engine.generate(sampling_params, prompts)

decode_time_taken = 0.0
prompt_time_taken = 0.0
total_decoded_tokens = 0
total_prompt_tokens = 0

for i, output in enumerate(outputs):
    print(f"\n[Prompt {i + 1}]")
    print(f"Prompt: {prompts[i]}")
    print(f"Response: {output.decode_output}")

    total_prompt_tokens += output.prompt_length
    total_decoded_tokens += output.decoded_length

    prompt_latency = (output.decode_start_time - start_time) / 1000.0
    prompt_time_taken = max(prompt_time_taken, prompt_latency)

    decode_latency = (current_millis() - output.decode_start_time) / 1000.0
    decode_time_taken = max(decode_time_taken, decode_latency)

print("\n--- Performance Metrics ---")
print(
    f"⏱️ Prompt tokens: {total_prompt_tokens} in {prompt_time_taken:.2f}s "
    f"({total_prompt_tokens / max(prompt_time_taken, 0.001):.2f} tokens/s)"
)
print(
    f"⏱️ Decoded tokens: {total_decoded_tokens} in {decode_time_taken:.2f}s "
    f"({total_decoded_tokens / max(decode_time_taken, 0.001):.2f} tokens/s)"
)