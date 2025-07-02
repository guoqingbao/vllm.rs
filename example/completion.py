import time
import os
import argparse
from vllm_rs import EngineConfig, SamplingParams, Message, GenerationOutput, Engine
# Before running this code, first perform maturin build and then install the package in target/wheels


def current_millis():
    return int(time.time() * 1000)


def run(args):
    cfg = EngineConfig(
        model_path=args.w,
        kvcache_mem_gpu=4096,  # MB
        device_ids=[int(d) for d in args.d.split(",")],
    )

    prompts = args.prompts

    engine = Engine(cfg, "bf16")

    if prompts == None:
        prompts = ["How are you?", "How to make money?"]
        print("⛔️ No prompts found, use default ", prompts)
    else:
        prompts = prompts.split("|")

    sampling_params = SamplingParams()
    for i in range(len(prompts)):
        msg = Message("user", prompts[i])
        prompts[i] = engine.apply_chat_template([msg], True)

    start_time = current_millis()
    print("Start inference with", len(prompts), "prompts")
    outputs: GenerationOutput = engine.generate_sync(sampling_params, prompts)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vllm.rs Python CLI")
    parser.add_argument("--w", type=str, default="")
    parser.add_argument("--prompts", type=str,
                        help="Use '|' to separate multiple prompts")
    parser.add_argument("--d", type=str, default="0")

    args = parser.parse_args()
    if not os.path.exists(args.w):
        print("⛔️ Model path is not provided (--w)!")
    else:
        run(args)
