import time
import os
import sys
import argparse
import warnings
from vllm_rs import EngineConfig, SamplingParams, Message, GenerationOutput, Engine
# Before running this code, first perform maturin build and then install the package in target/wheels


def current_millis():
    return int(time.time() * 1000)


def run(args):
    prompts = args.prompts
    if prompts == None:
        if args.batch > 1:
            prompts = ["Please talk about China in more details."] * args.batch
        else:
            prompts = ["How are you?", "How to make money?"]
            print("⛔️ No prompts found, use default ", prompts)
    else:
        prompts = prompts.split("|")
        if args.batch > 1:
            prompts = prompts[0] * args.batch

    if args.batch > 1:
        max_num_seqs = args.batch
    elif len(prompts) > 0:
        max_num_seqs = len(prompts)
    else:
        # limit default max_num_seqs to 8 on MacOs (due to limited gpu memory)
        max_num_seqs = 8 if sys.platform == "darwin" else args.max_num_seqs

    if args.max_model_len is None:
        max_model_len = 32768 // max_num_seqs
        warnings.warn(f"max_model_len is not given, default to {max_model_len}.")
    else:
        max_model_len = args.max_model_len

    cfg = EngineConfig(
        model_path=args.w,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        device_ids=[int(d) for d in args.d.split(",")],
    )


    engine = Engine(cfg, "bf16")

    sampling_params = []
    
    for i in range(len(prompts)):
        msg = Message("user", prompts[i])
        prompts[i] = engine.apply_chat_template([msg], args.batch == 1)
        sampling_params.append(SamplingParams(max_tokens=args.max_tokens))

    print("Start inference with", len(prompts), "prompts")
    outputs: GenerationOutput = engine.generate_sync(sampling_params, prompts)
    outputs.sort(key=lambda o: o.seq_id)

    decode_time_taken = 0.0
    prompt_time_taken = 0.0
    total_decoded_tokens = 0
    total_prompt_tokens = 0

    for i, output in enumerate(outputs):
        if args.batch == 1:
            print(f"\n[Prompt {i + 1}]")
            print(f"Prompt: {prompts[i]}")
            print(f"Response: {output.decode_output}")

        total_prompt_tokens += output.prompt_length
        total_decoded_tokens += output.decoded_length

        prompt_latency = (output.decode_start_time - output.prompt_start_time) / 1000.0
        prompt_time_taken = max(prompt_time_taken, prompt_latency)

        decode_latency = (output.decode_finish_time - output.decode_start_time) / 1000.0
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
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)

    args = parser.parse_args()
    if not os.path.exists(args.w):
        print("⛔️ Model path is not provided (--w)!")
    else:
        run(args)
