import time
import argparse
from vllm_rs import Engine, EngineConfig, SamplingParams, Message
from concurrent.futures import ThreadPoolExecutor
# Before running this code, first perform maturin build and then install the package in target/wheels

def current_millis():
    return int(time.time() * 1000)

def parse_args():
    parser = argparse.ArgumentParser(description="vllm.rs Python CLI")
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("-w", "--weight-path", type=str, required=True)
    parser.add_argument("--dtype", type=str, choices=["f16", "bf16", "f32"], default="bf16")
    parser.add_argument("--kvmem", type=int, default=4096)
    parser.add_argument("-d", "--device-ids", type=str)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--prompts", type=str, help="Use '|' to separate multiple prompts")
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("--max", dest="max_tokens", type=int, default=4096)

    return parser.parse_args()

def build_engine_config(args):
    return EngineConfig(
        model_path = args.weight_path,
        block_size = 32,
        max_num_seqs = 64,
        quant = None,
        num_shards = 1,
        kvcache_mem_gpu = 4096,
        device_ids = [0],
    )

def main():
    args = parse_args()
    econfig = build_engine_config(args)
    engine = Engine(econfig, args.dtype)

    prompts = (
        args.prompts.split("|")
        if args.prompts and not args.interactive
        else ["How are you today?"]
    )

    if args.prompts and args.interactive:
        print("[Warning] Ignoring predefined prompts in interactive mode.")
        prompts = []

    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=args.max_tokens,
        ignore_eos=False,
        top_k=None,
        top_p=None,
    )

    prompt_processed = []

    if not args.interactive:
        for prompt in prompts:
            msg = Message(role="user", content=prompt)
            processed = engine.apply_chat_template([msg], log=True)
            prompt_processed.append(processed)

    chat_history = []

    future = None
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        while True:
            if args.interactive:
                try:
                    prompt_input = input(
                        "\n🤖✨ Enter your prompt (Ctrl+C to reset chat, Ctrl+D to exit):\n> "
                    ).strip()

                    if not prompt_input:
                        continue

                    msg = Message(role="user", content=prompt_input)
                    chat_history.append(msg)
                    prompt_processed = [engine.apply_chat_template(chat_history, log=False)]

                except KeyboardInterrupt:
                    if chat_history:
                        print("\n🌀 Chat history cleared. Start a new conversation.")
                        chat_history.clear()
                        continue
                    else:
                        print("\n👋 Exiting.")
                        break

                except EOFError:
                    print("\n👋 Exiting.")
                    break

            start_time = current_millis()
            future = executor.submit(engine.generate, sampling_params, prompt_processed)
            outputs = future.result()  # Waits here until complete

            decode_time_taken = 0.0
            prompt_time_taken = 0.0
            total_decoded_tokens = 0
            total_prompt_tokens = 0

            for i, output in enumerate(outputs):
                if not args.interactive and len(prompts) > 1:
                    print(f"\n[Prompt {i + 1}]")
                    print(f"Prompt: {prompts[i]}")
                    print(f"Response: {output.decode_output}")

                total_prompt_tokens += output.prompt_length
                total_decoded_tokens += output.decoded_length

                prompt_latency = (output.decode_start_time - start_time) / 1000.0
                prompt_time_taken = max(prompt_time_taken, prompt_latency)

                decode_latency = (current_millis() - output.decode_start_time) / 1000.0
                decode_time_taken = max(decode_time_taken, decode_latency)

                if args.interactive:
                    assistant_msg = Message(role="assistant", content=output.decode_output)
                    chat_history.append(assistant_msg)

            print("\n--- Performance Metrics ---")
            print(
                f"⏱️ Prompt tokens: {total_prompt_tokens} in {prompt_time_taken:.2f}s "
                f"({total_prompt_tokens / max(prompt_time_taken, 0.001):.2f} tokens/s)"
            )
            print(
                f"⏱️ Decoded tokens: {total_decoded_tokens} in {decode_time_taken:.2f}s "
                f"({total_decoded_tokens / max(decode_time_taken, 0.001):.2f} tokens/s)"
            )

            if not args.interactive:
                break

    except KeyboardInterrupt:
        print("\n⛔️ Interrupted by user. Canceling generation...")
        if future is not None:
            future.cancel()
    finally:
        executor.shutdown(wait=False)
        
if __name__ == "__main__":
    main()
