import time
import argparse
from vllm_rs import Engine, EngineConfig, SamplingParams, Message
# Before running this code, first perform maturin build and then install the package in target/wheels


def current_millis():
    return int(time.time() * 1000)


def parse_args():
    parser = argparse.ArgumentParser(description="vllm.rs Python CLI")
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--w", type=str, required=True)
    parser.add_argument("--dtype", type=str,
                        choices=["f16", "bf16", "f32"], default="bf16")
    parser.add_argument("--kvmem", type=int, default=4096)
    parser.add_argument("--d", type=str, default="0")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--prompts", type=str,
                        help="Use '|' to separate multiple prompts")
    parser.add_argument("--i", action="store_true")
    parser.add_argument("--max", dest="max_tokens", type=int, default=4096)

    return parser.parse_args()


def build_engine_config(args):
    return EngineConfig(
        model_path=args.w,
        kvcache_mem_gpu=args.kvmem,
        device_ids=[int(d) for d in args.d.split(",")],
    )


def main():
    args = parse_args()
    interactive = args.i
    econfig = build_engine_config(args)
    engine = Engine(econfig, args.dtype)

    prompts = (
        args.prompts.split("|")
        if args.prompts and not interactive
        else ["How are you today?"]
    )

    if args.prompts and interactive:
        print("[Warning] Ignoring predefined prompts in interactive mode.")
        prompts = []

    sampling_params = SamplingParams()

    prompt_processed = []

    if not interactive:
        for prompt in prompts:
            msg = Message(role="user", content=prompt)
            processed = engine.apply_chat_template([msg], log=True)
            prompt_processed.append(processed)

    chat_history = []
    while True:
        if interactive:
            try:
                prompt_input = input(
                    "\n🤖✨ Enter your prompt (Ctrl+C to reset chat, Ctrl+D to exit):\n> ")
                if not prompt_input:
                    continue
                msg = Message(role="user", content=prompt_input)
                chat_history.append(msg)
                prompt_processed = [
                    engine.apply_chat_template(chat_history, log=False)]

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

        if interactive:
            start_time = current_millis()
            seq_id, prompt_length, stream = engine.generate_stream(
                sampling_params, prompt_processed[0])
            output_text = ""
            decode_start_time = 0
            decoded_length = 0
            try:
                for token in stream:
                    if not decode_start_time:
                        decode_start_time = current_millis()
                    decoded_length += 1
                    output_text += token
                    print(token, end="", flush=True)
            except KeyboardInterrupt:
                stream.cancel()
                print("\n⛔️ Interrupted by user. Canceling generation...")
            print()  # newline after streaming ends

            # Construct a GenerationOutput-like object manually
            output = type("GenerationOutput", (), {
                "seq_id": seq_id,
                "decode_output": output_text,
                "prompt_length": prompt_length,
                "decode_start_time": decode_start_time,
                "decoded_length": decoded_length,
            })()

            outputs = [output]
        else:
            outputs = engine.generate_sync(sampling_params, prompt_processed)

        decode_time_taken = 0.0
        prompt_time_taken = 0.0
        total_decoded_tokens = 0
        total_prompt_tokens = 0

        for i, output in enumerate(outputs):
            if not interactive and len(prompts) > 1:
                print(f"\n[Prompt {i + 1}]")
                print(f"Prompt: {prompts[i]}")
                print(f"Response: {output.decode_output}")

            total_prompt_tokens += output.prompt_length
            total_decoded_tokens += output.decoded_length

            prompt_latency = (output.decode_start_time - start_time) / 1000.0
            prompt_time_taken = max(prompt_time_taken, prompt_latency)

            decode_latency = (current_millis() -
                              output.decode_start_time) / 1000.0
            decode_time_taken = max(decode_time_taken, decode_latency)

            if interactive:
                assistant_msg = Message(
                    role="assistant", content=output.decode_output)
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

        if not interactive:
            break


if __name__ == "__main__":
    main()
