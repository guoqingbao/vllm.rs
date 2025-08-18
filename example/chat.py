import time
import argparse
import warnings
from vllm_rs import Engine, EngineConfig, SamplingParams, Message, GenerationConfig
# Before running this code, first perform maturin build and then install the package in target/wheels


def current_millis():
    return int(time.time() * 1000)


def parse_args():
    parser = argparse.ArgumentParser(description="vllm.rs Python CLI")
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--w", type=str, required=True)
    parser.add_argument("--dtype", type=str,
                        choices=["f16", "bf16", "f32"], default="bf16")
    parser.add_argument("--d", type=str, default="0")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--prompts", type=str,
                        help="Use '|' to separate multiple prompts")
    parser.add_argument("--i", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--isq", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--penalty", type=float, default=None)
    
    return parser.parse_args()


def build_engine_config(args, num_of_prompts):
    if args.max_model_len is None:
        if args.i:
            max_model_len = 32768
        elif num_of_prompts > 0:
            max_model_len = 32768 // num_of_prompts
        else:
            max_model_len = 32768 // args.max_num_seqs
        warnings.warn(f"max_model_len is not given, default to {max_model_len}.")
    else:
        max_model_len = args.max_model_len

    generation_cfg = None
    if (args.temperature != None and (args.top_p != None or args.top_k != None)) or args.penalty != None:
         generation_cfg = GenerationConfig(args.temperature, args.top_p, args.top_k, args.penalty)

    return EngineConfig(
        model_path=args.w,
        max_num_seqs=args.max_num_seqs,
        max_model_len=max_model_len,
        isq=args.isq,
        device_ids=[int(d) for d in args.d.split(",")],
        generation_cfg=generation_cfg,
    )

def show_tokens_left(tokens_left: int, total_tokens: int):
    import shutil
    width = shutil.get_terminal_size(fallback=(80, 20)).columns
    if tokens_left < 0: 
        tokens_left = 0
    token_info = f"Tokens left: {tokens_left}"

    # Choose color based on remaining tokens
    if tokens_left * 1.0 / total_tokens > 0.5:
        color = "\033[32m"  # Green
    elif tokens_left * 1.0 / total_tokens > 0.1:
        color = "\033[33m"  # Yellow
    else:
        color = "\033[31m"  # Red

    reset = "\033[0m"

    # Calculate padding
    space_padding = width - 2 - len(token_info)
    space_padding = max(1, space_padding)  # prevent negative spacing

    # Build the final line
    line = (" " * space_padding) + color + token_info + reset

    print(line)

def main():
    args = parse_args()
    interactive = args.i
    prompts = (
        args.prompts.split("|")
        if args.prompts and not interactive
        else ["How are you today?"]
    )

    econfig = build_engine_config(args, len(prompts))
    engine = Engine(econfig, args.dtype)

    if args.prompts and interactive:
        print("[Warning] Ignoring predefined prompts in interactive mode.")
        prompts = []

    sampling_params = []

    prompt_processed = []

    if not interactive:
        for prompt in prompts:
            msg = Message(role="user", content=prompt)
            processed = engine.apply_chat_template([msg], log=True)
            prompt_processed.append(processed)
            sampling_params.append(SamplingParams(max_tokens=args.max_tokens))
    else:
        sampling_params.append(SamplingParams(max_tokens=args.max_tokens))

    total_available_tokens = econfig.max_num_seqs * econfig.max_model_len
    tokens_left = total_available_tokens
    chat_history = []
    while True:
        if interactive:
            try:
                show_tokens_left(tokens_left, total_available_tokens)
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
                    tokens_left = total_available_tokens
                    continue
                else:
                    print("\n👋 Exiting.")
                    break

            except EOFError:
                print("\n👋 Exiting.")
                break

        start_time = current_millis()
        if interactive:
            seq_id, prompt_length, stream = engine.generate_stream(
                sampling_params[0], prompt_processed[0])
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
            decode_finish_time = current_millis()
            tokens_left = total_available_tokens - prompt_length - decoded_length
            # Construct a GenerationOutput-like object manually
            output = type("GenerationOutput", (), {
                "seq_id": seq_id,
                "decode_output": output_text,
                "prompt_length": prompt_length,
                "prompt_start_time": start_time,
                "decode_start_time": decode_start_time,
                "decode_finish_time": decode_finish_time,
                "decoded_length": decoded_length,
            })()

            outputs = [output]
        else:
            outputs = engine.generate_sync(sampling_params, prompt_processed)

        outputs.sort(key=lambda o: o.seq_id)

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

            prompt_latency = (output.decode_start_time - output.prompt_start_time) / 1000.0
            prompt_time_taken = max(prompt_time_taken, prompt_latency)

            decode_latency = (output.decode_finish_time -
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
