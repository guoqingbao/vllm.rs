import argparse
import asyncio
import json
import time
import sys
# pip install fastapi uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from vllm_rs import Engine, Message, EngineConfig, SamplingParams, GenerationOutput, GenerationConfig
import uvicorn
import warnings

def current_millis():
    return int(time.time() * 1000)

def performance_metric(outputs: GenerationOutput, total_tokens: int, cached_tokens: int, stream: bool):
    decode_time_taken = 0.0
    prompt_time_taken = 0.0
    total_decoded_tokens = 0
    total_prompt_tokens = 0

    for output in outputs:
        total_prompt_tokens += output.prompt_length
        total_decoded_tokens += output.decoded_length

        prompt_latency = (output.decode_start_time - output.prompt_start_time) / 1000.0
        prompt_time_taken = max(prompt_time_taken, prompt_latency)

        decode_latency = (current_millis() - output.decode_start_time) / 1000.0
        decode_time_taken = max(decode_time_taken, decode_latency)

    if stream:
        print(f"\n--- Performance Metrics [seq_id {outputs[0].seq_id}]---")
    else:
        print(f"\n--- Performance Metrics [{len(outputs)} reqeusts]---")

    if cached_tokens > 0:
        print(f"\n--- Context Cache Usage [{cached_tokens}/{total_tokens} tokens cached]---")

    print(
        f"⏱️ Prompt tokens: {total_prompt_tokens} in {prompt_time_taken:.2f}s "
        f"({total_prompt_tokens / max(prompt_time_taken, 0.001):.2f} tokens/s)"
    )
    print(
        f"⏱️ Decoded tokens: {total_decoded_tokens} in {decode_time_taken:.2f}s "
        f"({total_decoded_tokens / max(decode_time_taken, 0.001):.2f} tokens/s)"
    )


def create_app(cfg, dtype):
    engine = Engine(cfg, dtype)
    app = FastAPI()

    # chat completion for single and batch requests
    def chat_complete(params, messages):
        prompts = [engine.apply_chat_template(params, [Message("user", m["content"])], True) for m in messages]
        outputs = engine.generate_sync(params, prompts)
        performance_metric(outputs, False)
        return outputs

    # chat stream: stream response to single request
    async def chat_stream(params, messages):
        all_messages = [Message(m["role"], m["content"]) for m in messages]
        prompt = engine.apply_chat_template(params, all_messages, False)
        return prompt, engine

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        params = SamplingParams(body.get("temperature", 1.0),
                                body.get("max_tokens", 4096),
                                body.get("ignore_eos", False),
                                body.get("top_k", None),
                                body.get("top_p", None),
                                body.get("session_id", None))
        stream = body.get("stream", False)
        if stream:
            start_time = current_millis()
            prompt, engine = await chat_stream(params, body["messages"])
            print("session_id: ", params.session_id)
            async def streamer():
                try:
                    (seq_id, prompt_length, stream) = engine.generate_stream(params, prompt)
                    decode_start_time = 0
                    decoded_length = 0
                    output_text = ""
                    for token in stream:
                        if await request.is_disconnected():
                            print(
                                f"⛔️ Client has disconnected, stop streaming [seq_id {seq_id}].")
                            stream.cancel()
                            return

                        if not decode_start_time:
                            decode_start_time = current_millis()
                        decoded_length += 1
                        output_text += token
                        if token == "[DONE]":
                            break
                        try:
                            yield "data: " + json.dumps({
                                "choices": [{
                                    "delta": {
                                        "content": token
                                    },
                                    "finish_reason": None
                                }]
                            }) + "\n\n"
                        except Exception as send_err:
                            print(
                                f"⛔️ Sending token to client failed: {send_err}")
                            stream.cancel()
                            return  # Stop streaming
                    yield "data: [DONE]\n\n"
                    decode_finish_time = current_millis()
                    output = type("GenerationOutput", (), {
                        "seq_id": seq_id,
                        "decode_output": output_text,
                        "prompt_length": prompt_length,
                        "prompt_start_time":start_time,
                        "decode_start_time": decode_start_time,
                        "decode_finish_time": decode_finish_time,
                        "decoded_length": decoded_length,
                    })()
                    performance_metric([output], cfg.max_num_seqs * cfg.max_model_len, engine.get_num_cached_tokens(), True)
                except asyncio.CancelledError:
                    print("⛔️ Client disconnected. Cancelling stream.")
                    stream.cancel()
                    raise
                except Exception as e:
                    print(f"⛔️ Stream error: {e}")
                    stream.cancel()
                    raise

            return StreamingResponse(streamer(), media_type="text/event-stream")
        else:
            outputs = chat_complete(params, body["messages"])
            choices = [
                {"message": {"role": "assistant", "content": output.decode_output}}
                for output in outputs
            ]
            return JSONResponse({"choices": choices})

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Run Chat Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--m", help="huggingface model id", type=str, default=None)
    parser.add_argument("--w", help="safetensor weight path", type=str, default=None)
    parser.add_argument("--f", help="gguf file path or gguf file name when model_id is given", type=str, default=None)
    parser.add_argument("--dtype", choices=["f16", "bf16", "f32"], default="bf16")
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--d", type=str, default="0")
    parser.add_argument("--isq", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--penalty", type=float, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # limit default max_num_seqs to 1 on MacOs (due to limited gpu memory)
    max_num_seqs = 1 if sys.platform == "darwin" else args.max_num_seqs
    if args.max_model_len is None:
        if max_num_seqs > 0:
            max_model_len = 65536 // max_num_seqs
        else:
            max_model_len = 65536
        warnings.warn(f"max_model_len is not given, default to {max_model_len}.")
    else:
        max_model_len = args.max_model_len

    generation_cfg = None
    if (args.temperature != None and (args.top_p != None or args.top_k != None)) or args.penalty != None:
         generation_cfg = GenerationConfig(args.temperature, args.top_p, args.top_k, args.penalty)

    assert args.m or args.w or args.f, "Must provide model_id or weight_path or weight_file!"
    cfg = EngineConfig(
        model_id=args.m,
        weight_path=args.w,
        weight_file=args.f,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        isq=args.isq,
        device_ids=[int(d) for d in args.d.split(",")],
        generation_cfg=generation_cfg,
    )

    app = create_app(cfg, args.dtype)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
