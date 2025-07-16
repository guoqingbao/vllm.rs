import argparse
import asyncio
import json
import time
import sys
# pip install fastapi uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from vllm_rs import Engine, Message, EngineConfig, SamplingParams, GenerationOutput
import uvicorn

def current_millis():
    return int(time.time() * 1000)

def performance_metric(seq_id, start_time, outputs: GenerationOutput):
    decode_time_taken = 0.0
    prompt_time_taken = 0.0
    total_decoded_tokens = 0
    total_prompt_tokens = 0

    for output in outputs:
        total_prompt_tokens += output.prompt_length
        total_decoded_tokens += output.decoded_length

        prompt_latency = (output.decode_start_time - start_time) / 1000.0
        prompt_time_taken = max(prompt_time_taken, prompt_latency)

        decode_latency = (current_millis() - output.decode_start_time) / 1000.0
        decode_time_taken = max(decode_time_taken, decode_latency)

    print(f"\n--- Performance Metrics [seq_id {seq_id}]---")
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
        prompts = [engine.apply_chat_template(
            [Message("user", m["content"])], True) for m in messages]
        return engine.generate_sync(params, prompts)

    # chat stream: stream response to single request
    async def chat_stream(params, messages):
        all_messages = [Message(m["role"], m["content"]) for m in messages]
        prompt = engine.apply_chat_template(all_messages, False)
        return prompt, engine

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        params = SamplingParams(body.get("temperature", 1.0),
                                body.get("max_tokens", 4096),
                                body.get("ignore_eos", False),
                                body.get("top_k", None),
                                body.get("top_p", None))
        stream = body.get("stream", False)
        if stream:
            start_time = current_millis()
            prompt, engine = await chat_stream(params, body["messages"])

            async def streamer():
                (seq_id, prompt_length, stream) = engine.generate_stream(
                    params, prompt)
                print(body)
                decode_start_time = 0
                decoded_length = 0
                output_text = ""
                try:
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
                        "decode_start_time": decode_start_time,
                        "decode_finish_time": decode_finish_time,
                        "decoded_length": decoded_length,
                    })()
                    performance_metric(seq_id, start_time, [output])
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
    parser.add_argument("--w", type=str, required=True)  # weight path
    parser.add_argument(
        "--dtype", choices=["f16", "bf16", "f32"], default="bf16")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--d", type=str, default="0")
    return parser.parse_args()


def main():
    args = parse_args()

    # limit default max_num_seqs to 8 on MacOs (due to limited gpu memory)
    max_num_seqs = 8 if sys.platform == "darwin" else { args.max_num_seqs }
    cfg = EngineConfig(
        model_path=args.w,
        max_num_seqs=max_num_seqs,
        max_model_len=args.max_model_len,
        device_ids=[int(d) for d in args.d.split(",")],
    )

    app = create_app(cfg, args.dtype)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
