import argparse
import asyncio
import json
# pip install fastapi uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from vllm_rs import Engine, Message, EngineConfig, SamplingParams
import uvicorn

def create_app(cfg, dtype):
    engine = Engine(cfg, dtype)
    app = FastAPI()

    #chat completion for single and batch requests
    def chat_complete(params, messages):
        prompts = [engine.apply_chat_template([Message("user", m["content"])], True) for m in messages]
        return engine.generate_sync(params, prompts)

    #chat stream: stream response to single request
    async def chat_stream(params, messages):
        all_messages = [Message(m["role"], m["content"]) for m in messages]
        prompt = engine.apply_chat_template(all_messages, False)
        return prompt, engine

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        print(body)
        params = SamplingParams(body.get("temperature", 1.0), 
                                body.get("max_tokens", 4096),
                                body.get("ignore_eos", False),
                                body.get("top_k", None),
                                body.get("top_p", None))
        stream = body.get("stream", False)
        if stream:
            prompt, engine = await chat_stream(params, body["messages"])
            async def streamer():
                loop = asyncio.get_event_loop()
                stream = engine.generate_stream(params, prompt)

                try:
                    while True:
                        chunk = await loop.run_in_executor(None, lambda: next(stream, None))
                        if chunk is None:
                            break
                        tokens_str = "".join(str(token) for _, token in chunk)
                        if tokens_str == "[DONE]":
                            break
                        yield "data: " + json.dumps({
                            "choices": [{
                                "delta": {
                                    "content": tokens_str
                                },
                                "finish_reason": None
                            }]
                        }) + "\n\n"
                    yield "data: [DONE]\n\n"

                except asyncio.CancelledError:
                    print("Client disconnected. Cancelling stream.")
                    stream.cancel()
                    raise

                except Exception as e:
                    print(f"Stream error: {e}")
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
    parser.add_argument("--w", type=str, required=True) #weight path
    parser.add_argument("--dtype", choices=["f16", "bf16", "f32"], default="bf16")
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--kvmem", type=int, default=4096)
    parser.add_argument("--device-ids", type=str, default="0")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = EngineConfig(
        model_path=args.w,
        max_num_seqs=args.max_num_seqs,
        kvcache_mem_gpu=args.kvmem,
        device_ids=[int(d) for d in args.device_ids.split(",")],
    )

    app = create_app(cfg, args.dtype)

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
