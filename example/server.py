import argparse
import sys
import warnings
from vllm_rs import Engine, EngineConfig, GenerationConfig, PdConfig, PdMethod, PdRole

def parse_args():
    parser = argparse.ArgumentParser(description="Run Chat Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--m", help="huggingface model id", type=str, default=None)
    parser.add_argument("--w", help="safetensor weight path", type=str, default=None)
    parser.add_argument("--f", help="gguf file path or gguf file name when model_id is given", type=str, default=None)
    parser.add_argument("--dtype", choices=["f16", "bf16", "f32"], default="bf16")
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--d", type=str, default="0")
    parser.add_argument("--isq", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--context-cache", action="store_true")
    parser.add_argument("--fp8-kvcache", action="store_true")
    parser.add_argument("--cpu-mem-fold", type=float, default=None)
    parser.add_argument("--pd-server", action="store_true")
    parser.add_argument("--pd-client", action="store_true")
    parser.add_argument("--pd-url", help="Url like `192.168.1.100:8888` \
        used for TCP/IP communication between PD server and client", type=str, default=None)
    parser.add_argument("--ui-server", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()

    # limit default max_num_seqs to 1 on MacOs (due to limited gpu memory)
    max_num_seqs = 1 if sys.platform == "darwin" else args.max_num_seqs
    max_model_len = 32768 if sys.platform == "darwin" else 65536 * 2
    if args.max_model_len is None:
        if max_num_seqs > 0:
            max_model_len =  max_model_len // max_num_seqs
        warnings.warn(f"max_model_len is not given, default to {max_model_len}.")
    else:
        max_model_len = args.max_model_len

    generation_cfg = None
    if (args.temperature != None and (args.top_p != None or args.top_k != None)) or args.frequency_penalty != None or args.presence_penalty != None:
         generation_cfg = GenerationConfig(args.temperature, args.top_p, args.top_k, args.frequency_penalty, args.presence_penalty)

    assert args.m or args.w or args.f, "Must provide model_id or weight_path or weight_file!"
    args.max_tokens = max_model_len if args.max_tokens > max_model_len else args.max_tokens

    pd_config = None
    if args.pd_server or args.pd_client:
        pd_role = PdRole.Server if args.pd_server else PdRole.Client
        pd_method = PdMethod.RemoteTcp if args.pd_url != None else PdMethod.LocalIpc
        pd_config = PdConfig(role=pd_role, method=pd_method, url=args.pd_url)

    cfg = EngineConfig(
        model_id=args.m,
        weight_path=args.w,
        weight_file=args.f,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_tokens=args.max_tokens,
        isq=args.isq,
        device_ids=[int(d) for d in args.d.split(",")],
        generation_cfg=generation_cfg,
        flash_context=args.context_cache,
        fp8_kvcache=args.fp8_kvcache,
        server_mode=True,
        cpu_mem_fold=args.cpu_mem_fold,
        pd_config=pd_config,
    )

    engine = Engine(cfg, args.dtype)
    engine.start_server(args.port, args.ui_server) # this will block


if __name__ == "__main__":
    main()
