#!/usr/bin/env python3
import argparse
import json
import time
import sys
import urllib.request
import urllib.error

PROMPT = """You are a senior software engineer. Please provide a comprehensive analysis covering the following topics:

1. Compare and contrast the time complexities of quicksort, mergesort, and heapsort. Under what conditions would you prefer one over the others? Discuss cache locality, stability, and worst-case guarantees. Please include detailed Big-O analysis for best case, average case, and worst case scenarios. Show how quicksort's pivot selection affects performance, discuss the concept of introsort as a hybrid approach, and explain why mergesort is preferred for linked lists while quicksort works better for arrays.

2. Explain the CAP theorem in distributed systems. Give concrete examples of systems that prioritize CP vs AP. How does eventual consistency work in practice? Discuss how systems like Apache Cassandra, MongoDB, and Google Spanner handle the CAP trade-offs differently. Explain the PACELC theorem as an extension of CAP and discuss how network partitions affect real-world distributed databases.

3. Describe how a B+ tree works internally. Why are B+ trees preferred over binary search trees for database indexing? Walk through an insertion that causes a split. Explain the difference between B-trees and B+ trees, discuss the concept of fill factor, and show how range queries benefit from the linked leaf nodes in B+ trees. Include a comparison with hash indexes and LSM trees for different workload patterns.

4. What are the key differences between TCP and UDP? When would you choose UDP over TCP in a real application? Explain how TCP handles congestion control with algorithms like slow start, congestion avoidance, fast retransmit, and fast recovery. Discuss the differences between TCP Reno, TCP Cubic, and BBR congestion control algorithms. Explain how QUIC protocol builds on UDP to provide reliable transport.

5. Explain the concept of memory-mapped I/O. How does the operating system manage virtual memory? Describe page faults and the page replacement algorithms (LRU, Clock, Working Set). Discuss the TLB and its role in address translation. Explain how huge pages improve performance for large datasets. Discuss the differences between demand paging and pre-paging strategies.

6. Discuss the SOLID principles in object-oriented design with practical examples. How do these principles help in building maintainable software? Provide concrete code examples for each principle. Discuss the relationship between SOLID principles and design patterns. Explain how dependency injection frameworks implement the Dependency Inversion Principle.

7. What is the difference between optimistic and pessimistic concurrency control in databases? Explain MVCC and how PostgreSQL implements it. Discuss the concept of snapshot isolation and its relationship to serializability. Explain the write skew anomaly and how it can be prevented. Compare MVCC implementations in PostgreSQL, MySQL InnoDB, and Oracle.

8. Describe the architecture of a modern web browser rendering engine. How does the critical rendering path work from HTML parsing to painting pixels? Explain the role of the compositor thread, how GPU acceleration works for CSS transforms, and how the browser handles reflow vs repaint operations. Discuss the concept of layers and how browsers optimize scrolling performance.

Additionally, please cover these supplementary topics:

9. Explain the internals of a garbage collector. Compare mark-and-sweep, generational, and concurrent garbage collection approaches. Discuss how the Go garbage collector works with its tri-color marking algorithm. Compare with Java's G1 GC and ZGC. Explain the concept of safe points and how they affect application latency.

10. Describe the architecture of a modern CPU pipeline. Explain branch prediction, speculative execution, and out-of-order execution. Discuss how Spectre and Meltdown vulnerabilities exploited these features. Explain the concept of cache coherence protocols (MESI, MOESI) in multi-core processors.

Additional context for comprehensive coverage: The field of computer science encompasses a vast array of topics from theoretical foundations like computability theory, formal languages, and automata theory, to practical applications in software engineering, systems design, and machine learning. Understanding the interplay between algorithmic complexity and system architecture is crucial for building efficient, scalable software systems. Modern computing challenges often require knowledge spanning multiple domains - from low-level hardware optimization and cache-aware algorithms to high-level distributed system design patterns and cloud-native architectures. The evolution from monolithic applications to microservices has introduced new challenges in areas like service mesh networking, container orchestration, observability, and distributed tracing. Similarly, the rise of machine learning has created demand for specialized hardware accelerators, efficient inference engines, and novel data processing pipelines. Database technology has evolved from simple relational models to encompass graph databases, time-series databases, vector databases, and multi-model approaches. Each of these technologies brings its own set of trade-offs in terms of consistency, availability, performance, and operational complexity. The emergence of edge computing and IoT has further complicated the landscape, requiring new approaches to data synchronization, conflict resolution, and offline-first architectures. Quantum computing, while still in its early stages, promises to revolutionize certain classes of problems in cryptography, optimization, and molecular simulation. Understanding these emerging paradigms alongside classical computing fundamentals is essential for any comprehensive technical reference. Furthermore, the growing importance of security in software systems demands knowledge of threat modeling, secure coding practices, encryption algorithms, and authentication protocols. The shift toward zero-trust architectures and the increasing sophistication of cyber attacks make security considerations an integral part of every system design decision.
"""


def wait_for_server(port, timeout):
    url = f"http://localhost:{port}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def check_quality(text):
    issues = []
    if len(text) < 100:
        issues.append("Response too short (< 100 chars)")

    words = text.split()
    if len(words) >= 3:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        if trigrams:
            from collections import Counter
            counts = Counter(trigrams)
            threshold = max(10, int(0.05 * len(trigrams)))
            for gram, cnt in counts.most_common(5):
                if cnt > threshold:
                    issues.append(f"Excessive repetition: '{gram}' x{cnt} (threshold={threshold})")
                    break

    return issues


def send_request(port, thinking=False, max_tokens=2048, temperature=0.7):
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [
            {"role": "user", "content": PROMPT}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "thinking": thinking,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        return {"error": f"HTTP {e.code}: {error_body}", "time": time.time() - start}
    except Exception as e:
        return {"error": str(e), "time": time.time() - start}

    elapsed = time.time() - start
    choice = body.get("choices", [{}])[0]
    message = choice.get("message", {})
    text = message.get("content", "")
    usage = body.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    throughput = completion_tokens / elapsed if elapsed > 0 else 0

    quality_issues = check_quality(text)

    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "time": elapsed,
        "throughput": throughput,
        "quality_issues": quality_issues,
        "thinking": thinking,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--wait", type=int, default=300)
    args = parser.parse_args()

    print(f"Waiting for server on port {args.port}...")
    if not wait_for_server(args.port, args.wait):
        print("ERROR: Server not ready within timeout")
        sys.exit(1)
    print("Server is ready.\n")

    results = []

    for thinking in [False, True]:
        mode = "thinking=true" if thinking else "thinking=false"
        print(f"Testing {mode}...")
        result = send_request(args.port, thinking=thinking)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            results.append({"mode": mode, "error": result["error"], "time": result["time"]})
            continue

        quality = "OK" if not result["quality_issues"] else "; ".join(result["quality_issues"])
        print(f"  Prompt tokens:     {result['prompt_tokens']}")
        print(f"  Completion tokens: {result['completion_tokens']}")
        print(f"  Time:              {result['time']:.1f}s")
        print(f"  Throughput:        {result['throughput']:.1f} tok/s")
        print(f"  Quality:           {quality}")
        print(f"  Response preview:  {result['text'][:300]}...")
        print()

        results.append({
            "mode": mode,
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "time": round(result["time"], 1),
            "throughput": round(result["throughput"], 1),
            "quality": quality,
        })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Mode':<20} {'Prompt':>8} {'Completion':>12} {'Time':>8} {'Tok/s':>8} {'Quality'}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['mode']:<20} {'ERROR':>8} {r['error']}")
        else:
            print(f"{r['mode']:<20} {r['prompt_tokens']:>8} {r['completion_tokens']:>12} {r['time']:>7.1f}s {r['throughput']:>7.1f} {r.get('quality', 'N/A')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
