from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "traces"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def simulate_systolic(n: int, k: int) -> list[int]:
    total = 2 * (n - 1) + k
    active = []
    for t in range(total):
        count = 0
        for i in range(n):
            for j in range(n):
                if i + j <= t < i + j + k:
                    count += 1
        active.append(count)
    return active


def attention_cost(context_len: int, d_k: int, lanes: int = 1) -> dict:
    macs = 2 * context_len * d_k
    softmax_ops = 3 * context_len
    cycles = (macs + lanes - 1) // lanes + softmax_ops
    return {
        "kv_reads": 2 * context_len,
        "macs": macs,
        "softmax_ops": softmax_ops,
        "cycles": cycles,
    }


def toy_matmul_cycles(d_model: int, d_ff: int) -> int:
    return 4 * d_model * d_model + 2 * d_model * d_ff


def main() -> None:
    n = 8
    k = 16
    active = simulate_systolic(n, k)
    total_pes = n * n
    write_jsonl(
        TRACES / "systolic_N8_K16.jsonl",
        [
            {
                "engine": "systolic",
                "cycle": cycle,
                "active_pes": active_pes,
                "total_pes": total_pes,
                "util": active_pes / total_pes,
            }
            for cycle, active_pes in enumerate(active)
        ],
    )

    batches = []
    batch = 1
    while batch <= 256:
        cycles = 2 * (n - 1) + batch
        batches.append({"demo": "batch_sweep", "batch": batch, "cycles": cycles, "util": batch / cycles})
        batch *= 2
    write_jsonl(TRACES / "batch_sweep.jsonl", batches)

    d_model = 32
    d_k = 8
    heads = 4
    d_ff = 128
    mm_cycles = toy_matmul_cycles(d_model, d_ff)
    contrast = []
    for token in range(1, 129):
        attn = attention_cost(token, d_k)["cycles"] * heads
        contrast.append(
            {
                "demo": "contrast_decode",
                "token": token,
                "context_len": token,
                "matmul_cycles": mm_cycles,
                "attention_cycles": attn,
            }
        )
    write_jsonl(TRACES / "contrast_decode.jsonl", contrast)

    serial = []
    cursor = 0
    for point in contrast:
        start = cursor
        cursor += point["matmul_cycles"]
        serial.append(
            {
                "sched": "serial",
                "token": point["token"],
                "engine": "matmul",
                "op": "qkv_ffn_proj",
                "start": start,
                "end": cursor,
            }
        )
        start = cursor
        cursor += point["attention_cycles"]
        serial.append(
            {
                "sched": "serial",
                "token": point["token"],
                "engine": "attention",
                "op": "kv_stream_softmax",
                "start": start,
                "end": cursor,
            }
        )
    write_jsonl(TRACES / "schedule_serial.jsonl", serial)

    overlap = []
    mm_cursor = 0
    attn_cursor = 0
    for point in contrast:
        mm_start = mm_cursor
        mm_end = mm_start + point["matmul_cycles"]
        mm_cursor = mm_end
        overlap.append(
            {
                "sched": "overlap",
                "token": point["token"],
                "engine": "matmul",
                "op": "qkv_ffn_proj",
                "start": mm_start,
                "end": mm_end,
            }
        )
        attn_start = max(attn_cursor, mm_end)
        attn_end = attn_start + point["attention_cycles"]
        attn_cursor = attn_end
        overlap.append(
            {
                "sched": "overlap",
                "token": point["token"],
                "engine": "attention",
                "op": "kv_stream_softmax",
                "start": attn_start,
                "end": attn_end,
            }
        )
    write_jsonl(TRACES / "schedule_overlap.jsonl", overlap)

    print(f"wrote traces to {TRACES}")


if __name__ == "__main__":
    main()
