from __future__ import annotations


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


def attention_cycles(context_len: int, d_k: int, heads: int, lanes: int = 1) -> int:
    per_head_macs = 2 * context_len * d_k
    per_head_softmax = 3 * context_len
    return heads * ((per_head_macs + lanes - 1) // lanes + per_head_softmax)


def main() -> None:
    n = 8
    k = 16
    active = simulate_systolic(n, k)
    assert len(active) == 2 * (n - 1) + k
    assert max(active) == n * n

    util_b1 = 1 / (1 + 2 * (n - 1))
    util_b256 = 256 / (256 + 2 * (n - 1))
    assert round(util_b1, 6) == 0.066667
    assert util_b256 > 0.94

    first = attention_cycles(1, 8, 4)
    last = attention_cycles(128, 8, 4)
    assert first == 76
    assert last == 9728
    print("smoke check passed")
    print(f"systolic cycles={len(active)}, peak active={max(active)}/{n*n}")
    print(f"batch util B=1={util_b1:.6f}, B=256={util_b256:.6f}")
    print(f"attention cycles grow {first} -> {last}")


if __name__ == "__main__":
    main()
