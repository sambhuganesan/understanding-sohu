import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "traces"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def plot_systolic():
    rows = read_jsonl(TRACES / "systolic_N8_K16.jsonl")
    plt.figure()
    plt.plot([r["cycle"] for r in rows], [r["util"] for r in rows], marker="o")
    plt.xlabel("cycle")
    plt.ylabel("utilization")
    plt.title("Systolic fill, steady state, drain")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(TRACES / "systolic_util.png", dpi=160)


def plot_batch():
    rows = read_jsonl(TRACES / "batch_sweep.jsonl")
    plt.figure()
    plt.plot([r["batch"] for r in rows], [r["util"] for r in rows], marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("batch")
    plt.ylabel("utilization")
    plt.title("Batch amortizes systolic fill/drain overhead")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(TRACES / "batch_sweep.png", dpi=160)


def plot_contrast():
    rows = read_jsonl(TRACES / "contrast_decode.jsonl")
    plt.figure()
    plt.plot([r["token"] for r in rows], [r["matmul_cycles"] for r in rows], label="matmul")
    plt.plot([r["token"] for r in rows], [r["attention_cycles"] for r in rows], label="attention")
    plt.xlabel("decode token")
    plt.ylabel("cycles")
    plt.title("Matmul stays flat; attention grows with context")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(TRACES / "contrast_decode.png", dpi=160)


def main():
    plot_systolic()
    plot_batch()
    plot_contrast()
    print(f"wrote charts to {TRACES}")


if __name__ == "__main__":
    main()
