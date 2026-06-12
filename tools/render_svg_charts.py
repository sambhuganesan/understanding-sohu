from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "traces"
WIDTH = 760
HEIGHT = 420
LEFT = 70
RIGHT = 24
TOP = 36
BOTTOM = 56


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi == lo:
        return (out_lo + out_hi) / 2
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def polyline(points: list[tuple[float, float]], color: str) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>'


def svg_chart(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float]], str]],
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    all_x = [x for _, points, _ in series for x, _ in points]
    all_y = [y for _, points, _ in series for _, y in points]
    x_min, x_max = min(all_x), max(all_x)
    y_min = min(all_y) if y_min is None else y_min
    y_max = max(all_y) if y_max is None else y_max

    plot_w = WIDTH - LEFT - RIGHT
    plot_h = HEIGHT - TOP - BOTTOM
    x0, y0 = LEFT, HEIGHT - BOTTOM
    x1, y1 = WIDTH - RIGHT, TOP

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{WIDTH / 2}" y="24" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#222"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#222"/>',
    ]

    for i in range(6):
        frac = i / 5
        y = y0 - frac * plot_h
        val = y_min + frac * (y_max - y_min)
        lines.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#ddd"/>')
        lines.append(
            f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{val:.2f}</text>'
        )

    for i in range(5):
        frac = i / 4
        x = x0 + frac * plot_w
        val = x_min + frac * (x_max - x_min)
        lines.append(f'<text x="{x:.1f}" y="{y0 + 22}" text-anchor="middle" font-family="Arial" font-size="12">{val:.0f}</text>')

    for label, points, color in series:
        scaled = [
            (
                scale(x, x_min, x_max, x0, x1),
                scale(y, y_min, y_max, y0, y1),
            )
            for x, y in points
        ]
        lines.append(polyline(scaled, color))
        lx = x1 - 150
        ly = TOP + 22 + 24 * series.index((label, points, color))
        lines.append(f'<line x1="{lx}" y1="{ly - 5}" x2="{lx + 28}" y2="{ly - 5}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{lx + 36}" y="{ly}" font-family="Arial" font-size="13">{label}</text>')

    lines.append(f'<text x="{WIDTH / 2}" y="{HEIGHT - 14}" text-anchor="middle" font-family="Arial" font-size="14">{x_label}</text>')
    lines.append(
        f'<text x="18" y="{HEIGHT / 2}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 18 {HEIGHT / 2})">{y_label}</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    systolic = read_jsonl(TRACES / "systolic_N8_K16.jsonl")
    svg_chart(
        TRACES / "systolic_util.svg",
        "Systolic fill, steady state, drain",
        "cycle",
        "utilization",
        [("systolic", [(r["cycle"], r["util"]) for r in systolic], "#0b6e69")],
        0,
        1,
    )

    batch = read_jsonl(TRACES / "batch_sweep.jsonl")
    svg_chart(
        TRACES / "batch_sweep.svg",
        "Batch amortizes fill/drain overhead",
        "batch",
        "utilization",
        [("batch sweep", [(r["batch"], r["util"]) for r in batch], "#8f3f71")],
        0,
        1,
    )

    contrast = read_jsonl(TRACES / "contrast_decode.jsonl")
    svg_chart(
        TRACES / "contrast_decode.svg",
        "Matmul flat; attention grows with context",
        "decode token",
        "cycles",
        [
            ("matmul", [(r["token"], r["matmul_cycles"]) for r in contrast], "#335c9c"),
            ("attention", [(r["token"], r["attention_cycles"]) for r in contrast], "#c45a2a"),
        ],
    )
    print(f"wrote SVG charts to {TRACES}")


if __name__ == "__main__":
    main()
