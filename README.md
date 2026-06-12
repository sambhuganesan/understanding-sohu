# Dual-Engine Transformer Simulator

This repo is a small C++ simulator for building intuition about why transformer inference hardware
can benefit from splitting work across two specialized engines:

- a systolic/matmul engine for dense matrix multiplication
- an attention/KV-cache engine for context-dependent streaming work

It is a learning model for me to understand the basics of how Sohu works. The goal was to
make the tradeoff visible: matmul has regular, reusable structure, while decode-time attention grows
with context length and has a different memory/compute shape.

## How To Run

Build and run the C++ simulator:

```sh
make run
```

The program writes JSONL traces into `traces/`.

Generate SVG charts from the traces:

```sh
python3 tools/render_svg_charts.py
```

If you have matplotlib installed, you can also generate PNG charts:

```sh
python3 viz/plot.py
```

Run the lightweight trace check:

```sh
python3 tools/smoke_check.py
```

## What The Demo Produces

### 1. Systolic Utilization

![Systolic utilization](traces/systolic_util.svg)

The systolic trace models a PE grid computing a matrix multiply tile. For a general multiply:

```text
A: rows x k
B: k x cols
C: rows x cols
```

the simulator treats the systolic array as a `rows x cols` grid. PE `(i, j)` owns output element
`C(i, j)`.

The PE starts when the wavefront reaches it:

```text
start_cycle = i + j
```

At cycle `t`, that PE is working on dot-product index:

```text
dot_index = t - (i + j)
```

If `0 <= dot_index < k`, the PE performs:

```text
C(i, j) += A(i, dot_index) * B(dot_index, j)
```

The total cycle count is:

```text
(rows - 1) + (cols - 1) + k
```

The first two terms are the wavefront fill/drain cost. The `k` term is the actual dot-product depth.
For the default demo, `N = 8` and `K = 16`, so the array runs for `30` cycles and reaches `64/64`
active PEs at peak.

What we are seeing: the array starts almost empty, then the wavefront fills it up, then for a short
window every PE is doing useful work, then it drains back down. This is the whole systolic array
intuition in one picture. The math did not disappear, but once the wave is full, a bunch of MACs are
happening at the same time. That's the "oh wait this is why hardware matters" moment.

### 2. Batch Sweep

![Batch sweep](traces/batch_sweep.svg)

This chart is a simplified utilization experiment. It asks: if the systolic array has a fixed
fill/drain overhead, how much does more consecutive work help?

The model uses:

```text
overhead = (rows - 1) + (cols - 1)
cycles = overhead + batch
utilization = batch / cycles
```

Here, `batch` is a toy stand-in for useful steady-state work. In a real transformer matmul, batch is
often the number of activation rows:

```text
X: batch x hidden
W: hidden x output
Y: batch x output
```

Larger batches give the hardware more work to stream through after the array is full. That amortizes
the same fixed fill/drain overhead. In the default `8 x 8` demo, utilization rises from about `6.7%`
at batch `1` to about `94.8%` at batch `256`.

What we are seeing: batch `1` is brutal because you pay the fill/drain cost and barely give the
array anything to chew on. As batch grows, the same setup cost gets spread across way more work. This
is why the high-throughput Sohu story makes sense: if you can keep feeding the machine a big stream
of transformer work, utilization starts climbing fast.

### 3. Matmul vs Attention During Decode

![Decode contrast](traces/contrast_decode.svg)

The decode contrast shows a key transformer inference issue:

- matmul work per generated token is modeled as fixed
- attention work grows as the context length grows

For one attention head at context length `L` and head dimension `D`, the toy attention model uses:

```text
KV reads      = 2L
MACs          = 2LD
softmax ops   = 3L
cycles        = ceil(MACs / lanes) + softmax_ops
```

Why those terms:

- `2L` KV reads: for each context token, read one key vector and one value vector.
- `2LD` MACs: compute `q * K^T` and then `softmax(qK^T) * V`.
- `3L` softmax ops: simple softmax does exponentiate, sum, and divide over `L` scores.
- `lanes`: parallel MAC lanes in the attention engine. More lanes means more MACs per cycle.

The default demo uses `d_k = 8`, `num_heads = 4`, and `128` decode tokens. Attention grows from `76`
cycles to `9728` cycles, while the toy matmul cost stays fixed at `12288` cycles.

What we are seeing: matmul is this flat line because the toy model gives each decode step the same
feed-forward/projection cost. Attention is the line that keeps walking upward because every new token
has more KV cache to look back at. This is the split that makes the dual-engine idea feel natural:
one side is regular dense work, the other side is growing context work.

## Serial vs Overlapped Scheduling

![Schedule comparison](traces/schedule_comparison.svg)

The simulator emits two schedule traces:

- `traces/schedule_serial.jsonl`
- `traces/schedule_overlap.jsonl`

The serial schedule models one timeline:

```text
token 1 matmul -> token 1 attention -> token 2 matmul -> token 2 attention -> ...
```

The overlap schedule models two independent engines:

```text
matmul engine:    token 1 matmul -> token 2 matmul -> token 3 matmul -> ...
attention engine:        token 1 attention -> token 2 attention -> ...
```

This shows the architectural idea in its simplest form: if matmul and attention use different
engines, work that would otherwise occupy one shared path can make progress on two specialized paths.

For the default 128-token decode demo, the total work is the same in both schedules:

```text
matmul work:     1,572,864 cycles
attention work:    627,456 cycles
```

The one-engine schedule runs those pieces serially and finishes in:

```text
2,200,320 cycles
```

The two-engine schedule overlaps attention with following matmul work and finishes in:

```text
1,582,592 cycles
```

That is:

```text
617,728 cycles saved
1.39x throughput improvement
28.1% fewer total cycles
```

What we are seeing: the work did not shrink. The machine just stops waiting around as much. With one
engine, matmul and attention take turns. With two engines, the attention side can work while the
matmul side moves on. That is the whole Sohu-style win in mini form: same transformer, same basic
ops, way better timeline.

## Operation Cost Summary

### Naive Matmul

For:

```text
A: M x K
B: K x N
C: M x N
```

the reference implementation performs:

```text
M * N * K MACs
```

It is the golden model used to check the systolic result.

### Systolic Matmul

The systolic implementation performs the same number of MACs:

```text
M * N * K MACs
```

The win comes from doing many MACs in parallel across many PEs.
With one PE per output element, the wall-clock cycle estimate becomes:

```text
(M - 1) + (N - 1) + K
```

compared with a scalar one-MAC-at-a-time estimate of:

```text
M * N * K
```

Same work, more spatial parallelism.

### Attention

For decode attention, the new token attends over the KV cache:

```text
q: 1 x D
K: L x D
V: L x D
```

The two main compute phases are:

```text
q * K^T                 -> 1 x L scores
softmax(scores) * V     -> 1 x D output
```

That gives the toy cost:

```text
2 * L * D MACs
```

Unlike the fixed matmul cost in the demo, this grows with `L`, the current context length.

## How This Relates To Sohu

This project was inspired by an article analyzing Etched/Sohu from patents, public claims, and first
principles. The exciting idea is simple: transformers are structured enough that a chip can be built
around the operations they actually do all day.

Sohu is Etched's transformer-focused ASIC. This repo is a small educational model of the core
mechanisms that make transformer-only hardware compelling: utilization, batching, attention
specialization, and overlap.

The connection is conceptual:

- Transformer inference has two major personalities: dense matmul/feed-forward work and KV-cache
  attention work.
- Dense feed-forward work likes large, regular systolic-style dataflow and high PE utilization.
- Decode attention is more context-length-dependent, memory/streaming shaped, and includes softmax.
- A dual-engine design gives each workload a compute path shaped for its own behavior.
- Independent memory paths can let weight traffic and KV-cache traffic move at the same time.
- If the engines run independently, feed-forward/matmul work and attention work may overlap.

That is the architecture idea this demo makes visible. It helps explain why a transformer ASIC can
be an excellent fit for high-throughput inference alongside more general-purpose accelerators.

### Why The Upside Can Be Large

The toy computations show three compounding wins that make a Sohu-like architecture exciting.

First, systolic matmul does the same math as naive matmul, but it does many MACs in parallel. For an
`M x K` times `K x N` multiply, the scalar work is:

```text
M * N * K MACs
```

The systolic cycle model is:

```text
(M - 1) + (N - 1) + K
```

That is the core hardware idea: keep a large grid of MAC units busy with a wavefront of useful work.

Second, batching improves utilization. At high batch size, the same weights can be reused across many
rows/tokens/requests:

```text
X: batch x hidden
W: hidden x output
Y: batch x output
```

The systolic engine sees a bigger effective matrix, and fill/drain overhead gets amortized across
more useful work. The demo's batch sweep shows this clearly: utilization rises from about `6.7%` at
batch `1` to about `94.8%` at batch `256`.

Third, attention and feed-forward have different shapes. Feed-forward wants big regular matmuls.
Attention wants KV-cache streaming, score computation, softmax, and value aggregation. A dual-engine
chip can let these two streams make progress together.

Put together, the positive story is:

```text
more specialized compute
+ higher systolic utilization
+ better batch throughput
+ separate attention path
+ less memory-path contention
+ overlap between engines
= a credible path to very high transformer inference throughput
```

### Why Separate Attention?

Attention interleaves three different kinds of work:

```text
q * K^T
softmax
softmax(scores) * V
```

The matmul pieces want MAC throughput. The softmax piece wants vector/nonlinear work. The KV cache
wants streaming reads whose size grows with context length.

This simulator mirrors that point with two deliberately simple models:

- `systolic.cpp` models regular matmul timing and PE utilization.
- `attention.cpp` models attention cost as context-length-dependent KV reads, MACs, and softmax ops.

The overlap schedule then asks: what if these two kinds of work are allowed to make progress on
separate engines?

### The Positive Sohu Mental Model

The demo gives a compact mental model for why Sohu can be successful:

- Transformers spend enormous time in a small set of repeated operations.
- Those operations are regular enough to specialize.
- Systolic arrays can turn matrix multiplication into a high-utilization wavefront.
- High batch sizes let weights be reused across many rows of work.
- Attention has enough unique structure to deserve a dedicated path.
- Running matmul/feed-forward and attention in parallel can improve total token throughput.

That is why this toy model is useful: each chart corresponds to one piece of the Sohu story.

```text
systolic_util.svg     -> keep dense MAC hardware busy
batch_sweep.svg       -> reuse/amortization makes throughput climb
contrast_decode.svg   -> attention is a separate growing workload
schedule_comparison.svg -> two specialized engines shorten the timeline
schedule_*.jsonl      -> two engines can overlap work
```

## Files

- `cpp/matrix.*`: small row-major matrix class.
- `cpp/naive_matmul.*`: reference matmul implementation.
- `cpp/systolic.*`: systolic timing model and systolic-style matmul.
- `cpp/attention.*`: KV-cache attention cost model.
- `cpp/schedule.*`: serial and overlapped decode schedules.
- `cpp/trace.*`: JSONL trace writers.
- `tools/render_svg_charts.py`: no-dependency SVG chart generator.
- `tools/smoke_check.py`: sanity checks for generated traces.
- `viz/plot.py`: optional matplotlib PNG chart generator.

## Model Focus

This model keeps the spotlight on the pieces that make the Sohu story exciting:

- systolic wavefront timing
- PE utilization
- batch amortization
- attention growth with context length
- simple softmax cost
- separate matmul and attention schedules
- overlap between specialized engines

That simplicity is the feature: the simulator is small enough to reason about by hand while still
showing why transformer-focused hardware can be such a strong design direction.
