# Dual-Engine Simulator Assignment

Use the header files as the contract. Your job is to fill in the `.cpp` files by hand.

## Milestone 1: Matrix

File: `matrix.cpp`

- Store a 2D matrix in one flat `std::vector<float>`.
- Convert `(row, col)` into a flat index with `row * cols + col`.
- Implement `rows()`, `cols()`, element access, and `almost_equal`.

## Milestone 2: Reference Matmul

File: `naive_matmul.cpp`

- Check dimensions before multiplying.
- Use the classic three-loop matrix multiply.
- Count one MAC every time you do `a(r, k) * b(k, c)`.

This is your golden model.

## Milestone 3: Systolic Timing

File: `systolic.cpp`

- For an `N x N` tile and depth `K`, PE `(i, j)` starts at cycle `i + j`.
- It stays active for `K` cycles.
- Total cycles should include fill, steady work, and drain.

Then implement `systolic_tile_matmul` and compare it against `naive_matmul`.

## Milestone 4: Attention Cost

File: `attention.cpp`

- Model attention as streaming over `context_len` tokens.
- Track approximate KV reads, MACs, softmax work, and cycles.
- The important idea: attention cost grows as context length grows.

## Milestone 5: Decode Schedules

File: `schedule.cpp`

- `contrast_decode`: make one row per token showing fixed matmul cost and growing attention cost.
- `serial_schedule`: matmul then attention on one timeline.
- `overlap_schedule`: separate matmul and attention timelines.

## Milestone 6: Trace Output

File: `trace.cpp`

- Create `traces/` if needed.
- Write JSONL files, one object per line.
- Keep the output simple enough that the Python scripts can read it later.

## Commands

Build and run:

```sh
make run
```

Clean the compiled binary:

```sh
make clean
```
