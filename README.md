# Tiled attention (fp32) vs PyTorch SDPA math (fp32) vs PyTorch SDPA flash (fp16)

## Quick summary:
It is a short project that I did so I could understand what the buzz is about (I heard too many times about "avoiding `T x T`" and wanted to actually write it).  
Attention is `softmax(QK^T / sqrt(H))V`. The `QK^T` for a context window of size `T` is a `T x T` matrix. At `T = 4096` that is like 16 million elements that you then have to softmax and multiply by `V`.

### Naive (PyTorch SDPA, math backend):
PyTorch here writes the full `T x T` to VRAM, runs softmax, and then does the second matmul with `V`. This is basic attention, not FlashAttention.

### Tiled (this repo)
A CUDA forward kernel in FlashAttention style. It splits `Q`, `K`, `V` into tiles that fit in shared memory. Each block keeps a running `row_max` and `row_sum` to do online softmax (calculating the right values without ever seeing the full row at once). Once every K/V tile is done (the `(TILE_SIZE_M, head_dim)` matrix is calculated) the block writes that output tile to VRAM. It works in `fp32` with one sequence and one head. Forward pass only.

### Flash (PyTorch SDPA, flash backend):
This is the real thing that gapped my tiled version by a factor of over `20x`. It uses the same tiled idea + `fp16` (tensor cores) + tons and tons of optimizations.

### Measurements
I checked correctness against math `fp32` and timed all three. Both flash and math PyTorch cooked me.

## Benchmark!!!
**Hardware:** RTX 3070 Ti  
**Tensor Dimensions:** `batch = 1`, `num_heads = 1`, `T = 4096`, `H = 64`, 
I used CUDA Events, 100-step warmup / 1000-step test.  


| Kernel | Time (ms) | vs tiled | Max diff vs PyTorch FP32 |
| :--- | :--- | :--- | :--- |
| Tiled (this, FP32) | 4.282 | 1.00× | 0.0 |
| PyTorch SDPA math (FP32) | 1.122 | 3.82× | - |
| PyTorch SDPA Flash (FP16) | 0.181 | 23.7× | - |

## Takeaway
Tiling works and the output matches PyTorch math on `fp32` to a max abs diff of `0.000000`.   
Math SDPA is close to 4x faster and Flash is **destroyed me** by almost 24x (damn). 

## Run
```powershell
python bench.py
```

It needs CUDA + PyTorch to compile `attn.cu` through `torch.utils.cpp_extension`. On Windows it does so via MSVC.
