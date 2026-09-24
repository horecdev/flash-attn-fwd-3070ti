import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

flash = load(
    name="flash",
    sources=["attn.cu"],
    extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
    verbose=False
)

SEQ_LEN = 4096
H = 64
ITERS = 1000
WARMUP = 100

# QKV fp32
Q = torch.randn(SEQ_LEN, H, device='cuda', dtype=torch.float32)
K = torch.randn(SEQ_LEN, H, device='cuda', dtype=torch.float32)
V = torch.randn(SEQ_LEN, H, device='cuda', dtype=torch.float32)

# pt FA requires 4D
Q_pt = Q.view(1, 1, SEQ_LEN, H)
K_pt = K.view(1, 1, SEQ_LEN, H)
V_pt = V.view(1, 1, SEQ_LEN, H)

# QKV fp16
Q_pt_fp16 = Q_pt.to(torch.float16)
K_pt_fp16 = K_pt.to(torch.float16)
V_pt_fp16 = V_pt.to(torch.float16)

with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
    torch_out_fp32 = F.scaled_dot_product_attention(Q_pt, K_pt, V_pt).squeeze()

my_out = flash.run(Q, K, V)
max_diff = torch.max(torch.abs(torch_out_fp32 - my_out)).item()

def run_torch_fp16():
    # force flashattn in fp16
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        return F.scaled_dot_product_attention(Q_pt_fp16, K_pt_fp16, V_pt_fp16)

def run_torch_fp32():
    # force standard math fp32 (naive)
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        return F.scaled_dot_product_attention(Q_pt, K_pt, V_pt)

def run_mine():
    return flash.run(Q, K, V)

def benchmark(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / iters

torch_fp16_time = benchmark(run_torch_fp16)
torch_fp32_time = benchmark(run_torch_fp32)
custom_time = benchmark(run_mine)

print(f"Max Diff (Custom vs PyTorch FP32): {max_diff:.6f}")
print(f"PyTorch FP16 (Flash): {torch_fp16_time:.5f} ms")
print(f"PyTorch FP32 (Naive): {torch_fp32_time:.5f} ms")
print(f"My FP32:              {custom_time:.5f} ms")
