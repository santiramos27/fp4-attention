import torch
import torch.utils.benchmark as benchmark

# from sageattention.attn_qk_int8_per_block_hd128 import forward
from sageattention.triton.attn_qk_int8_per_block import forward
from sageattention.triton.attn_qk_int8_per_block_causal import forward as forward_causal

import argparse

# from flash_attn.utils.benchmark import benchmark_forward
# remove flash_attn dependency
def benchmark_forward(
    fn, *inputs, repeats=10, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m

parser = argparse.ArgumentParser(description='Benchmark QK Int8 PV FP16 Triton')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
args = parser.parse_args()

batch_size = args.batch_size
num_heads = args.num_heads
head_dim = args.head_dim

print(f"Triton QK Int8 PV FP16 Benchmark")
print(f"batch_size: {batch_size}, num_heads: {num_heads}, head_dim: {head_dim}")

for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
    flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len

    q = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
    k = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

    q_scale = torch.randn(batch_size, num_heads, (seq_len // 128), 1, dtype=torch.float16, device='cuda')
    k_scale = torch.randn(batch_size, num_heads, (seq_len // 64), 1, dtype=torch.float16, device='cuda')

    for i in range(5): forward(q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    _, time = benchmark_forward(forward, q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16, repeats=100, verbose=False, desc='Triton')
    print(f'{seq_len} flops:{flops/time.mean*1e-12}')

for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
    flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len // 2

    q = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
    k = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

    q_scale = torch.randn(batch_size, num_heads, (seq_len // 128), 1, dtype=torch.float16, device='cuda')
    k_scale = torch.randn(batch_size, num_heads, (seq_len // 64), 1, dtype=torch.float16, device='cuda')

    for i in range(5): forward_causal(q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    _, time = benchmark_forward(forward_causal, q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16, repeats=100, verbose=False, desc='Triton')
    print(f'{seq_len} flops:{flops/time.mean*1e-12}')
