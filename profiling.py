import sys
import os
import torch
import torch.distributed as dist
import time
from strassen import run_strassen_fp32_accum, run_matmul_fp32_accum, run_winograd_strassen

class DataDistributed:
    def __call__(self, a: torch.Tensor, b: torch.Tensor, func, three: bool = True) -> torch.Tensor:
        a = a.cuda()
        b = b.cuda()
        
        if three:
            c = torch.empty((*a.shape[:-1], b.shape[-1]), dtype=torch.float16, device='cuda')
            func(a, b, c)
            return c
        else:
            return func(a, b)

def benchmark_matmul(
        M: int,
        N: int,
        K: int,
        num_warmup: int = 10,
        num_runs: int = 100,
        device: str = "cuda"
) -> dict:

    a = torch.randn((M, K), device=device, dtype=torch.float32)
    b = torch.randn((K, N), device=device, dtype=torch.float32)
    c = torch.zeros((M, N), device=device, dtype=torch.float32)

    gt_mm = torch.matmul(a, b)
    dd = DataDistributed()

    for _ in range(num_warmup):
        dd(a, b, run_strassen_fp32_accum, three=True)
        dd(a, b, run_winograd_strassen, three=True)
        dd(a, b, run_matmul_fp32_accum, three=True)
        dd(a, b, torch.matmul, three=False)

    torch.cuda.synchronize()

    # Time Strassen
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    for i in range(num_runs):
        torch.cuda._sleep(1_000_000)
        start_events[i].record()
        dd(a, b, run_strassen_fp32_accum, three=True)
        end_events[i].record()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    triton_strassen_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    # Time Winograd Strassen
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    for i in range(num_runs):
        torch.cuda._sleep(1_000_000)
        start_events[i].record()
        dd(a, b, run_winograd_strassen, three=True)
        end_events[i].record()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    winograd_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    # Time Triton MM
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    for i in range(num_runs):
        torch.cuda._sleep(1_000_000)
        start_events[i].record()
        dd(a, b, run_matmul_fp32_accum, three=True)
        end_events[i].record()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    triton_mm_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    # Time PyTorch
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    for i in range(num_runs):
        torch.cuda._sleep(1_000_000)
        start_events[i].record()
        dd(a, b, torch.matmul, three=False)
        end_events[i].record()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    # Calculate statistics
    triton_strassen_avg = sum(triton_strassen_times) / len(triton_strassen_times)
    winograd_avg = sum(winograd_times) / len(winograd_times)
    triton_mm_avg = sum(triton_mm_times) / len(triton_mm_times)
    torch_avg = sum(torch_times) / len(torch_times)

    triton_strassen_min = min(triton_strassen_times)
    winograd_min = min(winograd_times)
    triton_mm_min = min(triton_mm_times)
    torch_min = min(torch_times)

    results = {
        "triton_strassen_mean_ms": triton_strassen_avg,
        "winograd_mean_ms": winograd_avg,
        "triton_mm_mean_ms": triton_mm_avg,
        "torch_mean_ms": torch_avg,
        "triton_strassen_min_ms": triton_strassen_min,
        "winograd_min_ms": winograd_min,
        "triton_mm_min_ms": triton_mm_min,
        "torch_min_ms": torch_min,
        "triton_strassen_tflops": (2 * M * N * K) / (triton_strassen_min * 1e12 + 1e-6),
        "winograd_tflops": (2 * M * N * K) / (winograd_min * 1e12 + 1e-6),
        "triton_mm_tflops": (2 * M * N * K) / (triton_mm_min * 1e12 + 1e-6),
        "torch_tflops": (2 * M * N * K) / (torch_min * 1e12 + 1e-6),
        "strassen_mean_speedup": torch_avg / triton_strassen_avg,
        "winograd_mean_speedup": torch_avg / winograd_avg,
        "triton_mm_mean_speedup": torch_avg / triton_mm_avg
    }

    return results

def profile_mats():
    sizes = [8192]

    headers = [
        "Size",
        "Strassen (ms)",
        "Winograd (ms)",
        "Triton_MM (ms)",
        "PyTorch (ms)",
        "Strassenx",
        "Winogradx",
        "Tritonx",
        "TF/s"
    ]
    
    fmt = "{:>8} | {:>13.3f} | {:>13.3f} | {:>13.3f} | {:>13.3f} | {:>9.3f} | {:>9.3f} | {:>9.3f} | {:>6.2f}"
    header_fmt = "{:>8} | {:>13} | {:>13} | {:>13} | {:>13} | {:>9} | {:>9} | {:>9} | {:>6}"
    
    print(header_fmt.format(*headers))
    print("-" * 110)

    for size in sizes:
        results = benchmark_matmul(
            M=size, N=size, K=size,
            num_warmup=10,
            num_runs=50
        )
        
        print(fmt.format(
            size,
            results['triton_strassen_mean_ms'],
            results['winograd_mean_ms'],
            results['triton_mm_mean_ms'],
            results['torch_mean_ms'],
            results['strassen_mean_speedup'],
            results['winograd_mean_speedup'],
            results['triton_mm_mean_speedup'],
            results['winograd_tflops']
        ))

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print("\nProfiling different matrix sizes:")
    profile_mats()
