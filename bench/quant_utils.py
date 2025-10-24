# SPDX-License-Identifier: Apache-2.0
# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2025

from typing import Tuple
import torch
from torch import Tensor
import triton
import triton.language as tl
from triton.language.extra import libdevice
# from .triton_kernels.utils import IS_HIP, get_num_SMs, next_power_of_2
#from .dtypes import *

# SPDX-License-Identifier: Apache-2.0
# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2025

import torch
import triton.language as tl
from enum import Enum

class DType(Enum):
    FP32   = 0
    FP16   = 1
    BF16   = 2
    FP8    = 3
    FP8e4  = 3 #alias for FP8
    INT8   = 4
    UINT8  = 5
    INT32  = 6
    UINT32 = 7
    FP8e5  = 8
    INT16  = 9
    UINT16 = 10
    INT64  = 11
    FP8e4nuz = 12
    FP8e5nuz = 13
    MXFP16 = 14
    MXBF16 = 15
    MXFP8  = 16
    MXFP4  = 17
    NVFP4  = 18
    E8M0   = 19


DTYPE_TO_TORCH = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
    3: torch.float8_e4m3fn,
    4: torch.int8,
    5: torch.uint8,
    6: torch.int32,
    7: torch.uint32,
    8: torch.float8_e5m2,
    9: torch.int16,
    10: torch.uint16,
    11: torch.int64,
    12: torch.float8_e4m3fnuz,
    13: torch.float8_e5m2fnuz,
    14: torch.float16,
    15: torch.bfloat16,
    16: torch.float8_e4m3fn,
    17: torch.uint8,
    18: torch.uint8,
    19: torch.float8_e8m0fnu,
}

TORCH_TO_DTYPE = {
    torch.float32: DType.FP32,
    torch.float16: DType.FP16,
    torch.bfloat16: DType.BF16,
    torch.int8: DType.INT8,
    torch.uint8: DType.UINT8,
    torch.int32: DType.INT32,
    torch.uint32: DType.UINT32,
    torch.int16: DType.INT16,
    torch.uint16: DType.UINT16,
    torch.int64: DType.INT64,
    torch.float8_e4m3fn: DType.FP8,
    torch.float8_e5m2: DType.FP8e5,
    torch.float8_e4m3fnuz: DType.FP8e4nuz,
    torch.float8_e5m2fnuz: DType.FP8e5nuz,
    torch.float8_e8m0fnu: DType.E8M0,
}

TORCH_DTYPE_TO_TRITON = {
    torch.float16:       tl.float16,
    torch.float32:       tl.float32,
    torch.bfloat16:      tl.bfloat16,
    torch.int8:          tl.int8,
    torch.uint8:         tl.uint8,
    torch.int16:         tl.int16,
    torch.uint16:        tl.uint16,
    torch.int32:         tl.int32,
    torch.uint32:        tl.uint32,
    torch.int16:         tl.int16,
    torch.uint16:        tl.uint16,
    torch.int64:         tl.int64,
    torch.float8_e4m3fn: tl.float8e4nv, #NVIDIA
    torch.float8_e5m2: tl.float8e5,#NVIDIA
    torch.float8_e4m3fnuz: tl.float8e4b8, #AMD
    torch.float8_e5m2fnuz: tl.float8e5b16, #AMD
    torch.float8_e8m0fnu: tl.uint8,
}

DTYPE_TO_TRITON = {k:TORCH_DTYPE_TO_TRITON[d] for k,d in DTYPE_TO_TORCH.items()}

PACKING_BITWIDTH_TO_TORCH_DTYPE = {
    8: torch.uint8,
    16: torch.int16,
    32: torch.int32,
    64: torch.int64,
}

FP8_DTYPES = [DType.FP8, DType.FP8e4, DType.FP8e5, DType.FP8e4nuz, DType.FP8e5nuz]
FP8_INT8_DTYPES = [DType.INT8] + FP8_DTYPES
MX_DTYPES = [DType.MXFP16, DType.MXBF16, DType.MXFP8, DType.MXFP4, DType.NVFP4]
MX_DTYPES_val = [dtype.value for dtype in MX_DTYPES]

def is_mx_dtype(input_dtype):
    if(type(input_dtype) == int):
        return input_dtype in MX_DTYPES_val
    elif(type(input_dtype) == DType):
        return input_dtype in MX_DTYPES

# SPDX-License-Identifier: Apache-2.0
# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2025

import torch, triton, math
import triton.language as tl
from triton.runtime import driver
# from ..dtypes import *

@triton.jit
def swizzle_tile_v1(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    grid_m     = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n     = tl.cdiv(N, BLOCK_SIZE_N)
    width      = GROUP_SIZE_M * grid_n
    group_id   = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m      = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n      = (pid % width) // group_size
    return pid_m, pid_n

@triton.jit
def swizzle_tile_v2(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    grid_m     = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n     = tl.cdiv(N, BLOCK_SIZE_N)
    width      = GROUP_SIZE_M * grid_m
    group_id   = pid // width
    group_size = tl.minimum(grid_n - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_n      = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_m      = (pid % width) // group_size
    return pid_m, pid_n

@triton.jit
def swizzle_tile_v3(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid_m  = pid % tl.cdiv(M, BLOCK_SIZE_M)
    pid_n  = pid // tl.cdiv(M, BLOCK_SIZE_M)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    return tl.swizzle2d(pid_m, pid_n, grid_m, grid_n, GROUP_SIZE_M)

@triton.jit
def swizzle_tile_persistent(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

swizzle_tile = swizzle_tile_v1

@triton.jit
def linear_tile(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid_m = pid % tl.cdiv(M, BLOCK_SIZE_M)
    pid_n = pid // tl.cdiv(M, BLOCK_SIZE_M)
    return pid_m, pid_n

#################################################################################################################
@triton.jit
def dequantize(
    b,
    scales,
    zeros,
    q_shift,
    meta_dtype,
    unpack_mask,
    elements_per_sample: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
):
    #Unpack
    if(elements_per_sample > 1):
        b = (b >> q_shift) & unpack_mask # int32 -> int32

    if(W_group_mode == 1): #Shift
        b = b.to(meta_dtype) - zeros

    if(W_group_mode == 2):
        b = b.to(meta_dtype) * scales #Symmetric no shift (Grouped)

    if(W_group_mode == 3): #Asymmetric / Symmetric with shift(Grouped - (b - zeros) * scales)
        #b = (b - zeros) * scales
        if(zero_is_scalar):
            b = (b - zeros).to(meta_dtype) * scales
        else:
            b = (b.to(meta_dtype) - zeros) * scales

    if(W_group_mode == 4):
        b = tl.fma(b.to(meta_dtype), scales, zeros) #Asymmetric (Grouped - b*scales + zeros)

    return b

@triton.jit
def atomic_add_cas(ptr, value, Lock, mask=None, sem: tl.constexpr = "release"):
    while tl.atomic_cas(Lock, 0, 1, sem=sem) == 1:
        pass
    tl.store(ptr, tl.load(ptr, mask=mask) + value, mask=mask)
    tl.debug_barrier()
    tl.atomic_xchg(Lock, 0)

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

def next_power_of_2(v):
    return 2 ** int(math.ceil(math.log2(v)))

def is_divisible(dividend, divisor):
    return dividend % divisor == 0

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def gpu_has_more_shared_memory(ref_gpus = ["a100", "h100", "h200", "h20", "h800", "b100", "b200"]):
    gpu_name = torch.cuda.get_device_properties(0).name.lower()
    return True in [g in gpu_name for g in ref_gpus]

def gpu_supports_float16_acc(
    ref_gpus=["5090", "5080", "5070", "5060",
              "4090", "4080", "4070", "4060",
              "3090", "3080", "3070", "3060",
              "2080", "2070"]
):
    gpu_name = torch.cuda.get_device_properties(0).name.lower()
    return True in [g in gpu_name for g in ref_gpus]


def gpu_supports_bfloat16_atomicadd():
    #Triton tl.atomic_add doens't support bfloat16 even for Hopper and above.
    #return torch.cuda.get_device_capability()[0] >= 9 #Hopper and above
    return False

NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
def get_num_SMs(device):
    #return torch.cuda.get_device_properties(device).multi_processor_count
    return NUM_SMS #cache it to avoid driver ping - should be ok as multi-gpu systems tend to have the same devices

#Only powers of 2
def generate_autotune_lookup_v1(max_m=16384):
    return [min(2 ** int(math.ceil(math.log2(M))), max_m) if (M > 0) else 0 for M in range(max_m + 1)]

#Powers of 2 but also (power of 2 + next power of 2) / divisor,
def generate_autotune_lookup_v2(max_m=16384, min_split=32, divisors=[2, 4], mode='next', include_vllm_config=False):
    lookup = [0] * (max_m + 1)
    autotune_vals = set()

    i = 0
    while (val := 2 ** i) <= max_m:
        autotune_vals.add(val)
        next_val = 2 ** (i + 1)
        if val >= min_split and next_val <= max_m:
            for d in divisors:
                interpolated = (val + next_val) // d
                autotune_vals.add(interpolated)
        i += 1

    if(include_vllm_config):
        autotune_vals.update([1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64,
                             72, 80, 88, 96, 104, 112, 120, 128,
                             136, 144, 152, 160, 168, 176, 184, 192,
                             200, 208, 216, 224, 232, 240, 248, 256, 384, 512])

    sorted_vals = sorted(autotune_vals)

    for m in range(max_m + 1):
        if(mode == 'next'):
            lookup[m] = min((x for x in sorted_vals if x >= m), default=None) #Next-value
        elif(mode == 'closest'):
            lookup[m] = min(sorted_vals, key=lambda x: (abs(x - m), x < m)) #Closest-Value
        else:
            raise Exception('Invalid mode.')
    return lookup

M_MAXVAL  = 4096 #1024, 4096, 16384
M_MAPPING = generate_autotune_lookup_v2(M_MAXVAL, mode='next')
def get_closest_m(M):
    return M_MAPPING[M] if M <= M_MAXVAL else M_MAXVAL

def get_gpu_shared_memory():
    return driver.active.utils.get_device_properties(0).get("max_shared_mem", 0)

###################################################################################
#Cached results to avoid runtime driver calls

IS_HIP = is_hip()
NATIVE_ATOMIC = gpu_supports_bfloat16_atomicadd()

#Get dtype min/max range based on compute dtype
def get_dtype_range(compute_dtype: torch.dtype) -> float:
    if(compute_dtype.is_floating_point):
        dtype_info = torch.finfo(compute_dtype)
    else:
        dtype_info = torch.iinfo(compute_dtype)
    return dtype_info.min, dtype_info.max

NVFP4_META_SCALE = 0.05 #Temporary NVFP logic
####################################################################################################################
#MXFP4 / NVFP4 weight quantizer
####################################################################################################################

#Cache workspace for multiple gpus (less than a KB per GPU)
fp4_values, fp4_p_vals, fp4_thresholds, thr_pos = [], [], [], []
for g_id in range(torch.cuda.device_count()):
    current_device = "cuda:" + str(g_id)

    fp4_values.append(
        torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float32,
            device=current_device,
        )
    )

    fp4_p_vals.append(
        torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6],
            dtype=torch.float32,
            device=current_device,
        )
    )

    fp4_thresholds.append(
        torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
            dtype=torch.float32,
            device=current_device,
        )
    )  # (fp4_p_vals[:-1] + fp4_p_vals[1:]) / 2

    fp4_pos = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6],
        dtype=torch.float32,
        device=current_device,
    )

    thr_pos.append(
        #last val is dummy to make len a power of 2
        torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0], 
            dtype=torch.float32,
            device=current_device,
        )
    )  # (fp4_p_vals[:-1] + fp4_p_vals[1:]) / 2

class WeightQuantizerMXFP:
    def __init__(self, compute_dtype=torch.bfloat16, device="cuda:0"):
        self.compute_dtype = compute_dtype
        self.device        = device

    def round_to_closest_fp4(self, tensor):
        device_index = tensor.device.index
        out = fp4_p_vals[device_index][
            torch.searchsorted(
                fp4_thresholds[device_index].to(tensor.dtype), tensor.abs()
            )
        ].to(tensor.dtype)
        out *= tensor.sign()
        return out

    def to_index(self, W_q):
        assert W_q.is_floating_point(), "Input should be floating point fp4 values."
        device_index = W_q.device.index
        return (
            (W_q.view(-1, 1) == fp4_values[device_index].to(W_q.dtype).view(1, -1))
            .to(torch.uint8)
            .argmax(dim=1)
            .to(torch.uint8)
            .view(W_q.shape)
        )

    @torch.compile(fullgraph=True)
    def quantize_mxfp8(
        self,
        W: torch.Tensor,
        index: bool = False,
        mx_fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> (torch.Tensor, torch.Tensor):
        group_size: int = 32
        eps_exp: int = -30
        eps: float = 2 ** eps_exp
        min_val = torch.finfo(mx_fp8_dtype).min
        max_val = torch.finfo(mx_fp8_dtype).max

        W_flat = W.view(-1, group_size).float()
        ideal_scale = W_flat.abs().amax(dim=1, keepdim=True)
        ideal_scale /= max_val

        scales = (2 ** torch.ceil(torch.log2(ideal_scale))).clamp_(min=eps)

        W_q = (W_flat / scales).clamp_(min=min_val, max=max_val)
        scales = scales.to(torch.float8_e8m0fnu)

        if(index):
            W_q = W_q.to(mx_fp8_dtype)
        else:
            W_q = W_q.to(mx_fp8_dtype).to(W_flat.dtype)

        return W_q, scales
    
    @torch.compile(fullgraph=True)
    def quantize_mxfp4(
        self, W: torch.Tensor, window_size: int = 0, index: bool = False
    ) -> (torch.Tensor, torch.Tensor):
        group_size: int = 32
        eps_exp: int = -30
        eps: float = 2 ** eps_exp
        W_nbits = 4
        max_val = 6

        W_flat = W.view(-1, group_size).float()
        ideal_scale = W_flat.abs().amax(dim=1, keepdim=True)
        ideal_scale /= max_val

        if(window_size == 0):
            scales = 2 ** torch.ceil(torch.log2(ideal_scale))
        else:
            initial_log2_scales = torch.ceil(torch.log2(ideal_scale))
            search_offsets = torch.arange(
                -window_size,
                window_size + 1,
                device=W.device,
                dtype=initial_log2_scales.dtype,
            ).view(1, -1)
            candidate_scales = torch.pow(2, initial_log2_scales + search_offsets)
            candidate_scales[candidate_scales < eps] = eps

            W_q_candidates = self.round_to_closest_fp4(W_flat.unsqueeze(1) / candidate_scales.unsqueeze(-1))
            W_r_candidates = W_q_candidates * candidate_scales.unsqueeze(-1)
            errors = (W_flat.unsqueeze(1) - W_r_candidates).abs().mean(dim=-1)
            scales = torch.gather(candidate_scales, 1, torch.argmin(errors, dim=1, keepdim=True))

        scales = scales.clamp_(eps)
        W_q = self.round_to_closest_fp4(W_flat / scales)
        scales = scales.to(torch.float8_e8m0fnu)

        if(index):
            W_q = self.to_index(W_q)
        return W_q, scales
    
    @torch.compile(fullgraph=True)
    def quantize_nvfp4(
        self, W: torch.Tensor, window_size: int = 0, index: bool = False
    ) -> (torch.Tensor, torch.Tensor):

        group_size: int = 16
        eps: float = 1e-6
        W_nbits = 4
        max_val = 6
        fp8_dtype = torch.float8_e4m3fn #This is for Nvidia only.
        max_fp8 = torch.finfo(fp8_dtype).max #448

        W_flat = W.view(-1, group_size).float()
        ideal_scale = W_flat.abs().amax(dim=1, keepdim=True)
        ideal_scale /= max_val
        meta_scales = NVFP4_META_SCALE #ideal_scale.max().clamp_(min=eps) - TODO: use max()
        ideal_scale /= meta_scales
        ideal_scale = ideal_scale.clamp_(max=max_fp8).to(fp8_dtype)

        if(window_size == 0):
            scales = ideal_scale
        else:
            search_offsets = torch.arange(
                -window_size, window_size + 1, device=W.device, dtype=torch.int
            ).view(1, -1)

            candidate_scales = (
                (ideal_scale.view(torch.int8) + search_offsets)
                .clamp_(-128, 127)
                .to(torch.int8)
            )

            #Avoid nan in int8 range (-1, 127 as int8 as e4m3 nans)
            candidate_scales[candidate_scales==-1] = 1
            candidate_scales[candidate_scales==127] = 1
            candidate_scales = candidate_scales.view(fp8_dtype).float()
            candidate_scales[candidate_scales < eps] = eps

            W_q_candidates = self.round_to_closest_fp4(W_flat.unsqueeze(1) / (candidate_scales * meta_scales).unsqueeze(-1))
            W_r_candidates = W_q_candidates * candidate_scales.unsqueeze(-1)
            errors = (W_flat.unsqueeze(1) - W_r_candidates).abs().mean(dim=-1)
            scales = torch.gather(candidate_scales, 1, torch.argmin(errors, dim=1, keepdim=True)).to(fp8_dtype)

        scales_full = (scales.to(W_flat.dtype) * meta_scales).clamp_(min=eps)
        W_q = self.round_to_closest_fp4(W_flat / scales_full)

        if(index):
            W_q = self.to_index(W_q)

        return W_q, scales

    def dequantize(self, W_q, scales, shape = None, dtype = None):
        if(W_q.dtype == torch.uint8): #from indices
            device_index = W_q.device.index
            W_q = fp4_values[device_index][W_q.int()]

        group_size = W_q.numel() // scales.numel()
        out = (W_q.view([-1, group_size]).float() * scales.float())
        if(shape is not None):
            out = out.view(shape)
        return out.to(self.compute_dtype if dtype is None else dtype)

####################################################################################################################
#INT8 / FP8 activations
####################################################################################################################
# Main activation scaling functions
@torch.compile(fullgraph=True)
def scale_activations_per_token_torch(
    tensor: Tensor, w_dtype: torch.dtype, fp32_scale: bool = True
) -> Tuple[Tensor, Tensor]:

    min_val, max_val = get_dtype_range(w_dtype)
    if fp32_scale:
        tensor = tensor.to(torch.float32, copy=False)
    out_shape = tensor.shape
    out = tensor.view(-1, tensor.shape[-1])
    scales = torch.abs(out).amax(axis=1, keepdim=True)
    # if(fp32_scale):
    #     scales = scales.to(torch.float32)
    scales.div_(max_val)
    scales.clamp_(min=1e-6)
    out = tensor / scales
    out.clamp_(min_val, max_val)

    if not w_dtype.is_floating_point:
        out.round_()

    out = out.to(dtype=w_dtype)
    return out.view(out_shape), scales

@triton.jit
def round_triton_nvidia(tensor):
    return libdevice.round(tensor)

@triton.jit
def round_triton_amd(tensor):
    return libdevice.floor(tensor + 0.50)

if IS_HIP:
    round_triton = round_triton_amd
else:
    round_triton = round_triton_nvidia

@triton.jit
def scale_activations_per_token_kernel(
    tensor_ptr, scale_ptr, y_ptr, 
    M, K,
    stride_m, stride_k, stride_sm,
    ROUND: tl.constexpr, 
    UNROLL: tl.constexpr,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    fp32_scale: tl.constexpr, 
    BLOCK_M: tl.constexpr, 
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0) * UNROLL
    pid_k = tl.program_id(1)

    offs_k  = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    for m in range(UNROLL):
        mask = ((offs_m < M)[:, None] & (offs_k < K)[None, :]).to(tl.int1)
        in_ptrs = offs_m[:, None] * stride_m + offs_k[None, :] * stride_k
        tensor = tl.load(tensor_ptr + in_ptrs, mask=mask, other=0.0)
        if fp32_scale:
            tensor = tensor.to(tl.float32)

        scales_x = tl.max(tl.abs(tensor), axis=1, keep_dims=True)
        scales_x /= max_val
        scales_x = tl.maximum(scales_x, 1e-6)
        tensor /= scales_x
        tensor = tl.minimum(tl.maximum(tensor, min_val), max_val)

        if ROUND:
            tensor = round_triton(tensor)

        tl.store(scale_ptr + offs_m[:, None] * stride_sm, scales_x)
        tl.store(y_ptr + in_ptrs, tensor, mask=mask)
        offs_m += BLOCK_M


def scale_activations_per_token_triton(
    tensor: Tensor, w_dtype: torch.dtype, fp32_scale: bool = True
) -> Tuple[Tensor, Tensor]:
    min_val, max_val = get_dtype_range(w_dtype)
    x_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape
    scales = torch.empty(
        (M, 1), dtype=torch.float32 if fp32_scale else tensor.dtype, device=tensor.device
    )
    y = torch.empty((M, K), dtype=w_dtype, device=tensor.device)

    UNROLL = 1  # max(1, M // 128)
    BLOCK_M = 1
    BLOCK_K = triton.next_power_of_2(K)
    grid = (triton.cdiv(M, BLOCK_M * UNROLL), triton.cdiv(K, BLOCK_K))

    ROUND = not w_dtype.is_floating_point

    scale_activations_per_token_kernel[grid](
        tensor,
        scales,
        y,
        M,
        K,
        tensor.stride(0),
        tensor.stride(1),
        scales.stride(0),
        min_val=min_val,
        max_val=max_val,
        fp32_scale=fp32_scale,
        ROUND=ROUND,
        UNROLL=UNROLL,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_stages=1,
        num_warps=4,
    )

    return y.view(x_shape), scales

####################################################################################################################
#MXFP8
####################################################################################################################
@triton.jit
def next_power_of_2_log_triton(val, eps: tl.constexpr):
    exp = tl.ceil(tl.log2(val)).to(tl.int32)
    exp = tl.maximum(tl.minimum(exp, 254), 127 + eps_exp)
    scales = tl.where(exp >= 0, 1 << scales_log2, 1.0 / (1 << (-exp)))
    return scales, exp

@triton.jit
def next_power_of_2_logapprox_triton(val, eps_exp: tl.constexpr):
    exp = tl.inline_asm_elementwise(
        """
        {
        lg2.approx.f32 $1, $1;
        cvt.rpi.f32.f32 $1, $1;
        cvt.rzi.s32.f32 $0, $1;
        }
        """,
        "=r,r",
        [val],
        dtype=tl.int32, 
        is_pure=True,
        pack=1
    )

    exp = tl.maximum(tl.minimum(exp, 254), 127 + eps_exp)
    scales = tl.where(exp >= 0, 1 << exp, 1.0 / (1 << (-exp)))
    return scales, exp

@triton.jit
def next_power_of_2_bitwise_triton(val, eps_exp: tl.constexpr):
    xi = tl.cast(val, tl.uint32, bitcast=True)
    exp  = (xi >> 23) & 0xFF
    mant = xi & 0x7FFFFF
    exp += tl.where(mant != 0, 1, 0)
    exp = tl.maximum(tl.minimum(exp, 254), 127 + eps_exp)
    yi = exp << 23
    scales = tl.cast(yi, tl.float32, bitcast=True)
    return scales, exp

next_power_of_2_triton = next_power_of_2_bitwise_triton

@torch.compile(fullgraph=True)
def scale_activations_mxfp8_torch(
    tensor: Tensor, w_dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[Tensor, Tensor]:

    group_size: int = 32
    eps_exp: int = -30
    eps: float = 2 ** eps_exp
    min_val, max_val = get_dtype_range(w_dtype)

    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = tensor.shape

    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    if(pad_rows > 0):
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_rows))
    post_pad_shape = tensor.shape

    W_flat = tensor.view(-1, group_size).float()
    scales = W_flat.abs().amax(dim=1, keepdim=True)
    scales /= max_val
    scales = (2 ** torch.ceil(torch.log2(scales))).clamp_(eps) 

    W_q = (W_flat / scales).clamp_(min_val, max_val).to(w_dtype)
    if(pad_rows > 0):
        W_q = W_q.view(post_pad_shape)[:inter_shape[0], :]

    W_q = W_q.view(orig_shape)
    scales = (
        scales.to(torch.float8_e8m0fnu)
        .view(torch.uint8)
        .view(post_pad_shape[0], post_pad_shape[1] // group_size)
    )

    return W_q, scales

@triton.jit
def scale_activations_mxfp8_triton_v1_kernel(
    tensor_ptr,
    out_ptr,
    scales_ptr,
    E,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    eps_exp: tl.constexpr,
    UNROLL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0) * UNROLL

    for m in range(UNROLL):
        offs = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = (offs < E).to(tl.int1)
        tensor = tl.load(tensor_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        scales, scales_log2 = next_power_of_2_triton(tl.max(tl.abs(tensor)) / max_val, eps_exp)

        out = tensor / scales
        out = tl.clamp(out, min=min_val, max=max_val)
        out = out.to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, out)
        tl.store(scales_ptr + pid, scales_log2)

        pid += 1

def scale_activations_mxfp8_triton_v1(
    tensor: torch.Tensor, w_dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    group_size: int = 32
    eps_exp: int = -30
    eps: float = 2 ** eps_exp
    min_val, max_val = get_dtype_range(w_dtype)
    tensor = tensor.contiguous()
    
    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = tensor.shape
    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    post_pad_shape = (inter_shape[0] + pad_rows, inter_shape[1])
    E = tensor.numel()

    UNROLL = min(triton.cdiv(triton.cdiv(E, group_size), get_num_SMs(tensor.device)), 1)

    out = torch.empty(inter_shape, device=tensor.device, dtype=w_dtype)

    scales = torch.empty(
        (post_pad_shape[0], post_pad_shape[1] // group_size),
        device=tensor.device,
        dtype=torch.uint8,
    )
    
    grid = lambda meta: (triton.cdiv(E // UNROLL, group_size), )
    scale_activations_mxfp8_triton_v1_kernel[grid](
                tensor, 
                out, 
                scales, 
                E=E,
                min_val=min_val,
                max_val=max_val,
                eps_exp=eps_exp,
                UNROLL=UNROLL,
                GROUP_SIZE=group_size,
                num_stages=1,
                num_warps=4,
                )

    return out.view(orig_shape), scales

@triton.jit
def scale_activations_mxfp8_triton_kernel_v2(
    tensor_ptr,
    out_ptr,
    scales_ptr,
    M, K,
    stride_m_t, stride_k_t,
    stride_m_s, stride_k_s,
    stride_m_o, stride_k_o,
    #########################
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    eps_exp: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    #Load
    mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
    tensor_ptrs = tensor_ptr + (offs_m[:, None] * stride_m_t + offs_k[None, :] * stride_k_t)
    tensor = tl.load(tensor_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    #next power of 2 via log
    scales, scales_log2 = next_power_of_2_triton(tl.max(tl.abs(tensor), axis=1, keep_dims=True) / max_val, eps_exp)

    #Map to index
    out = tensor / scales
    out = tl.clamp(out, min=min_val, max=max_val)
    out = out.to(out_dtype)

    #Store
    out_mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
    tl.store(out_ptr + (offs_m[:, None] * stride_m_o + offs_k[None, :] * stride_k_o), out, mask=out_mask)

    offs_k = pid_k * 1 + tl.arange(0, 1)
    tl.store(scales_ptr + (offs_m[:, None] * stride_m_s + offs_k[None, :] * stride_k_s), scales_log2)


def scale_activations_mxfp8_triton_v2(
    tensor: torch.Tensor, w_dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    group_size: int = 32
    eps_exp: int = -30
    eps: float = 2 ** -30
    min_val, max_val = get_dtype_range(w_dtype)

    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    pad_m = (group_size - M % group_size) % group_size
    M_padded = M + pad_m

    out = torch.empty((M, K), device=tensor.device, dtype=w_dtype)
    scales = torch.empty((M_padded, K // group_size), device=tensor.device, dtype=torch.uint8)

    #BLOCK_SIZE_M = min(max(next_power_of_2(M), group_size), 128)
    BLOCK_SIZE_M = group_size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, group_size))
    device_index = tensor.device.index

    scale_activations_mxfp8_triton_kernel_v2[grid](
        tensor,
        out,
        scales,
        M, K, 
        tensor.stride(0), tensor.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        #########################
        min_val=min_val,
        max_val=max_val,
        eps_exp=eps_exp,
        GROUP_SIZE=group_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        num_stages=2,
        num_warps=4,
    )

    return out, scales


####################################################################################################################
#MXPF4 / NVFP4
####################################################################################################################
@torch.compile(fullgraph=True)
def scale_activations_mxfp4_torch(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    group_size: int = 32
    eps_exp: int = -30
    eps: float = 2 ** eps_exp
    max_val: float = 6

    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = tensor.shape

    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    if(pad_rows > 0):
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_rows))
    post_pad_shape = tensor.shape

    W_flat = tensor.view(-1, group_size).float()
    scales = W_flat.abs().amax(dim=1, keepdim=True)
    scales /= max_val
    scales = (2 ** torch.ceil(torch.log2(scales))).clamp_(eps)

    W_q = W_flat / scales
    if(pad_rows > 0):
        W_q = W_q.view(post_pad_shape)[:inter_shape[0], :]

    #1) Map to closest index
    device_index = W_q.device.index

    W_q = (
        (W_q.view(-1, 1) - fp4_values[device_index].to(W_q.dtype).view(1, -1))
        .abs()
        .argmin(dim=1)
        .to(torch.uint8)
        .view(inter_shape)
    )
    #2) Pack
    W_q = (W_q[:,::2] | W_q[:,1::2] << 4).to(torch.uint8)

    #Reshape scales
    scales = (
        scales.to(torch.float8_e8m0fnu)
        .view(torch.uint8)
        .view(post_pad_shape[0], post_pad_shape[1] // group_size)
    )
    return W_q, scales

@torch.compile(fullgraph=True)
def scale_activations_nvfp4_torch(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    group_size: int = 16
    eps: float = 1e-6
    max_val: float = 6
    fp8_dtype = torch.float8_e4m3fn #Support Nvidia only
    max_fp8 = torch.finfo(fp8_dtype).max #448

    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = tensor.shape

    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    if(pad_rows > 0):
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_rows))
    post_pad_shape = tensor.shape

    W_flat = tensor.view(-1, group_size).float()
    scales = W_flat.abs().amax(dim=1, keepdim=True)
    scales /= max_val
    meta_scales = NVFP4_META_SCALE #scales.max().clamp_(min=eps) - TODO: use max()
    scales /= meta_scales
    scales = scales.clamp(max=max_fp8).to(fp8_dtype).to(W_flat.dtype)

    W_q = W_flat / (scales * meta_scales)
    if(pad_rows > 0):
        W_q = W_q.view(post_pad_shape)[:inter_shape[0], :]

    #1) Map to closest index
    device_index = W_q.device.index

    W_q = (
        (W_q.view(-1, 1) - fp4_values[device_index].to(W_q.dtype).view(1, -1))
        .abs()
        .argmin(dim=1)
        .to(torch.uint8)
        .view(inter_shape)
    )
    #2) Pack
    W_q = (W_q[:,::2] | W_q[:,1::2] << 4).to(torch.uint8)

    #Reshape scales
    scales = (
        scales
        .to(fp8_dtype)
        .view(post_pad_shape[0], post_pad_shape[1] // group_size)
    )
    return W_q, scales

@triton.jit
def scale_activations_mxfp4_triton_kernel_v1(
    tensor_ptr,
    out_ptr,
    scales_ptr,
    thr_pos_ptr,
    E,
    eps_exp: tl.constexpr,
    UNROLL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0) * UNROLL

    HALF_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty
    thr_pos = tl.load(thr_pos_ptr + tl.arange(0, 8), eviction_policy='evict_last')[None, :]

    for m in range(UNROLL):
        #Load
        offs = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = (offs < E).to(tl.int1)
        tensor = tl.load(tensor_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        scales, scales_log2 = next_power_of_2_triton(tl.max(tl.abs(tensor)) / 6., eps_exp)

        #Map to index
        wq = tensor / scales
        idx_abs = tl.sum(tl.abs(wq[:, None]) > thr_pos, axis=1)
        out = tl.where(wq >= 0, idx_abs, idx_abs + 8).to(out_dtype)

        #Pack
        lo, hi = tl.split(out.reshape((HALF_GROUP_SIZE, 2), can_reorder=False))
        out = lo | (hi << 4)

        #Store
        offs_out = pid * HALF_GROUP_SIZE + tl.arange(0, HALF_GROUP_SIZE)
        tl.store(out_ptr + offs_out, out)
        tl.store(scales_ptr + pid, scales_log2)

        pid += 1

def scale_activations_mxfp4_triton_v1(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    group_size: int = 32
    eps_exp: int = -30
    eps = 2 ** eps_exp
    tensor = tensor.contiguous()
    
    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = (tensor.shape[0], tensor.shape[1] // 2)
    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    post_pad_shape = (inter_shape[0] + pad_rows, inter_shape[1])
    E = tensor.numel()

    UNROLL = min(triton.cdiv(triton.cdiv(E, group_size), get_num_SMs(tensor.device)), 1)

    out = torch.empty(inter_shape, device=tensor.device, dtype=torch.uint8)
    scales = torch.empty(
        (post_pad_shape[0], post_pad_shape[1] * 2 // group_size),
        device=tensor.device,
        dtype=torch.uint8,
    )
    device_index = tensor.device.index
    
    grid = lambda meta: (triton.cdiv(E // UNROLL, group_size), )
    scale_activations_mxfp4_triton_kernel_v1[grid](
                tensor, 
                out, 
                scales,
                thr_pos[device_index],
                E,
                eps_exp=eps_exp,
                UNROLL=UNROLL,
                GROUP_SIZE=group_size,
                num_stages=1,
                num_warps=4,
                )

    return out, scales


@triton.jit
def scale_activations_mxfp4_triton_kernel_v2(
    tensor_ptr,
    out_ptr,
    scales_ptr,
    thr_pos_ptr,
    M, K,
    stride_m_t, stride_k_t,
    stride_m_s, stride_k_s,
    stride_m_o, stride_k_o,
    #########################
    eps_exp: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    HALF_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty
    thr_pos = tl.load(thr_pos_ptr + tl.arange(0, 8), eviction_policy='evict_last')[None, :]

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    #Load
    mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
    tensor_ptrs = tensor_ptr + (offs_m[:, None] * stride_m_t + offs_k[None, :] * stride_k_t)
    tensor = tl.load(tensor_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    #next power of 2 via log
    scales, scales_log2 = next_power_of_2_triton(tl.max(tl.abs(tensor), axis=1, keep_dims=True) / 6., eps_exp)

    #Map to index
    wq = tensor / scales
    idx_abs = tl.sum(tl.abs(wq[:, :, None]) > thr_pos[None, :, :], axis=2)
    out = tl.where(wq >= 0, idx_abs, idx_abs + 8).to(out_dtype)

    #Pack
    lo, hi = tl.split(out.reshape((BLOCK_SIZE_M, HALF_GROUP_SIZE, 2), can_reorder=False))
    out = lo | (hi << 4)

    #Store
    offs_k = pid_k * HALF_GROUP_SIZE + tl.arange(0, HALF_GROUP_SIZE)
    out_mask = ((offs_m[:, None] < M) & (offs_k[None, :] < (K // 2))).to(tl.int1)
    tl.store(out_ptr + (offs_m[:, None] * stride_m_o + offs_k[None, :] * stride_k_o), out, mask=out_mask)

    offs_k = pid_k * 1 + tl.arange(0, 1)
    tl.store(scales_ptr + (offs_m[:, None] * stride_m_s + offs_k[None, :] * stride_k_s), scales_log2)

def scale_activations_mxfp4_triton_v2(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    group_size: int = 32
    eps_exp: int = -30
    eps: float = 2 ** eps_exp

    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    pad_m = (group_size - M % group_size) % group_size
    M_padded = M + pad_m

    out = torch.empty((M, K // 2), device=tensor.device, dtype=torch.uint8)
    scales = torch.empty((M_padded, K // group_size), device=tensor.device, dtype=torch.uint8)

    #BLOCK_SIZE_M = min(max(next_power_of_2(M), group_size), 128)
    BLOCK_SIZE_M = group_size
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, group_size))
    device_index = tensor.device.index

    scale_activations_mxfp4_triton_kernel_v2[grid](
        tensor,
        out,
        scales,
        thr_pos[device_index],
        M, K, 
        tensor.stride(0), tensor.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        #########################
        eps_exp=eps_exp,
        GROUP_SIZE=group_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        num_stages=2,
        num_warps=4,
    )

    return out, scales

@triton.jit
def scale_activations_nvfp4_triton_kernel_v2(
    tensor_ptr,
    out_ptr,
    scales_ptr,
    thr_pos_ptr,
    M, K,
    stride_m_t, stride_k_t,
    stride_m_s, stride_k_s,
    stride_m_o, stride_k_o,
    #########################
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    meta_scales: tl.constexpr = NVFP4_META_SCALE,
):

    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    fp8_dtype: tl.constexpr = tl.float8e4nv
    max_fp8: tl.constexpr = 448.
    HALF_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty
    thr_pos = tl.load(thr_pos_ptr + tl.arange(0, 8), eviction_policy='evict_last')[None, :]
    #thr_pos += 1e-6

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    #Load
    mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
    tensor_ptrs = tensor_ptr + (offs_m[:, None] * stride_m_t + offs_k[None, :] * stride_k_t)
    tensor = tl.load(tensor_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    #FP8 scales
    scales = tl.max(tl.abs(tensor), axis=1, keep_dims=True) / (6. * meta_scales)
    scales = tl.minimum(scales, max_fp8).to(fp8_dtype)

    #Map to index
    scales_full = tl.maximum(scales.to(tl.float32) * meta_scales, eps)
    wq = tensor / scales_full
    idx_abs = tl.sum(tl.abs(wq[:, :, None]) > thr_pos[None, :, :], axis=2)
    out = tl.where(wq >= 0, idx_abs, idx_abs + 8).to(out_dtype)

    #Pack
    lo, hi = tl.split(out.reshape((BLOCK_SIZE_M, HALF_GROUP_SIZE, 2), can_reorder=False))
    out = lo | (hi << 4)

    #Store
    offs_k = pid_k * HALF_GROUP_SIZE + tl.arange(0, HALF_GROUP_SIZE)
    out_mask = ((offs_m[:, None] < M) & (offs_k[None, :] < (K // 2))).to(tl.int1)
    tl.store(out_ptr + (offs_m[:, None] * stride_m_o + offs_k[None, :] * stride_k_o), out, mask=out_mask)

    offs_k = pid_k + tl.arange(0, 1)
    tl.store(scales_ptr + (offs_m[:, None] * stride_m_s + offs_k[None, :] * stride_k_s), scales)


def scale_activations_nvfp4_triton_v2(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    group_size: int = 16
    eps: float = 1e-6
    fp8_dtype = torch.float8_e4m3fn #Nvidia only

    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    pad_m = (group_size - M % group_size) % group_size
    M_padded = M + pad_m

    out = torch.empty((M, K // 2), device=tensor.device, dtype=torch.uint8)
    scales = torch.empty((M_padded, K // group_size), device=tensor.device, dtype=fp8_dtype)

    #BLOCK_SIZE_M = min(max(next_power_of_2(M), group_size), 128)
    BLOCK_SIZE_M = group_size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, group_size))
    device_index = tensor.device.index

    scale_activations_nvfp4_triton_kernel_v2[grid](
        tensor,
        out,
        scales,
        thr_pos[device_index],
        M, K, 
        tensor.stride(0), tensor.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        #########################
        eps=eps,
        GROUP_SIZE=group_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        num_stages=2,
        num_warps=4,
    )

    return out, scales

####################################################################################################################
scale_activations_per_token = scale_activations_per_token_triton
scale_activations_mxfp8 = scale_activations_mxfp8_triton_v2
scale_activations_mxfp4 = scale_activations_mxfp4_triton_v2
scale_activations_nvfp4 = scale_activations_nvfp4_triton_v2
