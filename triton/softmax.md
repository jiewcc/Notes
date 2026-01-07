### softmax实现

[Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

##### 原始pytorch进行softmax

![](../图片/20260107112952_78805a221a988e79ef3f42d7c5bfd418.jfif)

```python
def naive_softmax(x):
    # x.shape = (M, N)
    # 步骤1: 计算每行的最大值（for数值稳定性）
    # 内存访问: 读MN，写M
    x_max = x.max(dim=1, keepdim=True).values  # 读x的所有元素(MN)，写最大值(M)
    
    # 步骤2: 减去最大值
    # 内存访问: 读MN+M，写MN
    x_exp = x - x_max  # 读x(MN)和x_max(M)，写x_exp(MN)
    
    # 步骤3: 计算指数
    # 内存访问: 读MN，写MN
    x_exp = torch.exp(x_exp)  # 读x_exp(MN)，写回x_exp(MN)
    
    # 步骤4: 计算每行和
    # 内存访问: 读MN，写M
    exp_sum = x_exp.sum(dim=1, keepdim=True)  # 读x_exp(MN)，写和(M)
    
    # 步骤5: 归一化
    # 内存访问: 读MN+M，写MN
    y = x_exp / exp_sum  # 读x_exp(MN)和exp_sum(M)，写y(MN)
    
    return y

def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]   # M
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret
```

- x_max[:, None] 这部分操作的作用是将 x_max 这个一维张量的形状从 变为 。这是通过在 x_max 的最后一个维度（索引为 1 的维度）添加一个大小为 1 的维度来实现的。在 PyTorch 中，None 是 NoneType 类型的唯一实例，用于在张量的形状中添加一个大小为 1 的维度。（3）和（3，）
- x - x_max[:, None] 这个操作是张量的广播（broadcasting）操作。因为 x 的形状是 ，而 x_max[:, None] 的形状是 ，根据广播规则，x_max[:, None] 会在最后一个维度上自动扩展为 ，然后和 x 进行逐元素的减法操作。

能不能写一个triton内核，分别读写$M*N$次实现softmax？

##### 代码理解

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```

`output_ptr, input_ptr`输入输出指针

`input_row_stride, output_row_stride`行偏移

`n_rows, n_cols`输入输出的行数和列数

`BLOCK_SIZE: tl.constexpr`block内的线程数。编译期常量

`num_stages: tl.constexpr`流水线阶段的数量。流水线阶段用于优化内存访问和计算性能。具体作用未知

```python
    row_start = tl.program_id(0)   # 第几个block，triton把一个block计算的过程叫program
    row_step = tl.num_programs(0)  # block的数量
```

```python
for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
```

- row_start确定了当前程序块的起始行。
- n_rows是矩阵的总行数，循环会从row_start开始，一直到超过n_rows结束。
- row_step是步长，每次循环时row_idx会增加row_step的值。
- num_stages是流水线阶段的数量，它会影响循环的执行方式，特别是在并行计算中。

```
# 假设 BLOCK_SIZE = 16，那么执行 col_offsets = tl.arange(0, BLOCK_SIZE) 后，col_offsets 的值将是：
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
```

我们可以创建一个辅助函数，该函数为任何给定的输入张量，将内核需要的各种（元）参数，针对输入张量拿到。

```python
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}
# 传入DEVICE.index作为参数，获取当前设备的属性信息。properties，通常是一个字典，包含了设备的各种硬件参数。
# 流式处理器（SM）数量
# 每个线程的最大寄存器数量。
# 每个线程块的最大共享内存大小。
# 每个warp（线程束）的大小。
# target：存储当前的运行目标。
# 初始化一个空字典kernels，用于存储后续定义的内核函数或其他相关数据。


def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`   块大小是大于`x`中列数的最小2的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software pipelining stages.
    # 当共享内存足够大时，可以设置更多的软件流水线阶段，这可能有助于提高程序的执行效率，因为更多的阶段可以更好地隐藏内存访问的延迟。
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()   #初始化内核执行所需的各种句柄，例如CUDA流（stream）、事件（event）或其他资源。
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    if is_hip():
        # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
        # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
        # ISA SECTION (3.6.4 for CDNA3)
        # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
        # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
        # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
        # not required to be equal numbers of both types.
        NUM_GPRS = NUM_REGS
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2

        # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
        # When we divide this number with WARP_SIZE we get maximum number of waves that can
        # execute on a CU (multi-processor)  in parallel.
        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)   # 在算gird上的block数量，来避免occupancy，找最优的num_programs

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y
```

### benchmark

```
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'naive_softmax'],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)
```

- {'M': 4096} 表示在基准测试中，参数 M 的值被固定为 4096。这意味着在测试过程中，无论 N 的值如何变化（N 是 x_names 中的参数，其值由 x_vals 定义），参数 M 的值始终为 4096。