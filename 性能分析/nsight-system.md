## NSight-System安装指南及使用

### 单个批处理性能测试

```bash
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python3 -m sglang.bench_one_batch --model-path /root/.cache/huggingface/DeepSeek-R1-Distill-Qwen-1.5B --batch-size 8 --input-len 512
```

### 安装nsys-ui

容器宿主机安装nsight-systems

直接apt安装版本过旧，打不开容器内新版本nsys生成的.nsys-rep包，下载安装新版本

[nsight-systems](https://developer.download.nvidia.com/devtools/nsight-systems/)

下载nsight-systems-2025.5.1_2025.5.1.121-1_amd64.deb，安装

```bash
sudo apt install ./nsight-systems-2025.5.1_2025.5.1.121-1_amd64.deb
```

### nsys-ui打开report文件

```bash
nsys-ui /home/jiew/.cache/huggingface/report1.nsys-rep 
```

*仍然打不开容器内新版本nsys生成的.nsys-rep包*....

### X11转发图形界面

宿主机临时开放X Server访问权限，允许Docker容器连接到宿主机的X11服务器

```
xhost +
# 或
xhost +local:docker
```

重新启动容器，挂载X11套接字，设置DISPLAY环境变量，并在容器中执行一些列X11配置命令

```bash
docker run -it --rm\
    --gpus all \
    -e DISPLAY=$DISPLAY \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --env "HF_TOKEN=hf_xxx" \
    --ipc=host \
    lmsysorg/sglang:latest bash -c " \
    apt update && \
    apt install -y x11-apps && \
    xclock " 
    
docker run -it \
    --gpus all \
    -e DISPLAY=host.docker.internal:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
#    -v $HOME/.Xauthority:/root/.Xauthority \
    --name sglang-x11 \
    your-image-name
    
docker run -it \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    --name x11-docker \
    ubuntu bash -c " \
    apt update && \
    apt install -y x11-apps && \
    xclock "                      # 在容器内安装 x11-apps 包，并运行 xclock 应用，显示一个图形化的时钟。
```

下载nsys-ui

Chromium内核

```
# 禁用所有GPU加速
export QT_QPA_PLATFORM=xcb
export QT_QUICK_BACKEND=software
export LIBGL_ALWAYS_SOFTWARE=1
export QSG_RENDER_LOOP=basic
export QT_OPENGL=software
export QT_XCB_NO_XI2=1
```

```bash
nsys-ui /sgl-workspace/sglang/report1.nsys-rep
```

```
nsys export /sgl-workspace/sglang/report1.nsys-rep --type=json -o /root/.cache/huggingface/analysis.json
```

*容器内需要chrome内核什么的，失败...*

### 最新发现

既然容器内能找到2025.6.1版本的包源，能不能查看这个包源，然后在宿主机中进行下载呢？

[有Nsys的2025.6.1版本的网站](https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/)

容器内找包源...哭了...聪明的一批

卸载原来安装的旧2025.5.1版本nsys-ui

```bash
# 完全卸载当前版本
sudo apt remove --purge nsight-systems-2025.5.1 nsight-systems
sudo apt autoremove
sudo apt clean

# 确认卸载
nsys-ui --version 2>/dev/null || echo "已卸载"
```

安装下载的2025.6.1版本的deb

```bash
# 安装新版本
sudo dpkg -i nsight-systems-2025.6.1_2025.6.1.190-1_amd64.deb
```

检查安装是否成功

```bash
nsys-ui --version

# 输出
OpenGL version: "4.6.0 NVIDIA 580.95.05"
NVIDIA Nsight Systems 2025.6.1 (build 2025.6.1.190-256136895201v0) (public-release)
```

## 总体性能瓶颈分析与针对性优化操作步骤

查看过去创建的容器，打开容器

```bash
docker ps -a
docker start upbeat_leakey
docker exec -it upbeat_leakey bash
```

生成.nsys-rep文件（批处理）

laptop上：

```bash
# 增加内存分配比例
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  python3 -m sglang.bench_one_batch \
    --model-path /root/.cache/huggingface/DeepSeek-R1-Distill-Qwen-1.5B \
    --batch-size 1 \
    --input-len 16 \
    --mem-fraction-static 0.95 \
    --dtype half \
    --max-running-requests 1 \
    --max-total-tokens 32 \
    --disable-cuda-graph
```

```bash
docker exec -it upbeat_leakey bash

python3 -c "
import torch
torch.cuda.empty_cache()
torch.cuda.synchronize()
print('内存已清理')
"
```

*内存没问题了，RMSnorm不支持sm75 Turing架构的实现....，当时不是成功跑起来了吗？*

PC上：

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  python3 -m sglang.bench_one_batch \
    --model-path /root/.cache/huggingface/DeepSeek-R1-Distill-Qwen-1.5B \
    --batch-size 1 \
    --input-len 16 \
    --mem-fraction-static 0.95 \
    --dtype half \
    --max-running-requests 1 \
    --max-total-tokens 32 \
    --disable-cuda-graph
```

拷贝：

```bash
cp report1.nsys-rep /root/.cache/huggingface
```

### NVTX

copilot改了sglang的源码，牛，nvtx trace了算子的调用，

```
export SGL_KERNEL_NVTX=1

nsys profile \
  --trace=cuda,nvtx,osrt \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --python-backtrace=cuda \
  --force-overwrite=true \
  -o report_sgl_kernel_nvtx \
  python3 -m sglang.bench_one_batch \
    --model-path /root/.cache/huggingface/DeepSeek-R1-Distill-Qwen-1.5B \
    --batch-size 1 \
    --input-len 16 \
    --mem-fraction-static 0.95 \
    --dtype half \
    --max-running-requests 1 \
    --max-total-tokens 32 \
    --disable-cuda-graph
```

### CUDA Python 回溯

[Nsight System官方文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-trace)

Nsight Systems 支持在分析过程中收集 Python 回溯信息，这对于调试和优化使用 CUDA 的 Python 应用程序非常有用。

- **命令行界面 (CLI)**：在启动分析时，添加 `--python-backtrace=cuda` 选项。

这种功能允许开发者在 CUDA 调用发生时，查看对应的 Python 堆栈信息，从而更方便地定位和调试代码中的问题。

```
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --python-backtrace=cuda \
  python3 -m sglang.bench_one_batch \
    --model-path /root/.cache/huggingface/DeepSeek-R1-Distill-Qwen-1.5B \
    --batch-size 1 \
    --input-len 16 \
    --mem-fraction-static 0.95 \
    --dtype half \
    --max-running-requests 1 \
    --max-total-tokens 32 \
    --disable-cuda-graph
```

*感觉没什么用*

是有原因的，还得是AI太吊了...

- 采集时没真正启用**“CUDA backtrace”**
  在 Nsight Systems 里，Python backtrace 往往需要同时启用“CUDA backtraces”。你贴的官网截图里那一块就是 Collect CUDA backtraces（GUI 采集配置中的选项）。
  用 CLI 时，很多版本还需要显式打开 CUDA backtrace（不同版本参数名略有差异），常见是类似：

  - --cudabacktrace=true（或同类开关）
    如果只写 --python-backtrace=cuda，有些版本/场景下不会把栈真正采下来或不会显示

  - 打开cuda back trace：

    ```bash
    export SGL_KERNEL_NVTX=1
    
    nsys profile \
      --trace=cuda,nvtx,osrt \
      --cudabacktrace=true \
      --python-backtrace=cuda \
      --trace-fork-before-exec=true \
      --cuda-graph-trace=node \
      --force-overwrite=true \
      -o report_sgl_kernel_nvtx \
      python3 -m sglang.bench_one_batch \
        --model-path /root/.cache/huggingface/DeepSeek-R1-Distill-Qwen-1.5B \
        --batch-size 1 \
        --input-len 16 \
        --mem-fraction-static 0.95 \
        --dtype half \
        --max-running-requests 1 \
        --max-total-tokens 32 \
        --disable-cuda-graph
    ```


- 要真正看到 Python backtrace（推荐做法）
  先把**内核采样权限**打开（这是根因）

  - 在宿主机执行（因为容器共享宿主内核）：

  ```bash
  sudo sysctl -w kernel.perf_event_paranoid=1
  # （可选）sudo sysctl -w kernel.kptr_restrict=0
  ```

  - 再在容器里跑一次 nsys status --environment，**确认 perf_event_open 从 Fail 变成 OK**
    重新 profile 时把采样/回溯方法说清楚（你机器没有 LBR，所以别用默认 lbr）：
    加上 --sample=process-tree -b dwarf

  - 新的：

    ```bash
    export SGL_KERNEL_NVTX=1
    
    nsys profile \
      --trace=cuda,nvtx,osrt \
      --sample=process-tree \
      -b dwarf \
      --cudabacktrace=kernel:0 \
      --python-backtrace=cuda \
      --resolve-symbols=true \
      --debug-symbols=/usr/lib/debug \
      --trace-fork-before-exec=true \
      --force-overwrite=true \
      -o report_sgl_kernel_nvtx \
      python3 -m sglang.bench_one_batch \
        --model-path /root/.cache/huggingface/DeepSeek-R1-Distill-Qwen-1.5B \
        --batch-size 1 \
        --input-len 16 \
        --mem-fraction-static 0.95 \
        --dtype half \
        --max-running-requests 1 \
        --max-total-tokens 32 \
        --disable-cuda-graph
    ```

  - 容器已经无法更改，新建docker容器扩充权限，允许捕获内核活动（--privileged）：

    ```bash
    docker run -it --rm \
      --privileged \
      --gpus all \
      --shm-size 32g \
      -p 30001:30001 \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      --env "HF_TOKEN=hf_xxx" \
      --ipc=host \
      lmsysorg/sglang:latest
    ```

    ```bash
    nsys status --environment
    ```

    后perf_event_open等项值从 Fail 变成 OK

##### 指令分析

- `--sample=process-tree`

  - 启用 **CPU 采样**（profiling sampling），并且采样范围覆盖“被启动进程及其子进程树”。

  - 这一步是 `--python-backtrace=cuda` / `--cudabacktrace=...` 能拿到“可读调用栈”的关键前提之一（nsys 需要 CPU sampling 才能在触发点采集 backtrace）
  - 

- `-b dwarf`
  - 指定 CPU 采样时收集 backtrace 的方式为 **DWARF 回溯**（靠调试信息做栈展开）。
  - 特点：一般比 frame-pointer 更稳、比 LBR 更通用，但开销可能更高；如果二进制缺少 unwind 信息/调试信息，回溯质量会下降。

- `--resolve-symbols=true`
  - 让 nsys 在采集后对 backtrace 里的地址做 **符号解析**（把“地址/offset”解析成函数名/库名/尽可能的源码位置）。
  - 没开它时，你经常会看到 `Unknown[...]` 或只有地址，没有函数名。

- `--debug-symbols=/usr/lib/debug`
  - 告诉 nsys 去哪些目录找 **调试符号文件（debug symbols）**。
  - 在 Ubuntu 上，通过 `*-dbgsym` 包安装的调试符号通常会落在 [debug](vscode-file://vscode-app/snap/code/217/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)，配置这个目录能显著提高符号解析质量（更容易看到具体函数名，甚至行号）。
    *这个好像没啥用*

- `--trace-fork-before-exec=true`
  - 追踪“fork/exec 发生之前”的进程/线程行为，保证当目标程序在启动阶段发生 `fork()`/`exec()`（或启动子进程）时，nsys 也能把这些子进程纳入本次采集。
  - 对 Python、launcher、worker 进程模型很有用，否则容易出现“只采到父进程，关键工作在子进程里”的情况。
- `--force-overwrite=true`
  - 如果输出文件（`-o report_sgl_kernel_nvtx` 对应的 `.nsys-rep/.sqlite` 等）已存在，**强制覆盖**，避免 nsys 因为文件存在而拒绝运行或产生一堆带编号的输出文件。

*看起来就那个没啥用*





