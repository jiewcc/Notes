### Sglang Code Walkthrough

*2025.12.22*

![总体流程](../图片/sglang-architecture.svg)

| 步骤 | 执行者（谁）                  | 动作（干什么）                       | 作用（为什么）                                               |
| ---- | ----------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| 1    | **Scheduler** 本身            | `recv_requests`                      | 把门口信箱里所有新工单一次性拿进来。                         |
| 2    | **Scheduler** 本身            | `process_input_requests`             | 按类型分拣：生成/嵌入/撤销，把生成类单子留下。               |
| 3    | **Scheduler** 本身            | `handle_generate_request`            | 给每张生成单贴“序列号”，扔进 **waiting\_queue** 大筐。       |
| 4    | **Scheduler** 本身            | `get_next_batch_to_run`              | 从筐里挑能拼桌的单子，组成一张 **ScheduleBatch** 生产卡。    |
| 5    | **Scheduler** 本身            | `run_batch`                          | 把生产卡再包一层，变成 **ModelWorkerBatch** 交给车间。       |
| 6    | **TpModelWorker**（车间班长） | `forward_batch_generation`           | 接收批次，准备开工：先造出 **ForwardBatch** 工单。           |
| 7    | **TpModelWorker**（车间班长） | 把 ForwardBatch 递给 **ModelRunner** | 让真正的工人去跑模型。                                       |
| 8    | **ModelRunner**（工人）       | `forward_extend`                     | 调用模型 + **AttentionBackend** 电钻，算出 logits（概率表）。 |
| 9    | **ModelRunner**（工人）       | 把 logits 交回 **TpModelWorker**     | 班长拿到概率表。                                             |
| 10   | **TpModelWorker**（车间班长） | 调 `sample` 抽 next\_token\_ids      | 从概率表里“抽签”决定下一个 token 数字。                      |
| 11   | **TpModelWorker**（车间班长） | 把签号发回 **Scheduler**             | 告诉调度员“这批字我产完了”。                                 |
| 12   | **Scheduler** 本身            | `process_batch_result`               | 更新每张单的状态：写完没？没写完继续排队。                   |
| 13   | **Scheduler** 本身            | `tree_cache.cache_finished_req`      | 把已写完的单缓存起来，下次可复用 KV-cache，省钱。            |
| 14   | **Scheduler** 本身            | `check_finished`                     | 判断单是否真正完结：完结→送走；未完→留在筐里。               |
| 15   | **Scheduler** 本身            | `stream_output`                      | 把完结单封装成 **BatchTokenIDOut** 快递袋，丢给 DetokenizerManager 的信箱。 |

具体参考：[SGLang 后端代码解析](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme-CN.md)

#### KV-Cache

![kv解释](../图片/Snipaste_2025-12-22_10-51-33.png)

因此，在逐个生成token的过程中，最后方词的不断迭代更新，只需要最后一个词进行forward，然后attention过程中，只需要计算它的q，与kv-cache中各个token这一次attention的key值点乘得到weight，然后和value值加权得到attention后的$\Delta E$，之后继续forward。

因此，KV-Cache很必要，但Q-Cache并没有（生成新词的$\Delta E$不需要旧词的query向量）

此外，表格中的"把已写完的单缓存起来，下次可复用 KV-cache，省钱。"是对相同前缀词提问的记录，对多个提问的复用，如：

- 提问1：*为什么英雄联盟的海克斯大乱斗*很好玩？
- 提问2：*为什么英雄联盟的海克斯大乱斗*在上架后讨论度很高？

#### 启动 Server（launch Sever）

“设置 logging、Server 参数、CUDA/NCCL 环境变量、进程间通信端口”是干嘛？
logging  打印哪些信息、存哪、级别多高，方便查错。
Server 参数 绑定 IP、端口、最大请求数、批大小等。
CUDA/NCCL 环境变量  告诉 GPU 驱动和通信库 **怎么组队、怎么跨卡传数据**。
进程间通信端口  ZeroMQ 的“门牌号”，让 TokenizerManager ↔ Scheduler ↔ DetokenizerManager 能互相喊话。 

tp = tensor-parallel，rank = 在本组里的序号
例：tp=4，四张卡一起算一个矩阵，tp_rank=0/1/2/3 分别表示“我在这一列里的第几个”。

data parallel replicas：完整模型的克隆副本，每个副本跑在不同 GPU 组上，同时服务不同请求。
例：8 张卡，tp=4 → 2 个 replica，各 4 张卡，互不相干，一起接单。

为什么开启数据并行要启动多个 replicas？
一个 replica 同一时间只能 batch 有限请求；**克隆 N 份就能同时吞吐 N 倍请求**，实现“加机器就加速”。

“TokenizerManager 和 DetokenizerManager 仅在第一个节点运行”——节点是啥？
**物理机器/容器**（node），不是单张 GPU。
多机部署时，只有 node-0 跑“翻译官”和“反向翻译官”，别的 node 只跑 Scheduler + 模型，**省得每台都加载同一份 tokenizer 浪费内存和跨网络同步**。

chat template 就是**“模型期望的聊天格式模板”**，大模型预训练时只见过固定格式的对话文本，如果你直接扔纯文本 `“帮我写快排”`，模型可能**回答质量下降**。chat template 负责把“裸文字”自动包成模型“吃过”的格式。
模型文件夹里一般自带 `tokenizer_config.json`，里面写了 `"chat_template": "..."`

##### （在两个节点上使用 共计 16 张 H100 部署 Llama 3.1 405B）部署的结构和协同工作流程：

| 物理节点               | GPU                     | 内存 ≈ | 在集群里的花名       | 负责活儿                                                |
| ---------------------- | ----------------------- | ------ | -------------------- | ------------------------------------------------------- |
| **Node-0**<br>(主节点) | 8×H100<br>≈ 640 GB VRAM | 640 GB | “前台+翻译+第一车间” | HTTP、TokenizerManager、DetokenizerManager、Scheduler-0 |
| **Node-1**<br>(副节点) | 8×H100<br>≈ 640 GB VRAM | 640 GB | “第二车间”           | 仅 Scheduler-1 + 模型 Worker                            |

1. **Node-0 上总指挥 `launch_engine` 先跑**
   - 设置 NCCL 环境：`MASTER_ADDR=node0 IP, MASTER_PORT=29500`
   - 加载 **FP8 权重**（405 GB → 203 GB，省显存）
   - 起 **TokenizerManager**(主进程) + **DetokenizerManager**(子进程)
   - 对本机 8 卡 tp=8 起 **Scheduler-0**（子进程）
2. **Node-1 上同样命令，但 `--node-rank=1`**
   - NCCL 自动连到 node0，完成 **world-size=16** 初始化
   - 只起 **Scheduler-1**（子进程），**不再起 Tokenizer/Detokenizer**
   - 两节点的 Scheduler 通过 **ZeroMQ** 向同一个 TokenizerManager 汇报“我就绪”
3. **DataParallelController（在 node0）**
   - 维护 **2 个 replica** 的健康状态
   - HTTP 流量用 **round-robin** 先送到 node0 或 node1 的 Scheduler；未来会换 **SGLang Router**

- **Tensor Parallelism(tp=8)** 在 **节点内** 完成；
- **Data Parallelism(dp=2)** 在 **节点间** 完成；
- KV-cache 只在本节点 8 卡间共享，**跨节点不共享**

ZeroMQ：
就像一条“**轻量级快递专线**”，专门帮不同进程（或不同机器）之间**快速收发小包裹（消息）**
是一个**独立的开源库**（libzmq），用 C 写成，但给 Python、Java、Go、Rust… 都提供了“pip-install 级”的语言绑定

#### 转发请求 (Forward Requests From Server)

| 步骤 | 谁                         | 怎么执行                                                     | 起什么作用                                                   |
| ---- | -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | **FastAPI 框架**           | 在代码里用装饰器 `@app.post("/v1/chat/completions")` 注册路由 | 把普通 Python 函数 `v1_chat_completions` 变成真正的“网页路口”，谁来 POST 都进这里。 |
| 2    | `v1_chat_completions` 函数 | 读 `raw_request.body` → `json()` → 校验成 **ChatCompletionRequest** 数据类 | 把原始 JSON 变成带代码提示的“结构化发票”，字段写错会立刻弹 422 错误。 |
| 3    | 同一函数内                 | 调辅助 `v1_chat_generate_request()` →**GenerateReqInput** 内部工单 + 填好 `sampling_params`（温度、top\_p、max\_tokens…） | 转成 SGLang 后台看得懂的“生产指令单”，温度、长度等备注一次到位。 |
| 4    | 仍在这个函数               | 调 `TokenizerManager.generate_request(req_input)` **并 await** | 把工单塞进 ZeroMQ 快递口，然后**原地阻塞/异步等待**直到后台把第一个 token 产出来。 |
| 5    | 拿到后台返回后             | 看 `stream` 字段：真→走 `generate_stream_resp()`；假→走 `v1_chat_generate_response()` | 决定是“边产边发”（打字效果）还是“全部收完再一次性给”。       |
| 6    | **流式分支**               | `generate_stream_resp` 是一个 **async 生成器**，内部不断 `yield` 小段 JSON | FastAPI 把每次 `yield` 立即塞进 HTTP chunk，浏览器看到就是实时跳动文字。 |
| 7    | **非流式分支**             | 直接 `await` 整个结果 → 调 `v1_chat_generate_response` 包成 **ChatCompletionResponse** → `ORJSONResponse` 回客户端 | 一次过给出完整回答，节省连接数，方便批量脚本解析。           |

FastAPI 是什么？
一个 Python 库，写几行就能搭出带自动文档、自动校验的 Web 接口。
负责监听 IP+端口，把浏览器/代码发来的 HTTP 包转成 Python 对象，再把 Python 结果转回 HTTP。

API endpoint：路径 + 方法
例子：`POST /v1/chat/completions` 这个字符串就是 endpoint；谁对它发 POST，FastAPI 就调对应函数。

请求本身是 HTTP 的 **请求体（body）**，格式是 JSON，例如：

```
{
  "model": "llama-3.1-405b",
  "messages": [{"role": "user", "content": "写快排"}],
  "stream": true,
  "temperature": 0.7
}
```

v1_chat_completions 只是转发吗？
不止转发，还负责：
① 把原始 JSON 校验成数据类 → ② 转成内部工单 → ③ 调后厨 → ④ 把后厨结果包成 OpenAI 兼容格式并流/非流地发回去。

generate_request 是 **后厨入口**，它只接收“已经转好的 GenerateReqInput”，然后做 tokenize + 发 ZMQ + 等结果。

“通过 v1_chat_generate_request 配置 sampling_params”在干什么？
v1_chat_generate_request 是个**普通函数**（不是类），名字前带 v1 只是**遵守 OpenAI 路径习惯（/v1/…）**。
它把 ChatCompletionRequest 里的 temperature、top_p、max_tokens 等字段 **抄到内部采样结构体 SamplingParams**，让后面模型知道“怎么抽样”。

##### 流式 vs 非流式响应区别

| 项目     | 流式 (stream=true)                           | 非流式 (stream=false)             |
| -------- | -------------------------------------------- | --------------------------------- |
| 返回节奏 | **每生成一个 token 立刻推一段**（SSE chunk） | **全部生成完一次性返回整段 JSON** |
| 用户体验 | 像 ChatGPT 实时“打字”                        | 等几秒后整段出现                  |
| 网络连接 | HTTP 长连接，分段传输                        | 普通短连接，一次打完              |
| 代码解析 | 前端需循环 `for chunk in response:`          | 直接 `response.json()` 即可       |

stream 参数什么时候设置？
用户发请求时自己写："stream": true 或 "stream": false
不写默认 false（非流式）。

Sampling（采样）就是 “从模型给出的概率表里抽签”，决定下一个 token 到底写谁。

1. 模型输出的是 logits（一长串概率），比如：

​	好: 35%， 的: 20%， 快: 15%， …

​	如果直接选概率最高的，文章会死板重复；Sampling 负责 按概率+随机性 抽一个，既合理又多样。

2. 常见采样“旋钮”

- temperature：概率温度。值高 → 更随机；值低 → 更确定。
- top_p：只从累积概率前 p% 的候选里抽，过滤长尾。
- top_k：只从概率最高的 k 个词里抽。

组合起来就能控制“创意 vs 稳定”的平衡。

#### TokenizerManager 生成请求（Generate Request In TokenizerManager）

检查 `update_weights_from_disk`：防止热更新权重时前后不一致，写完再接待新单。

看请求类型 vs 模型 `is_generation` 标志，不配就抛 400

调 `normalize_batch_and_arguments` → 补默认温度、top_p、max_tokens，并行采样、批处理，统一后续代码处理格式。

收到 Scheduler 的 `BatchTokenIDOut` → 调 DetokenizerManager 反向翻译 → 拿到 `BatchStrOut`

根据 stream 标志：流式→yield 每段文字；非流式→一次性返回完整对象

**TokenizerManager的批处理**取决于**GenerateReqInput 数据类**字段上，主要就两条：

1. `n > 1`（并行采样数）
   用户 HTTP 里写了 `"n": 8`，前端转内部工单时该字段就直接=8；`_handle_batch_request` 一看 `n>1`，立刻把同一条消息复制 8 份，每份再单独 tokenize，于是形成 8 条子请求。
2. `messages` 本身是列表且长度>1
   如果用户一次发来多轮对话（`messages=[...多轮...]`），而模型支持 batch对话，也会被当作“批”处理；此时不再复制，而是按轮次顺序逐条 tokenize 后打包。

TokenizerManager **自己并不“攒包”**，它收到一次 `generate_request()` 调用就**立即**把这条请求切成 token 并通过 ZeroMQ 塞进 Scheduler 的 waiting_queue；攒批（batching）是 **Scheduler 后面的事情**。

#### Scheduler 接收请求以及处理批次 (Scheduler Receive Requests and Process Batches)

![Scheduler 概览](../图片/sglang_scheduler.svg)

##### 名词解释

| 名词                              | 是什么                                                       | 干嘛用                                                       |
| --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **server\_args**                  | 命令行或脚本传进来的“大杂烩”配置对象（Python 的 SimpleNamespace/dataclass） | 告诉整个系统：用几张卡、温度多少、端口几号、日志级别、是否启用 stream、模型路径… |
| **port\_args**                    | server\_args 的子集，专门放 **ZeroMQ 端口 号** 和 **NCCL 端口 号** | 让不同进程间有固定“门牌号”可喊人，避免端口冲突。             |
| **model\_config**                 | 从模型文件夹 `config.json` 读出来的 \*\* HuggingFace 配置 dict \*\*，被包成 Python 对象 | 拿隐藏层数、head 数、vocab\_size、RoPE 参数等，后面建模型、算显存、分 tensor-parallel 都靠它。 |
| **sessions**                      | 一个 **dict\[session\_id, Session]**，存多轮对话的上下文 KV-cache 句柄 | 实现“继续聊”：同一 session\_id 再来请求，Scheduler 直接找到旧缓存接着生成，不用重算历史。 |
| **metrics**                       | Prometheus 格式的 **计数器/直方图** 集合（token 数、延迟、队列长度） | 让运维看面板：每秒处理多少 token、平均延迟多少，方便做告警和弹性伸缩。 |
| **多模态图像处理器 placeholders** | 提前把 `CLIPImageProcessor` 或 `Qwen-VL 视觉编码器` 实例化好，占个坑 | 后面收到带图片的请求时，可直接调用，不必临时加载模型，减少首包延迟。 |

*2025.12.23*

`run_scheduler_process`负责在子进程里把真正的火锅店长`Scheduler`实例化并开机。`run_scheduler_process`是入口函数（在主进程里被`launch_engine`用`multiprocessing.Process`拉起，函数里直接`scheduler = Scheduler(...)`然后`scheduler.run()`

**分词器**=`tokenizer`（HuggingFace 的`AutoTokenizer`）负责把字符串→List[int]。
**处理器**=多模态时用，如`CLIPImageProcessor`把图片→张量，或Qwen-VL的视觉编码器。

**ChunkCache**=按固定长度块缓存，简单但可能重复存。
**RadixCache**=前缀树共享，相同前缀只存一次，省显存。
SGLang默认启用RadixCache；ChunkCache只在旧版本或调试开关可见，二者**不会同时开**，Radix优先。

SchedulePolicy是Scheduler里的拼桌策略类。每轮按`max_batch_size`、`max_tokens`、`waiting time`等打分，挑一组请求下锅。平衡吞吐与延迟，实现“先来的先吃+能拼就拼”。

chunk prefill是prefill的一种方式吗？
**是**。它把**超长提示**切成≤chunk_size的小段，逐段extend，避免一次占爆显存。
**vs普通prefill**：普通prefill一次性算整句；chunk prefill=“分段extend再拼接KV”，仍是extend内核。

GrammarBackend
一个与Scheduler同级的服务进程。接收constraint请求→编译语法→每步给Scheduler返回合法token掩码。
让constraint decoding有“合法字典”，保证输出100%合规JSON/正则等。

constraint decoding
每步只从符合语法（JSON、正则、EBNF）的候选token里采样，让模型输出必定满足格式，如{"name":"...","age":...}不会缺括号。

关于prefill和decode概念，可以看[LLM Inference at scale with TGI](https://huggingface.co/blog/martinigoyanes/llm-inference-at-scale-with-tgi)前半部分。
TGI：Text Generation Inference（*TGI*）是HuggingFace推出的*大模型*推理部署框架

明确一些概念：
*"forward = 给定输入，计算输出的整个过程"*
Tokenizer 分词 + 转成数字(token_ids)，Embedding 查表转成向量

“Ragged Tensors 增量更新现有的 KV-Cache”问了gpt半天，依然没懂，回头细看吧。

### TpModelWorker 管理 forward pass 和 token sampling (TpModelWorker Manage Forward and Token Sampling)

embedding 请求就是“我不要模型继续写字，**只要它给我一段向量**”——相当于让模型当“特征提取器”。

### ModelRunner 管理模型执行 (ModelRunner Manages Model Execution)

**空闲** = 暂时没有用户请求，但 GPU 不能晾着；系统故意跑‘空锅’前向，用来**预热权重、刷 metrics、占住 CUDA 流，防止驱动降频**。

- **防降频**：GPU 空闲久了驱动会降功耗，下次真来请求时热身慢→首包延迟飙高。
- **保持 CUDA context 热**：权重一直留在寄存器/L2，不被换出。
- **顺手刷 metrics**：空跑也能测一次端到端 latency、带宽占用，方便面板显示“健康心跳”。
- **占流防抢占**：让 CUDA stream 一直有任务，避免其他进程插进来占资源。

### Model 加载权重并执行前向传递 (Model Load Weights and Perform Forward)

`lm_head` 就是 **最后一层线性投影**（权重形状 `[hidden_size, vocab_size]`），把每个 token 的隐藏向量乘一次矩阵 → 得到该位置在词表上的 logits 分数。

`pooler` 只在 **embedding/reward 任务** 被调用，对最后一层 hidden states 做平均或取最后一个 token，再投影到 `[batch, embed_dim]` 返回。

### AttentionBackend 加速模型前向传递 (AttentionBackend Accelerate Model Forward)

sliding window：只让 token 看「左边固定 W 个 token」的局部 attention（减少长文计算量）。
cross-attention：decoder  token 去 attend encoder 特征（多模态或 encoder-decoder 模型才用）。

metadata：元数据

CUDA Graph = GPU 的"宏录制"功能
把多个CUDA操作录下来，以后一键执行，大幅减少开销

```
# CPU 不断给 GPU 下指令
for 每个批次 in 数据:
    # 1. CPU: 启动kernel1
    kernel1<<<...>>>(...)
    
    # 2. CPU等待GPU完成
    cudaDeviceSynchronize()
    
    # 3. CPU: 启动kernel2
    kernel2<<<...>>>(...)
    
    # 4. 又等待...
    cudaDeviceSynchronize()
    
# 问题：CPU-GPU 频繁通信，大量时间浪费在"对话"上
```

FlashInfer wrapper
**C++ 类对象**，把 kernel 参数、workspace、调度逻辑包成“一键启动器”，让上层代码一句 `wrapper.run()` 就能调对 kernel。

prefix extension 是什么？

- 场景：旧前缀 `"The weather"` 已缓存，用户继续 `" is fine today"`。
- 工作：只算 `" is fine today"` 这段新 token 的 extend，旧 KV 直接索引不再计算。
- 结果：节点仍挂在 radix 树原前缀下，**物理上形成“旧块 + 新块”链条**；下次若再扩，继续只算新增即可。
- 与 ragged/paged 关系：
  - 旧块 → 必然 paged（已固定块化）
  - 新块 → 根据新 token 量选 ragged 或 paged
    → 所以 prefix extension 既可能走 paged 也可能走 ragged，只看“新增量”大小。

`indices_updater_prefill`更新索引：这个索引是 **“GPU 端 KV-Cache 物理地址索引”**，也就是告诉 CUDA kernel：“每条序列的新 token 该写到哪块显存、该从哪块旧显存读 KV”。

### DetokenizerManager 进行解码并发送回 TokenizerManager (DetokenizerManager Detokenize and Send to TokenizerManager)

为什么整理输出还要 metadata？

纯文本缺少 API 规定的字段：

- `finish_reason`：stop / length / content_filter

- `logprobs`：每个 token 的 -log(p)
- `usage`：prompt_tokens / completion_tokens

这些信息在 **Scheduler 阶段就已算好**并随 `BatchTokenIDOut` 一起发到 DetokenizerManager；
最后必须把 **文本 + 上述结构化数据** 合体成 `BatchStrOut`，否则前端无法构造符合 OpenAI 规范的 JSON 响应。

surr_ids 到底是什么？
当用户一次要 **n=3** 份答案（或 beam=3）时，同一段 token 序列会出现 **3 条略微不同的分支**。
DetokenizerManager 把它们扁平成一整条 `token_ids = [branch1; branch2; branch3]` 一起发过来；
`read_ids` = 真正要解码的主分支（或第一条），
`surr_ids` = 其余 **并行/候选分支**的 ID 数组。
用 **batch_decode** 一次性反编码整串，比多次调用快；解码后再按长度切回去即可。

