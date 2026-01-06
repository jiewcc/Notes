```bash
rm -rf ~/.triton/cache
```

**跑同一套 bench_serving，观察指标变化**

```bash
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --dataset-name sharegpt --dataset-path /root/.cache/huggingface/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 100 --request-rate 20 --max-concurrency 64
```

融合前：

```bash
python3 -m sglang.launch_server \
    --model-path /root/.cache/huggingface/DeepSeek-R1-Distill-Qwen-1.5B \
    --host 0.0.0.0 \
    --port 30000
```

```bash
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    20.0      
Max request concurrency:                 64        
Successful requests:                     100       
Benchmark duration (s):                  34.24     
Total input tokens:                      34200     
Total input text tokens:                 34200     
Total input vision tokens:               0         
Total generated tokens:                  21513     
Total generated tokens (retokenized):    21500     
Request throughput (req/s):              2.92      
Input token throughput (tok/s):          998.96    
Output token throughput (tok/s):         628.38    
Peak output token throughput (tok/s):    1236.00   
Peak concurrent requests:                72        
Total token throughput (tok/s):          1627.34   
Concurrency:                             30.79     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   10540.12  
Median E2E Latency (ms):                 9590.18   
---------------Time to First Token----------------
Mean TTFT (ms):                          7265.44   
Median TTFT (ms):                        5789.71   
P99 TTFT (ms):                           16299.52  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          16.41     
Median TPOT (ms):                        15.32     
P99 TPOT (ms):                           31.67     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           15.29     
Median ITL (ms):                         13.82     
P95 ITL (ms):                            16.32     
P99 ITL (ms):                            47.68     
Max ITL (ms):                            675.66    
==================================================

```

融合后

```bash
rm -rf ~/.triton/cache
```

```bash
export SGLANG_MOE_FUSE_DOWN_SUM_REDUCE=1 
```

```bash
python -m sglang.launch_server --model-path /root/.cache/huggingface/DeepSeek-R1-Distill-Qwen-1.5B --host 0.0.0.0 --port 30000
```

条件收紧topk==2

export SGLANG_MOE_FUSE_DOWN_SUM_REDUCE=1

```
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 64        
Successful requests:                     100       
Benchmark duration (s):                  24.26     
Total input tokens:                      34684     
Total input text tokens:                 34684     
Total input vision tokens:               0         
Total generated tokens:                  18999     
Total generated tokens (retokenized):    18949     
Request throughput (req/s):              4.12      
Input token throughput (tok/s):          1429.82   
Output token throughput (tok/s):         783.22    
Peak output token throughput (tok/s):    1245.00   
Peak concurrent requests:                71        
Total token throughput (tok/s):          2213.04   
Concurrency:                             35.24     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   8547.93   
Median E2E Latency (ms):                 8916.10   
---------------Time to First Token----------------
Mean TTFT (ms):                          5527.82   
Median TTFT (ms):                        6101.90   
P99 TTFT (ms):                           9807.61   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          17.29     
Median TPOT (ms):                        15.98     
P99 TPOT (ms):                           46.62     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           15.98     
Median ITL (ms):                         14.19     
P95 ITL (ms):                            16.52     
P99 ITL (ms):                            67.86     
Max ITL (ms):                            460.00    
==================================================
```

export SGLANG_MOE_FUSE_DOWN_SUM_REDUCE=0

```bash
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 64        
Successful requests:                     100       
Benchmark duration (s):                  24.05     
Total input tokens:                      34684     
Total input text tokens:                 34684     
Total input vision tokens:               0         
Total generated tokens:                  18999     
Total generated tokens (retokenized):    18983     
Request throughput (req/s):              4.16      
Input token throughput (tok/s):          1442.10   
Output token throughput (tok/s):         789.95    
Peak output token throughput (tok/s):    1482.00   
Peak concurrent requests:                72        
Total token throughput (tok/s):          2232.05   
Concurrency:                             34.15     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   8212.33   
Median E2E Latency (ms):                 8552.62   
---------------Time to First Token----------------
Mean TTFT (ms):                          5115.62   
Median TTFT (ms):                        5986.42   
P99 TTFT (ms):                           9528.11   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          18.41     
Median TPOT (ms):                        16.59     
P99 TPOT (ms):                           66.58     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           16.39     
Median ITL (ms):                         14.33     
P95 ITL (ms):                            17.45     
P99 ITL (ms):                            68.08     
Max ITL (ms):                            628.98    
==================================================
```

SGLANG_MOE_FUSE_DOWN_SUM_REDUCE

SGLANG_MOE_DEBUG_TOPK

```bash
在线 20 rps（更可重复）
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --dataset-name sharegpt --dataset-path /root/.cache/huggingface/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 100 --request-rate 20 --max-concurrency 64 --seed 0 --warmup-requests 50

饱和吞吐（最稳定）
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --dataset-name sharegpt --dataset-path /root/.cache/huggingface/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 100 --request-rate inf --max-concurrency 64 --seed 0 --warmup-requests 50
```



#### *不是MOE模型，验证不了。。。*

```bash
python - <<'PY'
import json
from pathlib import Path
model_path = Path("/root/.cache/huggingface/DeepSeek-R1-Distill-Qwen-1.5B")
# model_path = Path("/home/jiew/jiewc/model-download/DeepSeek-R1-Distill-Qwen-1.5B")
cfg = json.loads((model_path / "config.json").read_text())

print("architectures:", cfg.get("architectures"))
for k in ["num_experts_per_tok","router_top_k","moe_top_k","num_local_experts","num_experts"]:
    if k in cfg:
        print(k, "=", cfg[k])
PY
```



### 云端源码运行

```bash
export SGLANG_MOE_FUSE_DOWN_SUM_REDUCE=1 
```

```bash
bash scripts/launch_server_from_source.sh --model-path /model/HuggingFace/openai/gpt-oss-20b --host 0.0.0.0 --port 30000
```

```bash
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --dataset-name sharegpt --dataset-path /root/model-download/hfd/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 100 --request-rate 20 --max-concurrency 64 --seed 0 --warmup-requests 50
```



**融合前**

```bash
#Input tokens: 33694
#Output tokens: 19062
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    20.0      
Max request concurrency:                 64        
Successful requests:                     100       
Benchmark duration (s):                  67.53     
Total input tokens:                      33694     
Total input text tokens:                 33694     
Total input vision tokens:               0         
Total generated tokens:                  19062     
Total generated tokens (retokenized):    18900     
Request throughput (req/s):              1.48      
Input token throughput (tok/s):          498.98    
Output token throughput (tok/s):         282.29    
Peak output token throughput (tok/s):    1205.00   
Peak concurrent requests:                71        
Total token throughput (tok/s):          781.27    
Concurrency:                             49.65     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   33525.41  
Median E2E Latency (ms):                 33275.16  
---------------Time to First Token----------------
Mean TTFT (ms):                          6465.07   
Median TTFT (ms):                        7866.46   
P99 TTFT (ms):                           14051.11  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          532.97    
Median TPOT (ms):                        200.21    
P99 TPOT (ms):                           5450.87   
---------------Inter-Token Latency----------------
Mean ITL (ms):                           142.71    
Median ITL (ms):                         51.82     
P95 ITL (ms):                            677.33    
P99 ITL (ms):                            2200.30   
Max ITL (ms):                            15796.54  
==================================================
```

```bash
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    20.0      
Max request concurrency:                 64        
Successful requests:                     100       
Benchmark duration (s):                  31.04     
Total input tokens:                      33694     
Total input text tokens:                 33694     
Total input vision tokens:               0         
Total generated tokens:                  19062     
Total generated tokens (retokenized):    18877     
Request throughput (req/s):              3.22      
Input token throughput (tok/s):          1085.45   
Output token throughput (tok/s):         614.08    
Peak output token throughput (tok/s):    1120.00   
Peak concurrent requests:                73        
Total token throughput (tok/s):          1699.54   
Concurrency:                             32.93     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   10222.04  
Median E2E Latency (ms):                 9397.25   
---------------Time to First Token----------------
Mean TTFT (ms):                          138.11    
Median TTFT (ms):                        144.88    
P99 TTFT (ms):                           165.29    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          72.00     
Median TPOT (ms):                        63.04     
P99 TPOT (ms):                           215.57    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           53.18     
Median ITL (ms):                         52.48     
P95 ITL (ms):                            106.03    
P99 ITL (ms):                            220.76    
Max ITL (ms):                            559.62    
==================================================
```

```bash
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    20.0      
Max request concurrency:                 64        
Successful requests:                     100       
Benchmark duration (s):                  30.86     
Total input tokens:                      33694     
Total input text tokens:                 33694     
Total input vision tokens:               0         
Total generated tokens:                  19062     
Total generated tokens (retokenized):    18506     
Request throughput (req/s):              3.24      
Input token throughput (tok/s):          1091.95   
Output token throughput (tok/s):         617.76    
Peak output token throughput (tok/s):    1122.00   
Peak concurrent requests:                73        
Total token throughput (tok/s):          1709.71   
Concurrency:                             32.57     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   10051.19  
Median E2E Latency (ms):                 9234.31   
---------------Time to First Token----------------
Mean TTFT (ms):                          138.15    
Median TTFT (ms):                        145.13    
P99 TTFT (ms):                           177.52    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          67.99     
Median TPOT (ms):                        62.97     
P99 TPOT (ms):                           150.62    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           52.28     
Median ITL (ms):                         51.67     
P95 ITL (ms):                            104.85    
P99 ITL (ms):                            216.22    
Max ITL (ms):                            486.56    
==================================================
```

**融合后**

```bash
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    20.0      
Max request concurrency:                 64        
Successful requests:                     100       
Benchmark duration (s):                  43.71     
Total input tokens:                      33694     
Total input text tokens:                 33694     
Total input vision tokens:               0         
Total generated tokens:                  19062     
Total generated tokens (retokenized):    18894     
Request throughput (req/s):              2.29      
Input token throughput (tok/s):          770.93    
Output token throughput (tok/s):         436.15    
Peak output token throughput (tok/s):    1118.00   
Peak concurrent requests:                72        
Total token throughput (tok/s):          1207.08   
Concurrency:                             42.11     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   18404.39  
Median E2E Latency (ms):                 17447.91  
---------------Time to First Token----------------
Mean TTFT (ms):                          2831.31   
Median TTFT (ms):                        3059.50   
P99 TTFT (ms):                           8025.14   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          338.55    
Median TPOT (ms):                        100.31    
P99 TPOT (ms):                           4463.43   
---------------Inter-Token Latency----------------
Mean ITL (ms):                           82.13     
Median ITL (ms):                         51.92     
P95 ITL (ms):                            106.34    
P99 ITL (ms):                            889.07    
Max ITL (ms):                            11106.61  
==================================================
```

```bash
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    20.0      
Max request concurrency:                 64        
Successful requests:                     100       
Benchmark duration (s):                  30.75     
Total input tokens:                      33694     
Total input text tokens:                 33694     
Total input vision tokens:               0         
Total generated tokens:                  19062     
Total generated tokens (retokenized):    18880     
Request throughput (req/s):              3.25      
Input token throughput (tok/s):          1095.67   
Output token throughput (tok/s):         619.86    
Peak output token throughput (tok/s):    1180.00   
Peak concurrent requests:                73        
Total token throughput (tok/s):          1715.54   
Concurrency:                             32.44     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   9976.52   
Median E2E Latency (ms):                 9153.00   
---------------Time to First Token----------------
Mean TTFT (ms):                          134.76    
Median TTFT (ms):                        142.04    
P99 TTFT (ms):                           167.21    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          66.31     
Median TPOT (ms):                        62.17     
P99 TPOT (ms):                           148.48    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           51.90     
Median ITL (ms):                         51.37     
P95 ITL (ms):                            103.15    
P99 ITL (ms):                            179.46    
Max ITL (ms):                            476.47    
==================================================
```

```bash
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    20.0      
Max request concurrency:                 64        
Successful requests:                     100       
Benchmark duration (s):                  30.95     
Total input tokens:                      33694     
Total input text tokens:                 33694     
Total input vision tokens:               0         
Total generated tokens:                  19062     
Total generated tokens (retokenized):    18884     
Request throughput (req/s):              3.23      
Input token throughput (tok/s):          1088.63   
Output token throughput (tok/s):         615.88    
Peak output token throughput (tok/s):    1113.00   
Peak concurrent requests:                74        
Total token throughput (tok/s):          1704.50   
Concurrency:                             32.57     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   10080.83  
Median E2E Latency (ms):                 9218.10   
---------------Time to First Token----------------
Mean TTFT (ms):                          136.15    
Median TTFT (ms):                        142.91    
P99 TTFT (ms):                           164.91    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          69.19     
Median TPOT (ms):                        61.85     
P99 TPOT (ms):                           196.63    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           52.45     
Median ITL (ms):                         51.69     
P95 ITL (ms):                            105.37    
P99 ITL (ms):                            217.63    
Max ITL (ms):                            491.54    
==================================================
```

**分析**

1. 第一次 req/s=1.48 且 P99 TTFT=14s、Max ITL=15.8s：明显像“测量期间发生了几次秒级停顿”（编译/调优/初始化/偶发资源争用），导致队列积压，TTFT 被拉爆。
2. 第二次 req/s=3.22 且 TTFT/ITL 都很稳定：说明服务已经进入稳定态（shape/kernels/缓存都热了），吞吐接近实际长期能力。

**验证改动是否生效**

SGLANG_MOE_DEBUG_FUSE

```bash
export SGLANG_MOE_DEBUG_FUSE=1
# 强制改用 fused_moe_triton（若不兼容可能报错或退回）
bash scripts/launch_server_from_source.sh --model-path /model/HuggingFace/openai/gpt-oss-20b --host 0.0.0.0 --port 30000

bash scripts/launch_server_from_source.sh --moe-runner-backend triton --model-path /model/HuggingFace/openai/gpt-oss-20b --host 0.0.0.0 --port 30000
```

**更换--moe-runner-backend triton路径启动**

```bash
export SGLANG_MOE_DEBUG_FUSE=1
export SGLANG_MOE_FUSE_DOWN_SUM_REDUCE=1
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6,allocator:cuda_basic
bash scripts/launch_server_from_source.sh \
  --moe-runner-backend triton \
  --model-path /model/HuggingFace/openai/gpt-oss-20b \
  --cpu-offload-gb 120 \
  --offload-mode cpu \
  --mem-fraction-static 0.7 \
  --cuda-graph-max-bs 16 \
  --host 0.0.0.0 --port 30000
```

fp16

```
export SGLANG_MOE_DEBUG_FUSE=1
export SGLANG_MOE_FUSE_DOWN_SUM_REDUCE=1
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6,allocator:cuda_basic
bash scripts/launch_server_from_source.sh \
  --moe-runner-backend triton \
  --model-path /model/HuggingFace/openai/gpt-oss-20b \
  --dtype float16 \
  --cpu-offload-gb 120 \
  --offload-mode cpu \
  --mem-fraction-static 0.7 \
  --cuda-graph-max-bs 16 \
  --host 0.0.0.0 --port 30000
```

**Trinity-Nano**

```
bash scripts/launch_server_from_source.sh --model-path /root/model-download/hfd/Trinity-Nano-Base-Pre-Anneal --trust-remote-code --host 0.0.0.0 --port 30000
```

```
python3 -m sglang.launch_server --model-path /root/model-download/hfd/Trinity-Nano-Base-Pre-Anneal --trust-remote-code --reasoning-parser deepseek_r1 --host 0.0.0.0 --port 30000
```

```
python3 -m sglang.launch_server --model-path /root/model-download/hfd/Trinity-Nano-Base --trust-remote-code --reasoning-parser deepseek-r1 --mem-fraction-static 0.7 --cuda-graph-max-bs 16 --disable-cuda-graph --host 0.0.0.0 --port 30000
```

*这破模型模型文件自己写的，不走sglang本身的函数，也不用sglang的moe算子*

**重新梳理现状**

sglang能不能用--moe-runner-backend triton来打开gpt-oss-20b，目前输入没有输出的问题，是来自于算子修改，还是模型本身

```
python3 -m sglang.launch_server \
    --moe-runner-backend triton \
    --model-path /model/HuggingFace/openai/gpt-oss-20b \
    --host 0.0.0.0 \
    --port 30000
python3 -m sglang.launch_server \
  --moe-runner-backend triton \
  --model-path /model/HuggingFace/openai/gpt-oss-20b \
  --cpu-offload-gb 120 \
  --offload-mode cpu \
  --mem-fraction-static 0.7 \
  --cuda-graph-max-bs 16 \
  --host 0.0.0.0 --port 30000
```

错误位置：

- 错误发生在 `sglang/srt/layers/quantization/mxfp4.py` 文件的第 581 行。
- 函数 `upcast_from_mxfp()` 被调用时传入了 `dtype` 参数，但该函数的定义中并未声明接受此参数

源码启动，改了这个参数的问题，不启动融合，有没有问题？

```
export SGLANG_MOE_DEBUG_FUSE=0
export SGLANG_MOE_FUSE_DOWN_SUM_REDUCE=0
bash scripts/launch_server_from_source.sh \
  --moe-runner-backend triton \
  --model-path /model/HuggingFace/openai/gpt-oss-20b \
  --dtype float16 \
  --cpu-offload-gb 120 \
  --offload-mode cpu \
  --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 8 \
  --host 0.0.0.0 --port 30000
```

```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/launch_server_from_source.sh \
  --moe-runner-backend triton \
  --model-path /model/HuggingFace/openai/gpt-oss-20b \
  --max-total-tokens 1024 \
  --mem-fraction-static 0.7 \
  --cuda-graph-max-bs 1 \
  --disable-cuda-graph \
  --host 0.0.0.0 \
  --port 30000
bash scripts/launch_server_from_source.sh \
  --moe-runner-backend triton \
  --model-path /model/HuggingFace/openai/gpt-oss-20b \
  --max-total-tokens 256 \
  --cuda-graph-max-bs 1 \
  --host 0.0.0.0 \
  --port 30000
```

**/root/model-download/hfd/OLMoE-1B-7B-0125**

```
export SGLANG_MOE_DEBUG_FUSE=1
export SGLANG_MOE_FUSE_DOWN_SUM_REDUCE=1
bash scripts/launch_server_from_source.sh \
    --model-path /root/model-download/hfd/OLMoE-1B-7B-0125 \
    --moe-runner-backend triton \
    --host 0.0.0.0 \
    --port 30000
```

```
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --dataset-name sharegpt --dataset-path /root/model-download/hfd/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 100 --request-rate 20 --max-concurrency 64 --seed 0 --warmup-requests 50
```

*好像可以了，下次要选大容量显卡*

```
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    20.0      
Max request concurrency:                 64        
Successful requests:                     100       
Benchmark duration (s):                  17.61     
Total input tokens:                      37136     
Total input text tokens:                 37136     
Total input vision tokens:               0         
Total generated tokens:                  21239     
Total generated tokens (retokenized):    21228     
Request throughput (req/s):              5.68      
Input token throughput (tok/s):          2108.43   
Output token throughput (tok/s):         1205.86   
Peak output token throughput (tok/s):    2559.00   
Peak concurrent requests:                75        
Total token throughput (tok/s):          3314.29   
Concurrency:                             28.60     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   5037.59   
Median E2E Latency (ms):                 4601.20   
---------------Time to First Token----------------
Mean TTFT (ms):                          112.42    
Median TTFT (ms):                        91.17     
P99 TTFT (ms):                           416.38    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          30.32     
Median TPOT (ms):                        28.12     
P99 TPOT (ms):                           52.00     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           23.30     
Median ITL (ms):                         20.56     
P95 ITL (ms):                            41.55     
P99 ITL (ms):                            152.51    
Max ITL (ms):                            483.93    
==================================================
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    20.0      
Max request concurrency:                 64        
Successful requests:                     100       
Benchmark duration (s):                  17.06     
Total input tokens:                      37136     
Total input text tokens:                 37136     
Total input vision tokens:               0         
Total generated tokens:                  21239     
Total generated tokens (retokenized):    21229     
Request throughput (req/s):              5.86      
Input token throughput (tok/s):          2176.25   
Output token throughput (tok/s):         1244.65   
Peak output token throughput (tok/s):    2682.00   
Peak concurrent requests:                72        
Total token throughput (tok/s):          3420.90   
Concurrency:                             26.72     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   4559.49   
Median E2E Latency (ms):                 4009.48   
---------------Time to First Token----------------
Mean TTFT (ms):                          58.05     
Median TTFT (ms):                        58.07     
P99 TTFT (ms):                           84.94     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          25.13     
Median TPOT (ms):                        24.37     
P99 TPOT (ms):                           40.54     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           21.30     
Median ITL (ms):                         20.16     
P95 ITL (ms):                            42.69     
P99 ITL (ms):                            68.27     
Max ITL (ms):                            118.26    
==================================================
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    20.0      
Max request concurrency:                 64        
Successful requests:                     100       
Benchmark duration (s):                  17.78     
Total input tokens:                      37136     
Total input text tokens:                 37136     
Total input vision tokens:               0         
Total generated tokens:                  21239     
Total generated tokens (retokenized):    21229     
Request throughput (req/s):              5.62      
Input token throughput (tok/s):          2088.54   
Output token throughput (tok/s):         1194.49   
Peak output token throughput (tok/s):    2524.00   
Peak concurrent requests:                73        
Total token throughput (tok/s):          3283.03   
Concurrency:                             27.34     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   4860.87   
Median E2E Latency (ms):                 4199.54   
---------------Time to First Token----------------
Mean TTFT (ms):                          60.21     
Median TTFT (ms):                        58.53     
P99 TTFT (ms):                           87.60     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          26.72     
Median TPOT (ms):                        26.21     
P99 TPOT (ms):                           40.74     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           22.71     
Median ITL (ms):                         22.68     
P95 ITL (ms):                            43.49     
P99 ITL (ms):                            66.56     
Max ITL (ms):                            117.37    
==================================================
```

```
export SGLANG_MOE_DEBUG_FUSE=1
export SGLANG_MOE_FUSE_DOWN_SUM_REDUCE=0
bash scripts/launch_server_from_source.sh \
    --model-path /root/model-download/hfd/OLMoE-1B-7B-0125 \
    --moe-runner-backend triton \
    --host 0.0.0.0 \
    --port 30000
```

```
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    20.0      
Max request concurrency:                 64        
Successful requests:                     100       
Benchmark duration (s):                  18.03     
Total input tokens:                      37136     
Total input text tokens:                 37136     
Total input vision tokens:               0         
Total generated tokens:                  21239     
Total generated tokens (retokenized):    21226     
Request throughput (req/s):              5.55      
Input token throughput (tok/s):          2059.56   
Output token throughput (tok/s):         1177.92   
Peak output token throughput (tok/s):    2631.00   
Peak concurrent requests:                75        
Total token throughput (tok/s):          3237.48   
Concurrency:                             28.74     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   5182.70   
Median E2E Latency (ms):                 4679.29   
---------------Time to First Token----------------
Mean TTFT (ms):                          98.44     
Median TTFT (ms):                        86.20     
P99 TTFT (ms):                           230.88    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          30.84     
Median TPOT (ms):                        28.58     
P99 TPOT (ms):                           54.44     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           24.06     
Median ITL (ms):                         21.02     
P95 ITL (ms):                            41.25     
P99 ITL (ms):                            159.67    
Max ITL (ms):                            305.00    
==================================================
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    20.0      
Max request concurrency:                 64        
Successful requests:                     100       
Benchmark duration (s):                  18.20     
Total input tokens:                      37136     
Total input text tokens:                 37136     
Total input vision tokens:               0         
Total generated tokens:                  21239     
Total generated tokens (retokenized):    21226     
Request throughput (req/s):              5.50      
Input token throughput (tok/s):          2040.79   
Output token throughput (tok/s):         1167.18   
Peak output token throughput (tok/s):    2527.00   
Peak concurrent requests:                72        
Total token throughput (tok/s):          3207.97   
Concurrency:                             27.95     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   5085.74   
Median E2E Latency (ms):                 4361.38   
---------------Time to First Token----------------
Mean TTFT (ms):                          68.61     
Median TTFT (ms):                        69.27     
P99 TTFT (ms):                           103.84    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          28.98     
Median TPOT (ms):                        27.88     
P99 TPOT (ms):                           51.22     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           23.74     
Median ITL (ms):                         23.08     
P95 ITL (ms):                            46.62     
P99 ITL (ms):                            84.40     
Max ITL (ms):                            150.75    
==================================================
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    20.0      
Max request concurrency:                 64        
Successful requests:                     100       
Benchmark duration (s):                  17.42     
Total input tokens:                      37136     
Total input text tokens:                 37136     
Total input vision tokens:               0         
Total generated tokens:                  21239     
Total generated tokens (retokenized):    21227     
Request throughput (req/s):              5.74      
Input token throughput (tok/s):          2132.33   
Output token throughput (tok/s):         1219.53   
Peak output token throughput (tok/s):    2666.00   
Peak concurrent requests:                72        
Total token throughput (tok/s):          3351.87   
Concurrency:                             27.24     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   4744.28   
Median E2E Latency (ms):                 4106.28   
---------------Time to First Token----------------
Mean TTFT (ms):                          62.61     
Median TTFT (ms):                        63.47     
P99 TTFT (ms):                           97.29     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          27.01     
Median TPOT (ms):                        25.38     
P99 TPOT (ms):                           43.80     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           22.15     
Median ITL (ms):                         20.77     
P95 ITL (ms):                            43.60     
P99 ITL (ms):                            88.92     
Max ITL (ms):                            151.78    
==================================================
```

**加大prompt数量**

```
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --dataset-name sharegpt --dataset-path /root/model-download/hfd/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 --request-rate 20 --max-concurrency 64 --seed 0 --warmup-requests 50
```

```
export SGLANG_MOE_FUSE_DOWN_SUM_REDUCE=0
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    20.0      
Max request concurrency:                 64        
Successful requests:                     1000      
Benchmark duration (s):                  113.97    
Total input tokens:                      314096    
Total input text tokens:                 314096    
Total input vision tokens:               0         
Total generated tokens:                  217170    
Total generated tokens (retokenized):    215033    
Request throughput (req/s):              8.77      
Input token throughput (tok/s):          2755.88   
Output token throughput (tok/s):         1905.45   
Peak output token throughput (tok/s):    2555.00   
Peak concurrent requests:                81        
Total token throughput (tok/s):          4661.32   
Concurrency:                             58.85     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   6707.45   
Median E2E Latency (ms):                 4916.21   
---------------Time to First Token----------------
Mean TTFT (ms):                          83.76     
Median TTFT (ms):                        72.71     
P99 TTFT (ms):                           208.35    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          31.41     
Median TPOT (ms):                        31.01     
P99 TPOT (ms):                           49.88     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           30.93     
Median ITL (ms):                         23.21     
P95 ITL (ms):                            69.42     
P99 ITL (ms):                            131.23    
Max ITL (ms):                            470.70    
==================================================
```

```
export SGLANG_MOE_FUSE_DOWN_SUM_REDUCE=1
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    20.0      
Max request concurrency:                 64        
Successful requests:                     1000      
Benchmark duration (s):                  113.32    
Total input tokens:                      314096    
Total input text tokens:                 314096    
Total input vision tokens:               0         
Total generated tokens:                  217170    
Total generated tokens (retokenized):    215031    
Request throughput (req/s):              8.82      
Input token throughput (tok/s):          2771.67   
Output token throughput (tok/s):         1916.37   
Peak output token throughput (tok/s):    2907.00   
Peak concurrent requests:                82        
Total token throughput (tok/s):          4688.04   
Concurrency:                             58.84     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   6668.29   
Median E2E Latency (ms):                 4906.31   
---------------Time to First Token----------------
Mean TTFT (ms):                          84.01     
Median TTFT (ms):                        72.41     
P99 TTFT (ms):                           232.29    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          31.34     
Median TPOT (ms):                        30.84     
P99 TPOT (ms):                           49.00     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           30.75     
Median ITL (ms):                         23.23     
P95 ITL (ms):                            68.99     
P99 ITL (ms):                            128.89    
Max ITL (ms):                            473.63    
==================================================
```

```
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --dataset-name sharegpt --dataset-path /root/model-download/hfd/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 --request-rate 100 --max-concurrency 256 --seed 0 --warmup-requests 50
```

```
bash scripts/launch_server_from_source.sh   --model-path /root/model-download/hfd/OLMoE-1B-7B-0125   --moe-runner-backend triton   --cuda-graph-bs 64   --cuda-graph-max-bs 64   --host 0.0.0.0 --port 30000
```

