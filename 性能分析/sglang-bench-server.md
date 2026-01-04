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

