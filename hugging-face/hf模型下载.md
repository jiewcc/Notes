### 专用多线程下载器 hfd

[如何快速下载huggingface模型——全方法总结（原帖）](https://zhuanlan.zhihu.com/p/663712983)

[Huggingface Model Downloader](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f)

其原理是 Step1：通过Hugging Face API获取模型/数据集仓库对应的所有文件 url；Step2：利用 `aria2` 多线程下载文件。

该工具同样支持设置镜像端点的环境变量:

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

**基本命令：**

```bash
./hfd.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

如果没有安装 aria2，则可以改用 wget：

```bash
./hdf.sh bigscience/bloom-560m --tool wget
```

`--include` 指定下载特定文件

```bash
# Qwen2.5-Coder下载q2_k量化版本的模型
hfd Qwen/Qwen2.5-Coder-32B-Instruct-GGUF --include qwen2.5-coder-32b-instruct-q2_k.gguf
# gpt2下载onnx路径下的所有json文件
hfd gpt2 --include onnx/*.json 
```

**多线程和并行下载：**

hfd 在使用 aria2c 作为下载工具时，支持两种并行配置：

- **单文件线程数** (`-x`)：控制每个文件的连接数，用法：`hfd gpt2 -x 8`，建议值：4-8，默认：4 线程。限制最大为10，别开太多了，服务器压力太大了😂。
- **并发文件数** (`-j`)：控制同时下载的文件数，用法：`hfd gpt2 -j 3`，建议值：3-8，默认：5 个文件。限制最大为10，同上别开太大。

组合使用：

```bash
hfd deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B -x 8 -j 3  # 每个文件 8 个线程，同时下载 3 个文件
```

**需要安装aria2c :**

```bash
sudo apt update
sudo apt install -y aria2
```

首先，下载[`hfd.sh`](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f#file-hfd-sh)或克隆此仓库，授予脚本执行权限：

```
chmod a+x hfd.sh
```

为了方便起见，您可以创建一个别名：

```bash
hfd= " $PWD /hfd.sh "
```

使用说明：

```bash
$ ./hfd.sh --help
Usage:
  hfd <REPO_ID> [--include include_pattern1 include_pattern2 ...] [--exclude exclude_pattern1 exclude_pattern2 ...] [--hf_username username] [--hf_token token] [--tool aria2c|wget] [-x threads] [-j jobs] [--dataset] [--local-dir path] [--revision rev]

Description:
  Downloads a model or dataset from Hugging Face using the provided repo ID.

Arguments:
  REPO_ID         The Hugging Face repo ID (Required)
                  Format: 'org_name/repo_name' or legacy format (e.g., gpt2)
Options:
  include/exclude_pattern The patterns to match against file path, supports wildcard characters.
                  e.g., '--exclude *.safetensor *.md', '--include vae/*'.
  --include       (Optional) Patterns to include files for downloading (supports multiple patterns).
  --exclude       (Optional) Patterns to exclude files from downloading (supports multiple patterns).
  --hf_username   (Optional) Hugging Face username for authentication (not email).
  --hf_token      (Optional) Hugging Face token for authentication.
  --tool          (Optional) Download tool to use: aria2c (default) or wget.
  -x              (Optional) Number of download threads for aria2c (default: 4).
  -j              (Optional) Number of concurrent downloads for aria2c (default: 5).
  --dataset       (Optional) Flag to indicate downloading a dataset.
  --local-dir     (Optional) Directory path to store the downloaded data.
                             Defaults to the current directory with a subdirectory named 'repo_name'
                             if REPO_ID is is composed of 'org_name/repo_name'.
  --revision      (Optional) Model/Dataset revision to download (default: main).

Example:
  hfd gpt2
  hfd bigscience/bloom-560m --exclude *.bin *.msgpack onnx/*
  hfd meta-llama/Llama-2-7b --hf_username myuser --hf_token mytoken -x 4
  hfd lavita/medical-qa-shared-task-v1-toy --dataset
  hfd bartowski/Phi-3.5-mini-instruct-exl2 --revision 5_0
```

### hfd下载数据集：

在终端中运行以下命令：

```bash
./hfd.sh wikitext --dataset --tool aria2c -x 4
```

参数说明：

- `wikitext`：要下载的数据集名称，对应替换为你自己想下载的。
- `--dataset`：指定下载数据集。
- `--tool aria2c` 和 `-x 4`：同上，使用 `aria2c` 进行多线程下载。

```bash
./hfd.sh anon8231489123/ShareGPT_Vicuna_unfiltered --dataset --tool aria2c -x 4
```

