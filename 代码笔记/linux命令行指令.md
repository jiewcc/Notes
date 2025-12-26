### **grep 的上下文选项**：

```bash
docker info | grep -A 10 "Registry Mirrors"
```

-A 10是 grep命令的一个选项，意思是 "显示匹配行及之后10行"（After 10 lines）

| 选项   | 全称                 | 含义                        |
| ------ | -------------------- | --------------------------- |
| `-A n` | `--after-context=n`  | 显示匹配行及其**后n行**     |
| `-B n` | `--before-context=n` | 显示匹配行及其**前n行**     |
| `-C n` | `--context=n`        | 显示匹配行及其**前后各n行** |

### 安装多个软件包的命令

```
sudo apt install -y ca-certificates curl gnupg
```

权限：sudo（以管理员身份运行）
操作：install（安装）
选项：-y（自动确认）
软件包1：ca-certificates
软件包2：curl
软件包3：gnupg

### 获取系统版本

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
# $()：执行命令并获取输出
# . /etc/os-release：加载系统信息文件
# echo $ID$VERSION_ID：输出系统ID和版本
# distribution=：保存到变量

echo $distribution
# 输出示例：ubuntu22.04 
```

### 检查上一条命令是否成功
```bash
echo $?
# 返回 0 表示成功，非0表示失败
```

### 在命令行查看是否设置代理

```bash
env | grep -i proxy
```

可能的输出：

```bash
http_proxy=http://127.0.0.1:7890
https_proxy=http://127.0.0.1:7890
all_proxy=socks5://127.0.0.1:7891
```

使用以下命令取消：

```bash
unset http_proxy                                 
unset https_proxy
unset all_proxy
```

还有可能是因为你的 Git 之前配置了代理。查看配置（如果是当前项目配置，去掉 --global）：

```bash
git config --global --list
```

可能的输出：

```bash
http.proxy=http://127.0.0.1:7890
https.proxy=http://127.0.0.1:7890
```

如果存在代理，对应取消：

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```

现在应该可以正常下载。

### 重新设置代理

如果你想重新设置代理，下面也给出对应的命令，假设 HTTP/HTTPS 端口号为 7890， SOCKS5 为 7891。

- 终端代理：

```text
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7891
```

- Git 代理：

```text
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
```

### cp复制目录

```bash
cp 文件1 文件2        # 只能复制单个文件
cp DeepSeek-R1/* ~/.cache/  # 错误！不能复制目录

cp -r 目录 ~/.cache/  # 可以复制整个目录及其所有内容
```

```bash
cp -r ~/jiewc/model-download/DeepSeek-R1-Distill-Qwen-1.5B ~/.cache/huggingface/
```

**效果**：把整个 `DeepSeek-R1-Distill-Qwen-1.5B`文件夹（包含所有子文件夹和文件）复制到目标位置。

### root用户sudo git clone

```bash
sudo git clone git@github.com:mckaywrigley/chatbot-ui.git
```

`sudo`会切换到 root 用户，而 root 用户没有您的 SSH 密钥配置。

### 删除目录

```bash
# 删除空目录
rmdir 目录名

# 删除目录及其所有内容
rm -r 目录名
```

