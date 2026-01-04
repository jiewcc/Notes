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

### 剪切/移动命令

```bash
# 剪切单个文件
mv 源文件 目标目录/

# 示例：
mv 笔记.txt ~/文档/
# 将"笔记.txt"移动到"文档"文件夹
```

```bash
# 剪切多个文件
mv 文件1 文件2 文件3 目标目录/

# 示例：
mv 图片1.jpg 图片2.png ~/图片/
```

```bash
# 剪切整个目录
mv 源目录 目标目录/

# 示例：
mv 项目文件夹 ~/工作/
```

### 程序卸载

```bash
# 卸载nsight-systems（包含nsys-ui）
sudo apt remove --purge nsight-systems

# 或者使用更彻底的卸载
sudo apt autoremove --purge nsight-systems
```

**命令1：普通删除**

```bash

sudo apt remove nsight-systems
```

**作用**：只卸载程序文件，**保留配置文件**

**比喻**：搬家时只搬走家具，但**留下房间的布置图纸**

**命令2：彻底删除**

```bash

sudo apt remove --purge nsight-systems
```

**作用**：卸载程序文件**并且删除配置文件**

**比喻**：搬家时**家具和布置图纸一起清空**

**实际删除的内容对比**：

```bash

# 假设nsight-systems安装在以下位置：

# 1. 程序文件（两种方式都删除）
/usr/bin/nsys
/usr/bin/nsight-sys
/opt/nvidia/nsight-systems/

# 2. 配置文件（只有--purge会删除）
/etc/nsight-systems/              # 系统配置
~/.config/nsight-systems/         # 用户配置
~/.cache/nsight-systems/          # 缓存文件
~/.local/share/nsight-systems/    # 用户数据
```

**autoremove的作用：**

```bash
sudo apt autoremove --purge nsight-systems
└──┬─── └──┬────── └─┬─── └──────┬──────
   │       │         │           └── 要删除的包
   │       │         └── 彻底删除（含配置）
   │       └── 自动删除不需要的依赖
   └── 管理员权限
```

删除为这个包安装的依赖包，但这些依赖现在没有被其他包使用。
比喻：不仅拆房子，还清理建筑垃圾

### 解压缩包

```bash
sudo dpkg -i nsight-systems-2025.5.1_2025.5.1.121-1_amd64.deb
```

选项：-i (install，安装)

| 操作           | dpkg命令         | apt命令                   | 说明             |
| -------------- | ---------------- | ------------------------- | ---------------- |
| **安装本地包** | `dpkg -i 包.deb` | `apt install ./包.deb`    | 安装本地.deb文件 |
| **卸载包**     | `dpkg -r 包名`   | `apt remove 包名`         | 移除包但保留配置 |
| **彻底卸载**   | `dpkg -P 包名`   | `apt remove --purge 包名` | 完全移除         |
| **查询已安装** | `dpkg -l`        | `apt list --installed`    | 查看已安装包     |
| **查看包信息** | `dpkg -I 包.deb` | -                         | 查看包信息       |

### 调节屏幕亮度

[ubuntu调节屏幕亮度方法](https://blog.csdn.net/snoop_doog/article/details/149164339)

```bash
nvidia-settings
```

打开nvidia设置，然后进入GPU 0 --> DP-0 --> Color Correction就可以调节屏幕亮度了

### 修改环境变量

临时修改

```bash
export DISPLAY=:0
# 例如，设置PATH环境变量
export PATH=$PATH:/your/new/path
```

永久修改环境变量

```bash
# 修改~/.bashrc或~/.bash_profile文件
nano ~/.bashrc  # 或使用你喜欢的文本编辑器

# 在文件的末尾添加你的环境变量设置：
export VAR_NAME="value"
export PATH=$PATH:/your/new/path

# 保存并关闭文件，使更改生效
source ~/.bashrc  # 或重新登录你的会话
```

### 看ubuntu内核版本

可以查看`/proc/version`文件来获取内核版本信息：

```bash
cat /proc/version
```

对于使用systemd的系统（如Ubuntu 16.04及以后版本），可以使用`hostnamectl`命令来查看内核版本：

```bash
hostnamectl
```

### 截图快捷键

shift+print

### 改文件夹名

```bash
mv old_folder new_folder
```

这将把名为old_folder的文件夹重命名为new_folder

```bash
rename 's/old_folder/new_folder/' *
```

这将在当前目录下将所有文件夹名中的old_folder替换为new_folder。
