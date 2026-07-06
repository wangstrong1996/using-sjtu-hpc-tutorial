# using-sjtu-hpc-tutorial
上海交大HPC使用教程

Created by Claude, under supervision by Shizhuang Wang 

# 从笔记本到超算:PyTorch 深度学习环境配置实战教程

> 以 MNIST 手写数字识别为例,完整走通「Windows 笔记本(NVIDIA GPU)本地训练 → 上海交大思源一号超算 A100 训练」的全流程。
> 本教程基于真实配置过程整理,包含所有踩过的坑及解决方案。

## 目录

- [第一部分:本地环境配置(Windows + NVIDIA GPU)](#第一部分本地环境配置windows--nvidia-gpu)
- [第二部分:跑通第一个神经网络](#第二部分跑通第一个神经网络)
- [第三部分:MNIST 真实训练](#第三部分mnist-真实训练)
- [第四部分:上海交大思源一号超算配置](#第四部分上海交大思源一号超算配置)
- [第五部分:提交 Slurm 作业](#第五部分提交-slurm-作业)
- [常见坑与排查速查表](#常见坑与排查速查表)
- [性能对比参考](#性能对比参考)

---

## 第一部分:本地环境配置(Windows + NVIDIA GPU)

### 1.1 检查显卡驱动

打开 PowerShell,执行:

```powershell
nvidia-smi
```

关注右上角的 **CUDA Version**(例如 `12.3`)——这是驱动支持的最高 CUDA 版本,决定了后面能装哪个构建的 PyTorch。如果命令不存在,先去 NVIDIA 官网安装最新驱动。

> **核心原则:PyTorch 的 CUDA 构建版本必须 ≤ 驱动支持的 CUDA 版本。**
> 例如驱动显示 12.3,则 cu121、cu118 可用,cu130 不可用。

### 1.2 安装 Miniconda

1. 从官网下载 Windows 64-bit 安装包:https://docs.conda.io/en/latest/miniconda.html
2. 一路默认安装
3. 从开始菜单打开 **Anaconda Prompt**(注意:不是普通 PowerShell,普通终端默认找不到 conda 命令)

### 1.3 创建环境并安装 PyTorch

```bash
conda create -n torch python=3.10 -y
conda activate torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**坑 1:conda ToS 错误。** 新版 conda 首次使用官方源会报 `CondaToSNonInteractiveError`,按提示接受服务条款即可:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

然后重新执行 `conda create`。

**网络慢?** 换清华源:`pip install torch torchvision --index-url https://pypi.tuna.tsinghua.edu.cn/simple`(注意清华源只提供最新版本,需要指定旧版本时用 PyTorch 官方源)。

### 1.4 验证安装

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

期望输出类似:`2.5.1+cu121 True`。看到 `True` 说明 GPU 可用。

### 1.5 配置 VS Code(可选但推荐)

1. 从 https://code.visualstudio.com 下载安装
2. 安装扩展:**Python**(Microsoft 出品,会自动附带 Pylance 和 Debugger)
3. 打开项目文件夹后,按 `Ctrl+Shift+P` → 输入 `Python: Select Interpreter` → 选择带 `('torch')` 的解释器;列表里没有就选 "Enter interpreter path...",手动填写,例如:
   ```
   D:\Software\Miniconda3\envs\torch\python.exe
   ```

**坑 2:VS Code 弹窗 "No Python found"。** 点 Cancel 即可,不要让它另装一套 Python——选好 conda 解释器后就不会再弹。

**坑 3:AI 编程助手找不到 torch。** 如果使用 Claude Code 等 AI 助手,它启动的 shell 不会自动激活 conda 环境。解决办法:在项目根目录创建 `CLAUDE.md`(或对应助手的配置文件),写明必须使用完整解释器路径运行 Python:

```markdown
## 运行 Python 的规则
本项目使用 conda 的 torch 环境,运行 Python 时必须使用完整路径:
D:/Software/Miniconda3/envs/torch/python.exe
```

---

## 第二部分:跑通第一个神经网络

创建 `test_gpu.py`:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn(64, 100).to(device)
y = torch.randint(0, 10, (64,)).to(device)
model = torch.nn.Sequential(
    torch.nn.Linear(100, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10)
).to(device)
opt = torch.optim.Adam(model.parameters())

for i in range(100):
    loss = torch.nn.functional.cross_entropy(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    if i % 20 == 0:
        print(f"step {i}, loss {loss.item():.4f}")

print("final loss:", loss.item())
```

运行后 loss 应从约 2.30 降到接近 0。

> **小知识:** 初始 loss ≈ 2.30 = ln(10),即 10 分类随机猜测时的交叉熵。这是判断"模型是否开始学习"的常用参照:训练开始后 loss 明显低于 ln(类别数),说明在学东西。

---

## 第三部分:MNIST 真实训练

创建 `mnist.py`:

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 数据:6 万张 28x28 手写数字,首次运行自动下载(约 60MB)
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=tf)
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
test_dl = DataLoader(test_ds, batch_size=1000)

# 一个小卷积网络
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
    nn.Linear(128, 10),
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    model.train()
    t0 = time.time()
    total_loss = 0.0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * x.size(0)
    print(f"Epoch {epoch+1} training loss: {total_loss/len(train_ds):.4f}")

    # 测试集准确率
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    acc = correct / len(test_ds)
    print(f"Epoch {epoch+1} test accuracy: {acc:.2%} | time: {time.time()-t0:.2f}s")

torch.save(model.state_dict(), "mnist_cnn.pt")
print("Model saved to mnist_cnn.pt")
```

预期结果:3 个 epoch 后测试准确率约 **98%~99%**。RTX 4060 Laptop 上每 epoch 约 15~17 秒。

这份代码包含了真实训练的核心要素:DataLoader 分批加载、train/eval 模式切换、测试集验证、模型保存。**同一份代码将原封不动地在超算上运行。**

---

## 第四部分:上海交大思源一号超算配置

> 前置条件:已通过"交我办"申请到交我算账号。以下用 `YOUR_USERNAME` 代指你的账号名,`acct-XXXXX` 代指课题组账号。

### 4.1 认识集群架构

```
你的笔记本
   │ ssh
   ▼
登录节点 (sylogin.hpc.sjtu.edu.cn)   ← 装环境、编辑文件、提交作业
   │ Slurm 调度
   ▼
计算节点 (a100 / debuga100 队列)     ← 作业实际运行的地方,有 GPU

两者共享同一个 dssg 文件系统(你的家目录)
```

**关键规则(违反有实际后果):**

- 登录节点**禁止运行计算任务**,会被自动查杀并加入黑名单(30~120 分钟无法登录)。装环境、pip install 属于轻量操作,允许。
- 登录节点有 fail2ban:密码连续输错会被封禁 1 小时。
- 传输节点仅用于数据传输,勿在其上编译、提作业。
- 只有作业实际运行(状态 R)才计费,按 卡数 × 时长(卡时)计,登录节点操作和排队不收费。

**思源一号的队列(登录后的欢迎信息中有完整说明):**

| 队列 | 用途 | 限制 |
|---|---|---|
| `a100` | 正式 GPU 计算 | 每节点 4 块 40G A100,每卡最多配 16 CPU 核,单作业最长 7 天 |
| `debuga100` | GPU 调试 | 虚拟 GPU 卡(约 5G 显存),最长 20 分钟,仅单节点 |
| `64c512g` | CPU 计算 | 每节点 64 核 512G |

### 4.2 登录

思源一号登录节点支持公网直接访问,无需校园 VPN:

```bash
ssh YOUR_USERNAME@sylogin.hpc.sjtu.edu.cn
```

首次连接询问服务器指纹时输入 `yes`;输密码时屏幕无回显是正常的。

### 4.3 配置 conda + PyTorch 环境

集群预装了 Miniconda,通过 module 系统加载,无需自己安装:

```bash
module avail miniconda        # 查看可用版本
module load miniconda3        # 加载默认版本
conda create -n torch python=3.10 -y
source activate torch         # 集群上用 source activate,不是 conda activate
```

**安装 PyTorch 前,先确认计算节点驱动支持的 CUDA 版本**(这是本教程最大的坑,见下文坑 6)。思源一号 A100 节点的驱动支持到 CUDA 12.3(以实际为准),因此应安装 cu121 构建:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__)"   # 期望: 2.5.1+cu121
```

> 登录节点没有 GPU,此处 `torch.cuda.is_available()` 返回 False 是**正常的**,不代表安装失败。只需验证 import 成功和版本号。

**坑 4(重要):不要图省事直接用清华源装最新版。** 清华源只有最新版 PyTorch(例如 cu130 构建),而集群驱动往往较旧,装上后 CUDA 初始化会失败,作业**静默回退到 CPU 运行**——作业能跑完、结果正确,但完全没用上 GPU。症状见坑 6。

**可选:让登录后自动进入环境**(注意 Slurm 脚本中仍需保留加载命令):

```bash
echo "module load miniconda3" >> ~/.bashrc
echo "source activate torch" >> ~/.bashrc
```

### 4.4 上传代码和数据

**坑 5:交大 HPC 有两套相互隔离的文件系统,传输节点选错文件会"失踪"。**

| 文件系统 | 服务的队列 | 登录节点 | 传输节点 |
|---|---|---|---|
| dssg (gpfs) | 思源一号 `64c512g`、`a100` | sylogin | **sydata**.hpc.sjtu.edu.cn |
| lustre | π 2.0 的 small、cpu、dgx2 等 | pilogin | data.hpc.sjtu.edu.cn |

给思源一号传文件必须用 **sydata**;若误用 data 节点,文件会落到 lustre 家目录,而 lustre 在思源登录节点上根本没有挂载,`ls` 什么也看不到。

在**本地**终端(PowerShell)执行:

```powershell
scp D:\learnAI\mnist.py YOUR_USERNAME@sydata.hpc.sjtu.edu.cn:~/
scp -r D:\learnAI\data YOUR_USERNAME@sydata.hpc.sjtu.edu.cn:~/
```

- `scp` 语法:`scp [-r] 源 目标`,远程端写法为 `用户名@主机:路径`;`-r` 用于复制目录
- 上传下载都在本地机器上发起,方向由源/目标的位置决定
- **提前把 MNIST 数据一起传上去**:计算节点通常无法联网,`download=True` 在节点上会下载失败;检测到 `./data` 已有数据则自动跳过下载

传完在超算终端确认:

```bash
ls ~        # 应看到 mnist.py 和 data
```

---

## 第五部分:提交 Slurm 作业

### 5.1 编写作业脚本

在超算上创建 `job.slurm`(可用 heredoc 一键创建,也可用 `nano job.slurm` 编辑):

```bash
cat > job.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=mnist
#SBATCH --partition=debuga100
#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load miniconda3
source activate torch
python mnist.py
EOF
```

关键参数说明:

| 参数 | 含义 |
|---|---|
| `--partition=debuga100` | 提交到调试队列(首次跑强烈建议先用它试错) |
| `--qos=debug` | debug 队列可能要求显式指定 QOS,否则报 `Invalid qos specification`(坑 7) |
| `--gres=gpu:1` | 申请 1 块 GPU,**GPU 作业必写**,否则分不到卡 |
| `--cpus-per-task=8` | 配套 CPU 核数(供 DataLoader 等使用),a100 队列每卡上限 16 |
| `--time=00:15:00` | 时间上限,到点强制终止,防止死循环烧钱 |
| `--output=%j.out` | 程序输出写入 `作业号.out` |

> `#SBATCH` 行本质是"写给 Slurm 看的注释",必须位于脚本开头、任何实际命令之前。

查询自己账号可用的 QOS:

```bash
sacctmgr show assoc user=YOUR_USERNAME format=account,partition,qos%40
```

### 5.2 提交与监控

```bash
sbatch job.slurm              # 提交,返回作业号
squeue -u YOUR_USERNAME       # 查看状态:PD=排队, R=运行中, 消失=结束
tail -f 作业号.out            # 实时追踪输出(Ctrl+C 退出,不影响作业)
```

作业提交后由 Slurm 托管,**关闭终端、断开 ssh 都不影响作业运行**。

### 5.3 检查结果——务必确认 device!

```bash
cat 作业号.out
cat 作业号.err
```

**第一眼先看 `Using device:` 那一行。** "作业跑完了"不等于"跑对了":

**坑 6(本教程最大的坑):PyTorch CUDA 构建版本高于节点驱动 → 静默回退 CPU。**

症状:`.out` 显示 `Using device: cpu`,训练正常完成、准确率正常;`.err` 中出现:

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
(found version 12030). ...
```

`found version 12030` 表示驱动支持 CUDA 12.3,而装的 torch 是 cu130 构建。解决:

```bash
pip uninstall torch torchvision -y
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

重新提交作业,确认 `.out` 首行变为 `Using device: cuda`。

**坑 7:`Invalid qos specification`。** debug 队列需要 `--qos=debug`(前提是账号拥有该 QOS,用上面的 sacctmgr 命令查询);正式 a100 队列用默认 QOS(normal),不写 qos 行即可。

### 5.4 调试通过后,转正式 a100 队列

```bash
cp job.slurm job_a100.slurm
sed -i 's/debuga100/a100/' job_a100.slurm
sed -i '/--qos/d' job_a100.slurm
sed -i 's/cpus-per-task=8/cpus-per-task=16/' job_a100.slurm
sbatch job_a100.slurm
```

正式队列排队时间比 debug 长(`PD (Priority)` 属正常等待)。跑完后从超算下载结果到本地:

```powershell
scp YOUR_USERNAME@sydata.hpc.sjtu.edu.cn:~/mnist_cnn.pt D:\learnAI\
```

---

## 常见坑与排查速查表

| # | 症状 | 原因 | 解决 |
|---|---|---|---|
| 1 | `CondaToSNonInteractiveError` | 新版 conda 需先接受官方源服务条款 | 执行报错提示中的三条 `conda tos accept` |
| 2 | VS Code 弹 "No Python found" | conda 的 Python 不在系统 PATH | 点 Cancel,手动选择 conda 环境解释器 |
| 3 | AI 助手运行报 `No module named torch` | 助手的 shell 未激活 conda 环境 | 项目配置文件中写明完整解释器路径 |
| 4 | 超算作业 CUDA 不可用 | 清华源装了最新 cu130,驱动带不动 | 按节点驱动版本选构建(如 cu121) |
| 5 | scp 显示成功但 `ls ~` 为空 | 传输节点选错,文件落在另一个文件系统 | 思源一号用 `sydata.hpc.sjtu.edu.cn` |
| 6 | 作业跑完但 `Using device: cpu` | 同坑 4,CUDA 初始化失败静默回退 | 查 `.err` 中的驱动警告,降级 torch |
| 7 | `Invalid qos specification` | debug 队列需显式指定 QOS | 加 `#SBATCH --qos=debug` |
| 8 | 登录节点 `cuda.is_available()` 为 False | 登录节点本来就没有 GPU | 正常现象,到计算节点上验证 |
| 9 | 排队状态 `PD (Priority)` 很久 | 正式队列资源紧张 | 正常等待;调试期用 debug 队列 |

## 性能对比参考

同一份 MNIST 代码(CNN,batch_size=256,3 epochs)在不同环境的实测:

| 环境 | 设备 | 每 epoch 用时(稳态) | 测试准确率 |
|---|---|---|---|
| 笔记本 | RTX 4060 Laptop (8G) | ~15 s | ~98.9% |
| 思源一号 debuga100 | A100 虚拟切片 (5G) | ~5.4 s | ~98.8% |
| 思源一号 a100 | A100 40G 整卡 | 可自行实测 | ~98.9% |

两点观察:

1. **准确率跨硬件一致**——代码可复现,换硬件不换结果。
2. **MNIST 太小,喂不饱 A100**:任务瓶颈在数据加载和 Python 循环,而非 GPU 算力,因此加速比远低于硬件纸面差距。A100 的优势要在大模型、大 batch、大分辨率任务上才能体现——**选算力应匹配任务规模**。

## 下一步建议

- 换 CIFAR-10 + ResNet,加大 batch size,观察 A100 与本地 GPU 的差距被拉开
- 学习 `nano`/`vim` 基础编辑、`tmux` 会话保持
- 配置 SSH 密钥免密登录
- 了解多卡训练(`torch.nn.parallel.DistributedDataParallel`)与 Slurm 多卡申请(`--gres=gpu:2` 等)

## 参考资料

- 上海交大超算平台用户手册:https://docs.hpc.sjtu.edu.cn/
- 交我算简明使用手册(PDF):https://docs.hpc.sjtu.edu.cn/_static/hpcbriefmanual.pdf
- PyTorch 官方安装指引:https://pytorch.org/get-started/locally/

---

*本教程整理自一次真实的从零配置过程。欢迎提 Issue 补充你遇到的新坑。*

