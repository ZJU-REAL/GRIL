
# GRIL 项目快速开始指南

这是一个关于如何设置和运行 GRIL 项目的详细指南。

## 1. 环境设置

我们强烈建议使用 Conda 来管理您的环境，以确保所有依赖项的兼容性。

### a. 创建并激活 Conda 环境

打开终端并执行以下命令，从 `environment.yml` 文件创建 Conda 环境。这个文件包含了所有必需的依赖项，包括 Python 版本和所需的包。

```bash
conda env create -f environment.yml
```

环境创建成功后，使用以下命令激活它：

```bash
conda activate gril
```

## 2. 安装 `verl`

由于 `verl` 是一个本地包，我们需要手动进行安装。

在您的终端中，导航到 `verl` 目录并执行以下命令：

```bash
cd verl
pip install -e .
cd ..
```

这会以“可编辑”模式安装 `verl`，使其可以在您的环境中被直接调用。

## 3. 开始训练

一切准备就绪后，您可以通过运行训练脚本来开始训练模型。

### a. 运行训练脚本

在项目根目录下，执行以下 shell 脚本：

```bash
bash ./GRIL/train.sh
```

这个脚本会设置一些环境变量，并使用 `base` 配置来启动训练过程。

## 4. 评估模型

训练完成后，您可以运行评估脚本来验证您的模型。

```bash
python ./GRIL/ragen/eval.py
```
