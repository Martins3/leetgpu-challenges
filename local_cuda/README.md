# LeetGPU 本地 CUDA 快速开始

目标：直接改 `starter/starter.cu`，在本地直接跑题。

## 1. 准备环境

```bash
uv venv .venv --python python3
source .venv/bin/activate
uv pip install "torch==2.5.1+cu124" numpy --index-url https://download.pytorch.org/whl/cu124
python local_cuda/generate_compile_commands.py
```

上面这套在当前机器上已经验证可用。

如果你想让编辑器打开 `.cu` 时少报错，先跑一次 `python local_cuda/generate_compile_commands.py`。

## 2. 选一道题

每道题的关键文件是：

- `challenge.py`：签名、测试、标准答案
- `starter/starter.cu`：CUDA 模板

开始前先看一眼 `challenge.py` 里的 `get_solve_signature()`，你的 `solve(...)` 参数类型和顺序必须和这里一致。

## 3. 直接改 starter

直接打开这题的 starter：

```bash
$EDITOR challenges/easy/21_relu/starter/starter.cu
```

然后实现你自己的 `solve(...)`。

## 4. 运行

跑 example：

```bash
python local_cuda/local_test.py challenges/easy/21_relu
```

跑全部 functional tests：

```bash
python local_cuda/local_test.py challenges/easy/21_relu --functional
```

如果你不用 `local.cu` 这个文件名，也可以显式传 `.cu` 文件路径：

```bash
python local_cuda/local_test.py challenges/easy/21_relu/my_kernel.cu
```

## 5. 推荐工作流

```bash
source .venv/bin/activate
$EDITOR challenges/easy/21_relu/starter/starter.cu
python local_cuda/local_test.py challenges/easy/21_relu
python local_cuda/local_test.py challenges/easy/21_relu --functional
```

## 6. 一个常见坑

`starter/starter.cu` 里的 `solve(...)` 参数类型不一定完全对，最终以 `challenge.py` 里的 `get_solve_signature()` 为准。

## 7. 额外说明

- runner 会自动编译你的 `.cu`
- runner 会自动读取题目的 `challenge.py`
- 直接传 challenge 目录时，runner 默认使用 `starter/starter.cu`
- runner 只做正确性测试，不测线上性能
- 如果自动检测 GPU 架构有问题，可以手动指定：

```bash
LOCAL_CUDA_ARCH=sm_61 python local_cuda/local_test.py challenges/easy/21_relu
```
