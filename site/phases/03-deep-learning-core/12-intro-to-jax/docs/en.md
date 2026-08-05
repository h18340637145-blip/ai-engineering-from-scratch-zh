# JAX 简介

> PyTorch 修改张量。TensorFlow 构建图。JAX 编译纯函数。最后一个改变了你对深度学习的看法。

**类型:** 构建
**语言:** Python
**先决条件:** 第三阶段课程 01-10，基础 NumPy
**时间:** ~90 分钟

## 学习目标

- 使用 JAX 的函数式 API（jax.numpy, jax.grad, jax.jit, jax.vmap）编写纯函数神经网络代码
- 解释 PyTorch 的即时修改与 JAX 的函数式编译模型之间的关键设计差异
- 应用 jit 编译和 vmap 向量化来加速训练循环，与朴素的 Python 相比
- 在 JAX 中训练一个简单网络，并对比显式状态管理与 PyTorch 的面向对象方法

## 问题

你知道如何在 PyTorch 中构建神经网络。你定义一个 `nn.Module`，调用 `.backward()`，执行优化器。它有效。数百万人使用它。

但 PyTorch 在其 DNA 中有一个限制：它急切地追踪操作，一个接一个地在 Python 中执行。每个 `tensor + tensor` 都是一个独立的内核启动。每个训练步骤都重新解释相同的 Python 代码。这在你需要在 2048 个 TPUs 上训练一个 5400 亿参数模型时会变得非常困难。这时的开销会把你击垮。

Google DeepMind 使用 JAX 训练 Gemini。Anthropic 使用 JAX 训练 Claude。这些都不是小规模的操作——它们是地球上最大的神经网络训练运行。他们选择 JAX 是因为它将你的训练循环视为一个可编译的程序，而不是一系列 Python 调用。

JAX 是带有三个超能力的 NumPy：自动微分、编译为 XLA 的 JIT 编译和自动向量化。你编写一个处理单个示例的函数。JAX 给你一个可以处理一批数据、计算梯度、编译为机器代码，并在多个设备上运行的函数。所有这些都不需要改变原始函数。

## 概念

### JAX 的哲学

JAX 是一个函数式框架。没有类，没有可变状态，没有 `.backward()` 方法。相反：

| PyTorch | JAX |
|---------|-----|
| 带有状态的 `nn.Module` 类 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 即时执行 | 通过 XLA 的 JIT 编译 |
| `for x in batch:` 手动循环 | `jax.vmap(f)` 自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行 |
| 可变的 `model.parameters()` | 不可变的数组 pytree |

这不是一种风格偏好。这是一个编译器限制。JIT 编译需要纯函数——相同的输入总是产生相同的输出，没有副作用。正是这种限制使得 100 倍的速度提升成为可能。

### jax.numpy：熟悉的表面

JAX 在加速器上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

相同的函数名称。相同的广播规则。相同的切片语义。但数组位于 GPU/TPU 上，每个操作都可以被编译器追踪。

一个关键区别：JAX 数组是不可变的。没有 `a[0] = 5`。取而代之的是：`a = a.at[0].set(5)`。这在最初的一周里感觉很别扭，但之后就会明白——不可变性正是使得像 `grad`、`jit` 和 `vmap` 这样的转换可以组合在一起的原因。

### jax.grad：函数式自动微分

PyTorch 将梯度附加到张量（`.grad`）。JAX 将梯度附加到函数。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接受一个函数，并返回一个新函数，该函数用于计算梯度。不需要 `.backward()` 调用。张量上不存储计算图。梯度只是一个你可以调用、组合或 JIT 编译的函数。

这可以任意组合：

 /no_think

<>

`jax.grad` 接受一个函数，并返回一个新函数，该函数用于计算梯度。不需要 `.backward()` 调用。张量上不存储计算图。梯度只是一个你可以调用、组合或 JIT 编译的函数。

这可以任意组合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数。三阶导数。雅可比矩阵。海森矩阵。全部都可以通过组合 `grad` 来实现。PyTorch 也可以做到（`torch.autograd.functional.hessian`），但这是附加的功能。在 JAX 中，这是其基础。

约束条件：`grad` 只能用于纯函数。函数内部不能有打印语句（它们在追踪时运行，而不是执行时）。不能修改外部状态。如果没有显式的密钥管理，不能进行随机数生成。

### jit: 编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

在第一次调用时，JAX 会追踪该函数 —— 它记录哪些操作发生，但不会执行这些操作。然后它将这个追踪信息传递给 XLA（Accelerated Linear Algebra），即 Google 的 TPU 和 GPU 编译器。XLA 会合并操作，消除冗余的内存复制，并生成优化后的机器代码。

后续调用将完全跳过 Python。编译后的代码以 C++ 的速度在加速器上运行。

JIT 有帮助的情况：
- 训练步骤（相同的计算重复数千次）
- 推理（相同的模型，不同的输入）
- 任何被调用超过一次且输入形状相似的函数

JIT 有负面影响的情况：
- 包含依赖于值的 Python 控制流的函数（`if x > 0` 其中 x 是一个被追踪的数组）
- 一次性计算（编译开销超过运行时间）
- 调试（追踪隐藏了实际执行）

控制流的限制是真实的。`jax.lax.cond` 替代了 `if/else`。`jax.lax.scan` 替代了 `for` 循环。这些不是可选的 —— 它们是编译的代价。

### vmap：自动向量化

你编写一个处理单个示例的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 将其提升以处理一批数据：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 的意思是：不要对 `params`（共享）进行批量处理，而是对 `x` 的第 0 轴进行批量处理。不要手动编写 `for` 循环。不要进行重塑。不要使用批量维度线程。JAX 会自动识别批量维度，并对整个计算进行向量化。

这不是语法糖。`vmap` 会生成融合的向量化代码，其运行速度比 Python 循环快 10 到 100 倍。并且它能够与 `jit` 和 `grad` 组合使用：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

每个示例的梯度。一行代码。在不使用技巧的情况下，这在 PyTorch 中几乎是不可能实现的。

### pmap：跨设备的数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 在所有可用设备（GPU/TPU）上复制功能并拆分批次。在函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 在设备之间同步梯度。

Google 使用 `pmap`（及其后续版本 `shard_map`）在数千个 TPU v5e 芯片上训练 Gemini。编程模型：编写单设备版本，用 `pmap` 包裹，完成。

### Pytrees：通用数据结构

JAX 操作“pytrees”——列表、元组、字典和数组的嵌套组合。你的模型参数是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每一个 JAX 变换 -- `grad`, `jit`, `vmap` -- 都知道如何遍历 pytrees。`jax.tree.map(f, tree)` 将 `f` 应用于每一个叶子。这就是优化器一次性更新所有参数的方式：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()` 方法。没有参数注册。树结构即为模型。

### 功能性与面向对象

PyTorch 在对象内部存储状态：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```JAX 使用具有显式状态的纯函数：

```python
def f(x):
    return x + 1
```

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

参数被传入。没有任何内容被存储，也没有任何内容被修改。这使得每个函数都可测试、可组合和可编译。这也意味着你必须自己管理这些参数——或者使用像 Flax 或 Equinox 这样的库。

### JAX 生态系统

JAX 为你提供原语。库则为你提供使用上的便利性：

| 库 | 角色 | 风格 |
|---------|------|-------|
| **Flax** (Google) | 神经网络层 | `nn.Module`，带有显式状态 |
| **Equinox** (Patrick Kidger) | 神经网络层 | 基于 Pytree，Pythonic 风格 |
| **Optax** (DeepMind) | 优化器 + 学习率调度 | 可组合的梯度变换 |
| **Orbax** (Google) | 检查点 | 保存/恢复 pytree |
| **CLU** (Google) | 指标 + 日志 | 训练循环工具 |

Optax 是标准的优化库。它将梯度变换（Adam、SGD、裁剪）与参数更新分开，使得组合变得非常简单：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时使用 JAX 与 PyTorch

| 因素 | JAX | PyTorch |
|-----|-----|---------|
| TPU 支持 | 一流（Google 开发了两者） | 社区维护（torch_xla） |
| GPU 支持 | 良好（通过 XLA 的 CUDA） | 首屈一指（原生 CUDA） |
| 调试 | 困难（追踪 + 编译） | 简单（即时执行，逐行调试） |
| 生态系统 | 研究导向（Flax, Equinox） | 庞大（HuggingFace, torchvision 等） |
| 招聘 | 尼iche（Google/DeepMind/Anthropic） | 主流（处处可见） |
| 大规模训练 | 优越（XLA, pmap, mesh） | 良好（FSDP, DeepSpeed） |
| 原型开发速度 | 较慢（函数式开销） | 更快（修改后直接运行） |
| 生产推理 | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| 使用者 | DeepMind（Gemini），Anthropic（Claude） | Meta（Llama），OpenAI（GPT），Stability AI |

诚实的答案：除非有特定理由使用 JAX，否则应使用 PyTorch。这些特定理由包括：TPU 访问权限、需要每个样本的梯度、大规模多设备训练，或者在 Google/DeepMind/Anthropic 工作。

### JAX 中的随机数

JAX 没有全局的随机状态。每个随机操作都需要一个显式的 PRNG 密钥：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

这在一开始会让人感到烦恼。但它可以保证在不同设备和编译之间的一致性——这是 PyTorch 的 `torch.manual_seed` 在多 GPU 设置中无法保证的属性。

```figure
batchnorm-effect
```

## 构建它

### 第一步：设置和数据

我们将使用 JAX 和 Optax 在 MNIST 上训练一个 3 层的 MLP。784 个输入，两个隐藏层分别有 256 和 128 个神经元，10 个输出类别。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 步骤 2：初始化参数

没有类。只是一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

手动进行 He 初始化。从一个种子中分割出三个 PRNG 密钥。每个权重都是嵌套字典中的一个不可变数组。

### 第三步：前向传播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数。参数输入，预测输出。不依赖 `self`，不存储状态。`loss_fn` 从头开始计算交叉熵——softmax、log、负均值。

### 步骤 4：JIT 编译的训练步骤

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 在一次传递中返回损失值和梯度。`@jax.jit` 装饰器将这两个函数编译为 XLA。第一次调用之后，每个训练步骤都无需接触 Python。

### 第 5 步：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```10 个 epoch。~97% 的测试准确率。第一个 epoch 较慢（JIT 编译）。第 2-10 个 epoch 较快。

注意缺少的内容：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个更新是一个组合函数调用。梯度被计算，通过 Adam 进行变换，然后应用到参数——所有操作都在 `train_step` 中完成。

## 使用它

### Flax：Google 标准

Flax 是最常用的 JAX 神经网络库。它重新引入了 `nn.Module`，但需要显式的状态管理：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

与 PyTorch 的结构相同，但 `params` 与模型是分开的。`model.init()` 创建参数。`model.apply(params, x)` 执行前向传播。模型对象没有状态。

### Equinox：Pythonic 的替代方案

Equinox（由 Patrick Kidger 开发）将模型表示为 pytrees：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

该模型本身是一个 pytree。不需要 `.apply()`。参数仅仅是模型的叶子。这更接近 JAX 的思维方式。

### Optax：可组合的优化器

Optax 将梯度变换与更新过程解耦：

 /no_think

<>

该模型本身是一个 pytree。不需要 `.apply()`。参数仅仅是模型的叶子。这更接近 JAX 的思维方式。

### Optax：可组合的优化器

Optax 将梯度变换与更新过程解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热、权重衰减 —— 所有这些都作为一系列变换组成。每个变换都会看到梯度，修改它们，然后将它们传递给下一个变换。没有单一的优化器类。

## 发布它

**安装：**

```bash
pip install jax jaxlib optax flax
```

对于 GPU 支持：

```bash
pip install jax[cuda12]
```

对于 TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```**性能陷阱：**

- 第一次 JIT 调用较慢（编译）。在进行基准测试前请进行预热。
- 避免在 JIT 中对 JAX 数组使用 Python 循环。使用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 在 JIT 内部有效。常规的 `print()` 无效。
- 使用 `jax.profiler` 或 TensorBoard 进行性能分析。XLA 编译可能会隐藏瓶颈。
- JAX 默认预分配 75% 的 GPU 内存。设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可以禁用此行为。

**检查点：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```**本课将产生以下内容：**
- `outputs/prompt-jax-optimizer.md` -- 用于选择合适 JAX 优化器配置的提示
- `outputs/skill-jax-patterns.md` -- 涵盖 JAX 函数式模式的技能

## 练习

1. 向 MLP 添加 dropout。在 JAX 中，dropout 需要一个 PRNG 密钥 -- 将密钥穿行于前向传播过程中，并为每个 dropout 层拆分密钥。比较有无 dropout 的测试准确率。

2. 使用 `jax.vmap` 为 32 张 MNIST 图像的批次计算每个样本的梯度。计算每个样本的梯度范数。哪些样本的梯度最大，为什么？

3. 用一个通用的 `mlp_forward(params, x)` 替换手动编写的前向函数，该函数适用于任意数量的层。使用 `jax.tree.leaves` 自动确定深度。

4. 使用和不使用 `@jax.jit` 对训练步骤进行基准测试。分别对每种情况运行 100 步，记录时间。在你的硬件上速度提升有多大？第一次调用时的编译开销是多少？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现梯度裁剪。分别使用和不使用裁剪进行训练。绘制训练过程中的梯度范数，以观察效果。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------|
| XLA | “让 JAX 快速的原因” | 加速线性代数 -- 一个编译器，可以融合操作，并从计算图生成优化的 GPU/TPU 内核 |
| JIT | “即时编译” | JAX 在第一次调用时跟踪函数，将其编译为 XLA，然后在后续调用中运行编译后的版本 |
| 纯函数 | “没有副作用” | 输出仅依赖于输入的函数 -- 没有全局状态、没有突变、没有不带显式密钥的随机性 |
| vmap | “自动批处理” | 将处理一个样本的函数转换为处理一个批次的函数，无需重写 |
| pmap | “自动并行化” | 将函数复制到多个设备上，并分割输入批次 |
| Pytree | “嵌套的数组字典” | JAX 可以遍历和转换的任何嵌套结构，包括列表、元组、字典和数组 |
| 跟踪 | “记录计算” | JAX 使用抽象值执行函数，构建计算图，不计算实际结果 |
| 函数式自动微分 | “函数的梯度” | 通过转换函数计算导数，而不是通过将梯度存储附加到张量 |
| Optax | “JAX 的优化器库” | 一个可组合的梯度变换库 -- Adam、SGD、裁剪、调度等，可以串联使用 |
| Flax | “JAX 的 nn.Module” | Google 为 JAX 提供的神经网络库，添加了层抽象，同时保持状态显式 |

## 进一步阅读

- JAX 文档: https://jax.readthedocs.io/ -- 官方文档，包含关于 grad、jit 和 vmap 的优秀教程
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- 解释设计哲学的原始论文
- Flax 文档: https://flax.readthedocs.io/ -- Google 为 JAX 提供的神经网络库
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- Flax 的 Pythonic 替代方案
- DeepMind, "Optax: composable gradient transformation and optimisation" -- 标准的优化器库
- "You Don't Know JAX" (Colin Raffel, 2020) -- 一位 T5 作者撰写的关于 JAX 陷阱和模式的实用指南
