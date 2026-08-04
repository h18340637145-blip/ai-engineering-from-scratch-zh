# ML 管道

> 模型不是一个产品。管道才是。管道是从原始数据到部署预测的全过程，每一步都必须可复现。

**类型:** 构建  
**语言:** Python  
**先决条件:** 第二阶段，第12课（超参数调优）  
**时间:** ~120 分钟

## 学习目标

- 从零开始构建一个 ML 管道，将插补、缩放、编码和模型训练链式连接成一个可复现的对象
- 识别数据泄露场景，并解释管道如何通过仅在训练数据上拟合转换器来防止它们
- 构建一个 ColumnTransformer，对数值和分类特征应用不同的预处理
- 实现管道的序列化，并演示相同的拟合管道在训练和生产中产生相同的结果

## 问题

你有一个笔记本，它加载数据，用中位数填充缺失值，缩放特征，训练模型，并打印准确性。它工作正常。你将其部署。

一个月后，有人重新训练模型并得到不同的结果。中位数是在包括测试数据的整个数据集上计算的（数据泄露）。缩放参数没有保存，因此推理使用了不同的统计信息。特征工程代码在训练和部署之间被复制粘贴，导致副本出现分歧。一个分类列在生产环境中出现了一个编码器从未见过的新值。

这些问题不是假设性的。它们是机器学习系统在生产中失败的最常见原因。管道通过将每个转换步骤包装成一个单一的、有序的、可复现的对象来解决所有这些问题。

## 概念

### 管道是什么

管道是一系列按顺序排列的数据转换，随后是一个模型。每一步都以前一步的输出作为输入。整个管道仅在训练数据上拟合一次。在推理时，相同的拟合管道转换新数据并生成预测。```mermaid
flowchart LR
    A[Raw Data] --> B[Impute Missing Values]
    B --> C[Scale Numeric Features]
    C --> D[Encode Categoricals]
    D --> E[Train Model]
    E --> F[Prediction]
```该流程保证：
- 变换仅在训练数据上拟合（无信息泄露）
- 推理时应用相同的变换
- 整个对象可以序列化并部署为一个工件
- 交叉验证按每个折叠应用流程，防止细微的信息泄露

### 数据泄露：沉默的杀手

当测试集或未来数据的信息污染训练数据时，就会发生数据泄露。流程可以防止最常见的泄露形式。

**泄露（错误）：**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think

<>

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference time
- The entire object can be serialized and deployed as one artifact
- Cross-validation applies the pipeline per fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**Leaky (wrong):**
 /no_think```python
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test = X_scaled[:800], X_scaled[800:]
y_train, y_test = y[:800], y[800:]
```缩放器看到了测试数据。均值和标准差包含测试样本。这会夸大准确性估计。

**正确：**```python
X_train, X_test = X[:800], X[800:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```使用管道，你不需要考虑这些。管道会自动处理这些。

### sklearn 管道

sklearn 的 `Pipeline` 链接转换器和一个估计器。它暴露了 `.fit()`、`.predict()` 和 `.score()`，它们按顺序应用所有步骤。```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```当你调用 `pipe.fit(X_train, y_train)` 时：
1. Scaler 在 X_train 上调用 `fit_transform`
2. Model 在缩放后的 X_train 上调用 `fit`

当你调用 `pipe.predict(X_test)` 时：
1. Scaler 在 X_test 上调用 `transform`（不是 fit_transform） 
2. Model 在缩放后的 X_test 上调用 `predict`

在拟合过程中，Scaler 从不会看到测试数据。这正是其关键所在。

### ColumnTransformer：为不同列使用不同的管道

真实的数据集包含需要不同预处理的数值列和分类列。`ColumnTransformer` 处理这种情况。```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, ["age", "income", "score"]),
    ("cat", categorical_pipe, ["city", "gender", "plan"]),
])

full_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", GradientBoostingClassifier()),
])
```OneHotEncoder 中的 `handle_unknown="ignore"` 在生产环境中至关重要。当出现新的类别（模型从未见过的城市）时，它会生成一个零向量，而不是崩溃。

### 实验跟踪

一个流水线可以使训练过程可重复，但你还需要跟踪不同实验中发生的情况：使用了哪些超参数，使用了哪个数据集版本，指标如何，运行了哪些代码。

**MLflow** 是最常用的开源解决方案：```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)

    pipe.fit(X_train, y_train)
    accuracy = pipe.score(X_test, y_test)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(pipe, "model")
```每次运行都会记录参数、指标、工件和完整的模型。你可以比较运行结果，重现任何实验，并部署任何模型版本。

**Weights & Biases (wandb)** 通过托管仪表板提供相同的功能：

 /no_think

<>

每次运行都会记录参数、指标、工件和完整的模型。你可以比较运行结果，重现任何实验，并部署任何模型版本。

**Weights & Biases (wandb)** 通过托管仪表板提供相同的功能：```python
import wandb

wandb.init(project="my-pipeline")
wandb.config.update({"max_depth": 5, "n_estimators": 100})

pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)

wandb.log({"accuracy": accuracy})
```### 模型版本管理

在实验跟踪之后，你需要管理模型版本。哪个模型正在生产环境中使用？哪个处于测试阶段？哪个是上周的？

MLflow 的模型注册表提供以下功能：
- **版本跟踪：** 每个保存的模型都会获得一个版本号
- **阶段转换：** “测试中”、“生产中”、“已归档”
- **审批流程：** 模型必须经过明确的审批才能进入生产环境
- **回滚：** 可以立即切换回之前的版本

### 使用 DVC 进行数据版本管理

代码使用 git 进行版本管理。数据也应该进行版本管理，但 git 无法处理大文件。DVC（数据版本控制）解决了这个问题。```
dvc init
dvc add data/training.csv
git add data/training.csv.dvc data/.gitignore
git commit -m "Track training data"
dvc push
```DVC 将实际数据存储在远程存储（S3、GCS、Azure）中，并在 git 中保留一个小型的 `.dvc` 文件，该文件记录哈希值。当你切换到一个 git 提交时，`dvc checkout` 会恢复当时使用的精确数据。

这意味着每个 git 提交都会同时固定代码和数据。完全的可复现性。

### 可复现的实验

一个可复现的实验需要以下四点：

1. **固定的随机种子：** 为 numpy、random 和框架（torch、sklearn）设置种子
2. **固定的依赖项：** 用 requirements.txt 或 poetry.lock 并指定精确版本
3. **版本化的数据：** 使用 DVC 或类似工具
4. **配置文件：** 所有超参数放在配置文件中，而不是硬编码```python
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
```### 从笔记本到生产流水线```mermaid
flowchart TD
    A[Jupyter Notebook] --> B[Extract functions]
    B --> C[Build Pipeline object]
    C --> D[Add config file for hyperparameters]
    D --> E[Add experiment tracking]
    E --> F[Add data validation]
    F --> G[Add tests]
    G --> H[Package for deployment]

    style A fill:#fdd,stroke:#333
    style H fill:#dfd,stroke:#333
```典型的开发流程：

1. **笔记本探索：** 快速实验、可视化、特征想法
2. **提取函数：** 将预处理、特征工程、评估移动到模块中
3. **构建管道：** 将转换操作链式连接到 sklearn Pipeline 或自定义类中
4. **配置管理：** 将所有超参数移动到 YAML/JSON 配置文件中
5. **实验跟踪：** 添加 MLflow 或 wandb 日志记录
6. **数据验证：** 在训练之前检查模式、分布和缺失值模式
7. **测试：** 为转换器添加单元测试，为完整管道添加集成测试
8. **部署：** 序列化管道，包装成 API（FastAPI、Flask），容器化

### 常见管道错误

| 错误 | 为什么不好 | 解决方法 |
|-----|----------|---------|
| 在拆分之前对完整数据进行拟合 | 数据泄露 | 使用带有 cross_val_score 的 Pipeline |
| 管道之外进行特征工程 | 训练和部署时转换不一致 | 将所有转换操作放入 Pipeline 中 |
| 没有处理未知类别 | 新值导致生产崩溃 | OneHotEncoder(handle_unknown="ignore") |
| 硬编码列名 | 模式变化时会出错 | 使用配置文件中的列名列表 |
| 没有数据验证 | 在坏数据上静默生成错误预测 | 在预测前添加模式检查 |
| 训练/部署偏差 | 模型在生产中看到不同的特征 | 为训练和部署使用同一个 Pipeline 对象 |

## 构建它

`code/pipeline.py` 中的代码从头开始构建了一个完整的 ML 管道：

### 第一步：自定义转换器```python
class CustomTransformer:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.means) / self.stds

    def fit_transform(self, X):
        return self.fit(X).transform(X)
```### 步骤 2：从零开始构建流水线```python
class PipelineFromScratch:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        X_current = X.copy()
        for name, step in self.steps[:-1]:
            X_current = step.fit_transform(X_current)
        name, model = self.steps[-1]
        model.fit(X_current, y)
        return self

    def predict(self, X):
        X_current = X.copy()
        for name, step in self.steps[:-1]:
            X_current = step.transform(X_current)
        name, model = self.steps[-1]
        return model.predict(X_current)
```### 步骤 3：使用流水线进行交叉验证

代码展示了如何使用流水线进行交叉验证以防止数据泄露：缩放器在每个折叠的训练数据上分别拟合。

### 步骤 4：使用 sklearn 的完整生产流水线

一个完整的流水线，包含 `ColumnTransformer`、多个预处理路径和一个模型，并通过适当的交叉验证和实验日志进行训练。

## 发布它

本课将产出以下内容：
- `outputs/prompt-ml-pipeline.md` -- 构建和调试机器学习流水线的技能
- `code/pipeline.py` -- 从零开始通过 sklearn 构建的完整流水线

## 练习

1. 构建一个流水线，处理包含 3 个数值列和 2 个分类列的数据集。使用 `ColumnTransformer` 对数值列应用中位数填充 + 缩放，对分类列应用最频繁填充 + 一次热编码。使用 5 折交叉验证进行训练。

2. 故意引入数据泄露：在拆分数据集之前，对整个数据集拟合缩放器。将交叉验证得分（泄露）与流水线交叉验证得分（干净）进行比较。差异有多大？

3. 使用 `joblib.dump` 序列化你的流水线。在另一个脚本中加载它并运行预测。验证预测结果是否相同。

4. 向流水线添加一个自定义转换器，为两个最重要的数值列生成多项式特征（2 次）。它应该放在流水线的哪里？

5. 为流水线设置 MLflow 跟踪。运行 5 次实验，使用不同的超参数。使用 MLflow UI (`mlflow ui`) 比较运行结果并选择最佳模型。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|----------------|------------------|
| 流水线 | "转换和模型的链" | 一组已拟合的转换器和模型的有序序列，作为一个单元应用以防止泄露 |
| 数据泄露 | "测试信息泄露到训练中" | 使用训练集以外的信息构建模型，导致性能估计膨胀 |
| ColumnTransformer | "每列不同的预处理" | 对不同列子集应用不同的流水线，合并结果 |
| 实验跟踪 | "记录运行" | 记录每次训练运行的参数、指标、工件和代码版本 |
| MLflow | "跟踪和部署模型" | 用于实验跟踪、模型注册和部署的开源平台 |
| DVC | "数据的 Git" | 大型数据文件的版本控制系统，将哈希存储在 Git 中，数据存储在远程存储中 |
| 模型注册表 | "模型版本目录" | 跟踪带有阶段标签（测试、生产、归档）的模型版本的系统 |
| 训练/服务偏差 | "在笔记本中有效" | 训练和推理过程中数据处理方式的差异，导致静默错误 |
| 可重复性 | "相同代码，相同结果" | 从相同的代码、数据和配置中获得相同结果的能力 |

## 进一步阅读

- [scikit-learn 流水线文档](https://scikit-learn.org/stable/modules/compose.html) -- 官方流水线参考
- [MLflow 文档](https://mlflow.org/docs/latest/index.html) -- 实验跟踪和模型注册
- [DVC 文档](https://dvc.org/doc) -- 数据版本控制
- [Sculley 等人，机器学习系统中的隐藏技术债务 (2015)](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) -- 关于机器学习系统复杂性的开创性论文
- [Google ML 最佳实践：机器学习规则](https://developers.google.com/machine-learning/guides/rules-of-ml) -- 实用生产机器学习建议
