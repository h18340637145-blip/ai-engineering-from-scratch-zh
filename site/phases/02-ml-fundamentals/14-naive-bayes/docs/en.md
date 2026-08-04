# 朴素贝叶斯

> “朴素”的假设是错误的，但它依然有效。这就是它的魅力所在。

**类型:** 构建
**语言:** Python
**先决条件:** 第二阶段，第01-07课（分类，贝叶斯定理）
**时间:** ~75分钟

## 学习目标

- 使用拉普拉斯平滑从零开始实现用于文本分类的多项式朴素贝叶斯
- 解释为什么朴素的独立性假设在数学上是错误的，但在实践中却能产生正确的分类排名
- 比较多项式、伯努利和高斯朴素贝叶斯的变体，并根据给定的特征类型选择合适的模型
- 在高维稀疏数据上将朴素贝叶斯与逻辑回归进行比较，并解释其中的偏差-方差权衡

## 问题

你需要对文本进行分类。将电子邮件分为垃圾邮件或非垃圾邮件。将客户评论分为正面或负面。将支持票分类到不同类别中。你有成千上万的特征（每个单词一个特征），但训练数据有限。

大多数分类器在这里会遇到困难。逻辑回归需要足够的样本才能可靠地估计成千上万的权重。决策树一次只根据一个单词进行拆分，容易过度拟合。在10,000维空间中，KNN毫无意义，因为每个点与其他点之间的距离都是一样的。

朴素贝叶斯可以处理这个问题。它做出一个数学上错误的假设（即在给定类别的情况下，每个特征与其他特征是独立的），但它在文本分类任务中依然优于“更聪明”的模型，尤其是在训练数据量较小的情况下。它只需要遍历数据一次即可训练。它可以扩展到数百万个特征。它可以生成概率估计（尽管由于独立性假设，通常校准得不够好）。

理解为什么一个错误的假设能够导致良好的预测，可以让你了解到机器学习中的一个基本原理：最好的模型不是最正确的模型，而是对你的数据具有最佳偏差-方差权衡的模型。

## 概念

### 贝叶斯定理（快速回顾）

贝叶斯定理可以翻转条件概率：```
P(class | features) = P(features | class) * P(class) / P(features)
```我们想要 `P(class | features)` -- 给定文档中的词语，文档属于某一类的概率。我们可以从以下内容计算出这个概率：
- `P(features | class)` -- 在这类文档中看到这些词语的可能性
- `P(class)` -- 类的先验概率（垃圾邮件总体上有多普遍？）
- `P(features)` -- 证据，对所有类都相同，因此在比较时可以忽略它

具有最高 `P(class | features)` 的类获胜。

### 朴素独立性假设

精确计算 `P(features | class)` 需要估计所有特征一起出现的联合概率。如果词汇表有 10,000 个词，你需要估计 2^10,000 种可能组合的分布。这是不可能的。

朴素假设：在给定类别的情况下，每个特征都是相互独立的。```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```与其估计一个不可能的联合分布，你估计 n 个简单的每个特征的分布。每个分布只需要一个计数。

这个假设显然是错误的。在任何文档中，“machine”和“learning”这两个词并不是独立的。但分类器并不需要正确的概率估计。它只需要正确的排序——哪个类别的概率最高。独立性假设引入了系统性误差，但这些误差对所有类别都影响相似，因此排序仍然正确。

### 为什么它仍然有效

三个原因：

1. **排序胜过校准。** 分类只需要最高排名的类别是正确的。即使当真实概率是 0.7 时，P(spam) = 0.99999，分类器仍然正确地选择 spam。我们不需要正确的概率。我们需要正确的胜者。

2. **高偏差，低方差。** 独立性假设是一个强先验。它对模型施加了严格的限制，这可以防止过拟合。在有限的训练数据下，一个略微错误但稳定的模型胜过一个理论上正确但极不稳定模型。这就是偏差-方差权衡的体现。

3. **特征冗余抵消。** 相关的特征提供了冗余的证据。分类器会重复计算这些证据，但对正确的类别也会重复计算。如果“machine”和“learning”总是同时出现，两者都为“tech”类别提供证据。朴素贝叶斯会将它们计算两次，但对正确的类别也计算两次。

第四个实际原因：朴素贝叶斯非常快。训练只需要一次遍历数据来统计频率。预测是一个矩阵乘法。你可以在几秒钟内对一百万份文档进行训练。这种速度意味着你可以更快地迭代，尝试更多的特征集，并比使用较慢模型运行更多的实验。

### 数学步骤详解

让我们通过一个具体例子来追踪。假设我们有两个类别：spam 和 not-spam。我们的词汇表有三个词：“free”、“money”、“meeting”。

训练数据：
- Spam 电子邮件中，“free” 出现 80 次，“money” 出现 60 次，“meeting” 出现 10 次（总共 150 个词）
- Not-spam 电子邮件中，“free” 出现 5 次，“money” 出现 10 次，“meeting” 出现 100 次（总共 115 个词）
- 40% 的电子邮件是 spam，60% 是 not-spam

使用拉普拉斯平滑（alpha=1）：```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```新邮件包含： "free"（2次），"money"（1次），"meeting"（0次）。```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```垃圾邮件以较大优势获胜。单词“免费”出现两次是垃圾邮件的有力证据。请注意，“会议”一词未出现对两个对数总和的贡献均为零（0 * log(P)）——在多项式朴素贝叶斯中，未出现的单词没有影响。是伯努利朴素贝叶斯明确地对单词的缺失进行了建模。

### 三种变体

朴素贝叶斯有三种变体。每种变体对 `P(feature | class)` 的建模方式不同。

#### 多项式朴素贝叶斯

将每个特征建模为计数。最适合文本数据，其中特征是单词频率或 TF-IDF 值。```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
````alpha` 是拉普拉斯平滑（下文将进行解释）。这种变体是文本分类的主要方法。

#### 高斯朴素贝叶斯

将每个特征建模为正态分布。最适合连续特征。```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```每个类别在每个特征上都有自己的均值和方差。当每个类别中的特征确实遵循钟形曲线时，这种方法效果很好。

#### 伯努利朴素贝叶斯

将每个特征建模为二元（存在或不存在）。最适合短文本或二元特征向量。```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```与多项分布不同，伯努利分布明确惩罚某个词语的缺失。如果“免费”一词通常出现在垃圾邮件中，但在这封电子邮件中却缺失了，伯努利分布会将这种情况视为反对垃圾邮件的证据。

### 何时使用每种变体

| 变体     | 特征类型         | 最适合           | 示例             |
|----------|------------------|------------------|------------------|
| 多项分布   | 计数或频率       | 文本分类，词袋模型 | 电子邮件垃圾分类，主题分类 |
| 高斯分布   | 连续值           | 具有近似正态特征的表格数据 | 鸢尾花分类，传感器数据   |
| 伯努利分布 | 二进制（0/1）    | 短文本，二进制特征向量 | 短信垃圾分类，存在/缺失特征 |

### 拉普拉斯平滑

当一个词语出现在测试数据中，但在某个类别的训练数据中从未出现过时会发生什么？

不进行平滑处理时：`P(word | class) = 0/N = 0`。一个零乘以整个乘积会使`P(class | features) = 0`，不管其他所有证据如何。一个未见过的词会破坏整个预测，无论其他证据有多强。

拉普拉斯平滑向每个特征计数中添加了一个小的计数 `alpha`（通常为1）：```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```当 alpha=1 时，每个词至少会获得一个极小的概率。在测试邮件中出现的单词 "discombobulate" 不再会使垃圾邮件的概率归零。平滑处理具有贝叶斯解释：它等价于对词分布施加一个均匀的 Dirichlet 先验。

更高的 alpha 值意味着更强的平滑处理（分布更均匀）。更低的 alpha 值意味着模型对数据的信任程度更高。alpha 是一个需要调节的超参数。

alpha 的影响：

| Alpha | 效果 | 使用时机 |
|-------|------|----------|
| 0.001 | 几乎没有平滑，完全信任数据 | 训练集非常大，预计不会出现新特征 |
| 0.1 | 轻度平滑 | 训练集较大 |
| 1.0 | 标准的拉普拉斯平滑 | 默认起始点 |
| 10.0 | 强烈平滑，使分布更加平坦 | 训练集非常小，预计会出现很多新特征 |

### 对数空间计算

将数百个概率（每个都小于 1）相乘会导致浮点下溢。在浮点运算中，乘积会变成零，尽管其真实值是一个非常小的正数。

解决方案：在对数空间中进行计算。不是将概率相乘，而是将它们的对数相加：```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```这将预测转化为点积：```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```矩阵乘法。这就是为什么朴素贝叶斯预测如此快速——它与单层线性模型执行的是相同的运算。

### 朴素贝叶斯与逻辑回归

两者都是用于文本的线性分类器。它们的区别在于所建模的内容。

| 方面 | 朴素贝叶斯 | 逻辑回归 |
|------|------------|-----------|
| 类型 | 生成式（建模 P(X|Y)） | 判别式（建模 P(Y|X)） |
| 训练 | 统计频率 | 优化损失函数 |
| 小数据 | 更好（强先验有帮助） | 更差（不足以估计权重） |
| 大数据 | 更差（错误假设带来伤害） | 更好（灵活的边界） |
| 特征 | 假设独立性 | 处理相关性 |
| 速度 | 单次遍历，非常快 | 迭代优化 |
| 校准 | 概率较差 | 概率较好 |

经验法则：从朴素贝叶斯开始。如果你有足够的数据且朴素贝叶斯达到平台期，就切换到逻辑回归。

### 分类流程```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```实际上，为了避免浮点下溢，我们通常在对数空间中进行计算。我们不是将许多小概率相乘，而是将它们的对数相加：```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```## 构建它

`code/naive_bayes.py` 中的代码从零开始实现了 MultinomialNB 和 GaussianNB。

### MultinomialNB

从零开始的实现：

1. **fit(X, y)**: 对于每个类别，统计每个特征的频率。添加拉普拉斯平滑。计算对数概率。存储类别先验（类别频率的对数）。

2. **predict_log_proba(X)**: 对于每个样本，计算 log P(类别) + 所有类别中 log P(特征_i | 类别) 的总和。这是一个矩阵乘法：X @ log_probs.T + log_priors。

3. **predict(X)**: 返回具有最高对数概率的类别。```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```关键见解：拟合后，预测只是矩阵乘法加上一个偏置。这就是为什么朴素贝叶斯如此快速。

### GaussianNB

对于连续特征，我们按每个特征每个类别估计均值和方差：```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```预测使用每个特征的高斯概率密度函数（PDF），并在特征之间相乘（在对数空间中相加）。

### 示例：文本分类

代码生成合成的词袋数据，模拟两个类别（科技文章与体育文章）。每个类别的单词频率分布不同。MultinomialNB 使用单词计数对它们进行分类。

合成数据的工作方式如下：我们创建 200 个“单词”（特征列）。单词 0-39 在科技文章中高频出现而在体育文章中低频；单词 80-119 在体育文章中高频出现而在科技文章中低频；单词 40-79 在两者中均为中等频率。这创建了一个现实的场景，其中一些单词是强类别指示器，而其他单词则是噪声。

### 示例：连续特征

代码生成类似鸢尾花的数据（3 个类别，4 个特征，高斯分布的聚类）。GaussianNB 使用每个类别的均值和方差进行分类。每个类别的中心（均值向量）和扩展（方差）不同，模拟现实世界中类别之间测量值系统性差异的数据。

代码还展示了以下内容：
- **平滑比较：** 使用不同的 alpha 值训练 MultinomialNB，以展示平滑强度对准确率的影响。
- **训练样本大小实验：** 随着训练数据从 20 个样本增长到 1600 个样本，NB 准确率的提升。即使样本非常少，NB 也能达到不错的准确率——这是它的主要优势。
- **混淆矩阵：** 每个类别的精确率、召回率和 F1 分数，以展示 NB 出错的地方。

### 预测速度

朴素贝叶斯预测是一个矩阵乘法。对于 n 个样本、d 个特征和 k 个类别：
- MultinomialNB：一次矩阵乘法（n x d）@（d x k）= O(n * d * k)
- GaussianNB：n * k 个高斯 PDF 评估，每个评估涉及 d 个特征 = O(n * d * k)

两者在每个维度上都是线性的。与 KNN（需要计算与所有训练点的距离）或带有 RBF 核的 SVM（需要与所有支持向量进行核评估）相比，NB 在预测时间上快几个数量级。

## 使用它

使用 sklearn，两种变体都只需一行代码：

```python
from sklearn.naive_bayes import MultinomialNB, GaussianNB
``````python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```使用 sklearn 进行文本分类：```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
````naive_bayes.py` 中的代码将从零开始的实现与 sklearn 在相同数据上的实现进行比较，以验证正确性。

### 使用朴素贝叶斯的 TF-IDF

原始词频统计给每个词的每次出现赋予相同的权重。但是像 "the" 和 "is" 这样的常见词在每个类别中都会频繁出现，它们并不携带信息。TF-IDF（词频 - 逆文档频率）会降低常见词的权重，同时提升罕见且有区分性的词的权重。```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```TF-IDF 值是非负的，因此它们可以与 MultinomialNB 一起使用。TF-IDF 加上 MultinomialNB 的组合是文本分类中最强的基线之一。在训练样本少于 10,000 的数据集上，它经常能击败更复杂的模型。

### 短文本的 BernoulliNB

对于短文本（如推文、短信、聊天消息），BernoulliNB 可能会优于 MultinomialNB。短文本的词汇量较低，因此 MultinomialNB 依赖的频率信息会变得嘈杂。BernoulliNB 仅关注词的出现或缺失，这在短文本中更为可靠。```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```CountVectorizer 中的 `binary=True` 标志会将所有计数转换为 0/1。如果没有这个标志，BernoulliNB 仍然可以工作，但它看到的是它没有设计用来处理的计数。

### 校准 NB 概率

NB 概率的校准效果较差。当 NB 说 P(垃圾邮件) = 0.95 时，实际概率可能是 0.7。如果你需要可靠的概率估计（例如，设置阈值或与其他模型结合使用），请使用 sklearn 的 CalibratedClassifierCV：```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```这在 NB 的原始得分上拟合了一个逻辑回归，使用交叉验证。得到的概率更接近真实的类别频率。

### 常见陷阱

1. **负的特征值。** MultinomialNB 需要非负特征。如果你有负值（比如某些设置下的 TF-IDF 或标准化特征），使用 GaussianNB 或者将特征转换为正值。

2. **零方差特征。** GaussianNB 会除以方差。如果某个特征在某个类别中方差为零（所有值相同），概率计算就会失败。代码会向所有方差添加一个很小的平滑项（1e-9）以避免这种情况。

3. **类别不平衡。** 如果 99% 的电子邮件都不是垃圾邮件，先验 P(非垃圾邮件) = 0.99 这么强以至于会压倒似然证据。你可以手动设置类别先验，或者使用 sklearn 的 class_prior 参数。

4. **特征缩放。** MultinomialNB 不需要缩放（它基于计数）。GaussianNB 也不需要缩放（它估计每个特征的统计信息）。这相对于对特征尺度敏感的逻辑回归和 SVM 是一个优势。

## 发布它

本课产出：
- `outputs/skill-naive-bayes-chooser.md` -- 选择合适的 NB 变体的决策技能
- `code/naive_bayes.py` -- 从头实现 MultinomialNB 和 GaussianNB，并与 sklearn 进行比较

### 朴素贝叶斯失败的情况

当独立性假设导致错误的排名（不仅仅是错误的概率）时，朴素贝叶斯会失败。这发生在以下情况：

1. **强特征交互。** 如果类别取决于两个特征的组合但不单独取决于任何一个（类似异或模式），朴素贝叶斯将完全无法识别。每个特征单独提供不了证据，朴素贝叶斯也无法非线性地将它们组合起来。

2. **高度相关特征与相反证据。** 如果特征 A 说“垃圾邮件”，特征 B 说“非垃圾邮件”，但 A 和 B 完全相关（它们在现实中总是同意），朴素贝叶斯将看到不存在的冲突证据。

3. **非常大的训练集。** 随着数据量足够大，像逻辑回归这样的判别模型会学习到真实的决策边界并超越朴素贝叶斯。帮助小数据集的独立性假设现在反而限制了模型。

在实践中，文本分类中这些失败情况很少见。文本特征众多，单独较弱，独立性假设的错误往往会相互抵消。对于具有少量强相关特征的表格数据，优先考虑逻辑回归或树模型。

## 练习

1. **平滑实验。** 在文本数据上使用 alpha 值为 0.01、0.1、1.0、10.0 和 100.0 训练 MultinomialNB。绘制准确率与 alpha 的关系图。性能在何处达到峰值？为什么非常高的 alpha 会损害性能？

2. **特征独立性测试。** 使用一个真实的文本数据集。选择两个明显相关的词（例如“machine”和“learning”）。计算 P(word1 | class) * P(word2 | class) 并与 P(word1 AND word2 | class) 进行比较。独立性假设错得有多远？是否影响分类准确率？

3. **伯努利实现。** 扩展代码，添加一个 BernoulliNB 类。将词袋转换为二进制（存在/不存在），并与 MultinomialNB 在文本数据上的准确率进行比较。伯努利在什么情况下表现更好？

4. **朴素贝叶斯与逻辑回归。** 在文本数据上训练两者。从 100 个训练样本开始，增加到 10,000 个。绘制两者准确率与训练集大小的关系图。逻辑回归在什么点上超过朴素贝叶斯？

5. **垃圾邮件过滤器。** 构建一个完整的垃圾邮件分类器：对原始电子邮件文本进行分词，构建词汇表，创建词袋特征，训练 MultinomialNB，并用精确率和召回率（而不仅仅是准确率——为什么？）进行评估。

## 关键术语

| 术语 | 人们常说 | 它实际上意味着 |
|------|----------------|----------------|
| 朴素贝叶斯 | “简单的概率分类器” | 一个假设特征在给定类别下条件独立的分类器 |
| 条件独立 | “特征之间没有影响” | P(A, B | C) = P(A | C) * P(B | C) —— 知道 B 不会告诉你关于 A 的新信息，一旦你知道了 C |
| 拉普拉斯平滑 | “加一平滑” | 为每个特征添加一个很小的计数，以防止零概率主导预测 |
| 先验 | “在看到数据之前你的信念” | P(class) —— 在观察任何特征之前每个类别的概率 |
| 似然 | “数据拟合程度” | P(features | class) —— 如果类别已知，观察这些特征的概率 |
| 后验 | “看到数据后的信念” | P(class | features) —— 观察特征后类别概率的更新值 |
| 生成模型 | “建模数据是如何生成的” | 学习 P(X | Y) 和 P(Y)，然后使用贝叶斯定理得到 P(Y | X) 的模型 |
| 判别模型 | “建模决策边界” | 直接学习 P(Y | X) 而不建模 X 是如何生成的模型 |
| 对数概率 | “避免下溢” | 使用 log P 而不是 P 以防止许多小数相乘在浮点数中变为零 |

## 进一步阅读

- [scikit-learn 朴素贝叶斯文档](https://scikit-learn.org/stable/modules/naive_bayes.html) -- 三种变体及其数学细节
- [McCallum 和 Nigam，朴素贝叶斯文本分类的事件模型比较 (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- 多项式与伯努利文本分类的经典比较
- [Rennie 等，解决朴素贝叶斯文本分类器的差假设 (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) -- 文本分类中朴素贝叶斯的改进
- [Ng 和 Jordan，判别模型与生成模型分类器比较 (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) -- 证明朴素贝叶斯在数据较少时收敛速度比逻辑回归快
