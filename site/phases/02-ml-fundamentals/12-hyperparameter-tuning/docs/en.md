# 超参数调优

> 超参数是在训练开始之前您调整的旋钮。调整得当与否，决定了模型是平庸还是出色。

**类型:** 构建
**语言:** Python
**先决条件:** 第二阶段，第11课（集成方法）
**时间:** 约90分钟

## 学习目标

- 从零开始实现网格搜索、随机搜索和贝叶斯优化，并比较它们的样本效率
- 解释为什么当大多数超参数具有较低的有效维度时，随机搜索的表现优于网格搜索
- 使用替代模型和获取函数构建贝叶斯优化循环，以指导搜索
- 设计一个超参数调优策略，通过适当的交叉验证来避免对验证集的过拟合

## 问题

您的梯度提升模型具有学习率、树的数量、最大深度、每片叶子的最小样本数、子样本比例和列样本比例。这是六个超参数。如果每个参数都有5个合理的值，那么网格的组合数为5^6 = 15,625。训练每个模型需要10秒。要尝试所有组合，需要43小时的计算时间。

网格搜索是显而易见的方法，但也是在规模上最差的方法。随机搜索使用更少的计算资源表现更好。贝叶斯优化通过从过去的评估中学习，表现甚至更好。知道使用哪种策略，以及哪些超参数真正重要，可以节省几天的GPU计算时间。

## 概念

### 参数 vs 超参数

参数是在训练过程中学习的（权重、偏差、分割阈值）。超参数是在训练开始前设置的，控制学习的过程。

| 超参数 | 控制的内容 | 典型范围 |
|------|----------|------|
| 学习率 | 每次更新的步长 | 0.001 到 1.0 |
| 树的数量/周期数 | 训练时长 | 10 到 10,000 |
| 最大深度 | 模型复杂度 | 1 到 30 |
| 正则化（lambda） | 防止过拟合 | 0.0001 到 100 |
| 批量大小 | 梯度估计的噪声 | 16 到 512 |
| Dropout率 | 被丢弃的神经元比例 | 0.0 到 0.5 |

### 网格搜索

网格搜索评估指定值的每一个组合。它虽然全面且易于理解，但随着超参数数量的增加，其计算量呈指数级增长。```
Grid for 2 hyperparameters:

  learning_rate: [0.01, 0.1, 1.0]
  max_depth:     [3, 5, 7]

  Evaluations: 3 x 3 = 9 combinations

  (0.01, 3)  (0.01, 5)  (0.01, 7)
  (0.1,  3)  (0.1,  5)  (0.1,  7)
  (1.0,  3)  (1.0,  5)  (1.0,  7)
```网格搜索有一个根本性的缺陷：如果一个超参数重要而另一个不重要，大多数评估都是浪费的。在9次评估中，你只能得到重要参数的3个唯一值。

### 随机搜索

随机搜索从分布中采样超参数，而不是从网格中采样。在相同的9次评估预算下，每个超参数都能得到9个唯一的值。```mermaid
flowchart LR
    subgraph Grid Search
        G1[3 unique learning rates]
        G2[3 unique max depths]
        G3[9 total evaluations]
    end

    subgraph Random Search
        R1[9 unique learning rates]
        R2[9 unique max depths]
        R3[9 total evaluations]
    end
```为什么随机搜索优于网格搜索（Bergstra & Bengio, 2012）：

- 大多数超参数的有效维度较低。通常对于给定的问题，只有6个超参数中的1-2个是重要的。
- 网格搜索在不重要的维度上浪费评估次数。
- 在相同的预算下，随机搜索在重要的维度上覆盖得更密集。
- 在进行60次随机试验后，你有95%的机会找到一个距离最优值5%以内的点（如果搜索空间中存在这样的点）。

### 贝叶斯优化

随机搜索忽略结果。它不会学习到高学习率会导致发散，或者深度3始终优于深度10。贝叶斯优化利用过去的评估结果来决定下一步搜索的位置。```mermaid
flowchart TD
    A[Define search space] --> B[Evaluate initial random points]
    B --> C[Fit surrogate model to results]
    C --> D[Use acquisition function to pick next point]
    D --> E[Evaluate the model at that point]
    E --> F{Budget exhausted?}
    F -->|No| C
    F -->|Yes| G[Return best hyperparameters found]
```两个关键组件：

**代理模型（Surrogate model）：** 一个易于评估的模型（通常是高斯过程），用于近似昂贵的目标函数。它在搜索空间的任何一点上都能提供预测值和不确定性估计。

**获取函数（Acquisition function）：** 通过在利用（搜索已知良好点附近）和探索（搜索不确定性高的区域）之间取得平衡，决定下一步评估的位置。常见选择包括：

- **期望改进（Expected Improvement, EI）：** 在这一点上，我们预计能比当前最佳值改进多少？
- **上置信界（Upper Confidence Bound, UCB）：** 预测值加上不确定性的倍数。较高的UCB表示该点可能是有前景的或尚未探索的。
- **改进概率（Probability of Improvement, PI）：** 该点超越当前最佳值的概率是多少？

贝叶斯优化通常比随机搜索在2-5倍更少的评估次数下就能找到更好的超参数。拟合代理模型的开销与实际模型的训练相比可以忽略不计。

### 早期停止（Early Stopping）

并非每一轮训练都需要完成。如果某个配置在10个训练周期后明显表现不佳，就停止它并继续进行下一轮。这在超参数搜索的背景下称为早期停止。

策略：
- **基于耐心（Patience-based）：** 如果验证损失在连续N个训练周期内没有改善，就停止。
- **中位数剪枝（Median pruning）：** 如果该试验的中间结果比相同步骤下已完成试验的中位数结果更差，就停止。
- **Hyperband：** 为大量配置分配小预算，然后逐步增加预算给表现最好的配置。

Hyperband特别有效。它从81个配置开始，每个配置进行1个训练周期，保留表现最好的1/3，再为它们分配3个周期，继续保留表现最好的1/3，依此类推。与对所有配置使用完整预算进行评估相比，Hyperband可以将找到良好配置的速度提高10到50倍。

### 学习率调度器（Learning Rate Schedulers）

学习率几乎总是最重要的超参数。与其保持固定，调度器会在训练过程中对其进行调整。

| 调度器 | 公式 | 使用场景 |
|------|------|--------|
| 步长衰减（Step decay） | 每N个周期乘以0.1 | 经典CNN训练 |
| 余弦退火（Cosine annealing） | lr * 0.5 * (1 + cos(pi * t / T)) | 现代默认选择 |
| 预热+衰减（Warmup + decay） | 线性增加后余弦衰减 | Transformer模型 |
| 单周期（One-cycle） | 在一个周期内先增加后减少 | 快速收敛 |
| 平台减少（Reduce on plateau） | 当指标停滞时按比例减少 | 安全默认选择 |

### 超参数重要性

并非所有超参数都同等重要。关于随机森林（Probst等，2019）和梯度提升的研究显示了一致的模式：

**重要性高：**
- 学习率（始终优先调整）
- 估计器数量/训练周期（使用早期停止而非调整）
- 正则化强度

**重要性中等：**
- 最大深度/层数
- 每个叶子的最小样本数/权重衰减
- 子样本比例

**重要性低：**
- 最大特征数（针对随机森林）
- 特定激活函数的选择
- 批量大小（在合理范围内）

优先调整重要的超参数，其余的保持默认值。

### 实用策略```mermaid
flowchart TD
    A[Start with defaults] --> B[Coarse random search: 20-50 trials]
    B --> C[Identify important hyperparameters]
    C --> D[Fine random or Bayesian search: 50-100 trials in narrowed space]
    D --> E[Final model with best hyperparameters]
    E --> F[Retrain on full training data]
```具体的工作流程：

1. **从库的默认值开始。** 它们由经验丰富的实践者选择，通常已经达到了80%的效果。
2. **粗粒度随机搜索。** 使用广泛的范围，进行20-50次试验。使用提前停止机制快速终止表现差的运行。
3. **分析结果。** 哪些超参数与性能相关？缩小搜索空间。
4. **精细搜索。** 在缩小后的空间中使用贝叶斯优化或聚焦的随机搜索。进行50-100次试验。
5. **使用找到的最佳超参数在全部训练数据上重新训练。**

### 交叉验证集成

仅在一个验证划分上调整超参数是具有风险的。最佳的超参数可能会对特定的验证划分过拟合。嵌套交叉验证通过使用两个循环来解决这个问题：

- **外层循环**（评估）：将数据划分为训练+验证和测试。报告无偏的性能。
- **内层循环**（调整）：将训练+验证划分为训练和验证。找到最佳的超参数。```mermaid
flowchart TD
    D[Full Dataset] --> O1[Outer Fold 1: Test]
    D --> O2[Outer Fold 2: Test]
    D --> O3[Outer Fold 3: Test]
    D --> O4[Outer Fold 4: Test]
    D --> O5[Outer Fold 5: Test]

    O1 --> I1[Inner 5-fold CV on remaining data]
    I1 --> T1[Best hyperparams for fold 1]
    T1 --> E1[Evaluate on outer test fold 1]

    O2 --> I2[Inner 5-fold CV on remaining data]
    I2 --> T2[Best hyperparams for fold 2]
    T2 --> E2[Evaluate on outer test fold 2]
```每个外部折叠独立地找到自己的最佳超参数。外部评分是对泛化性能的无偏估计。

使用 sklearn:```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

inner_cv = GridSearchCV(
    GradientBoostingRegressor(),
    param_grid={
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5],
        "n_estimators": [50, 100, 200],
    },
    cv=5,
    scoring="neg_mean_squared_error",
)

outer_scores = cross_val_score(
    inner_cv, X, y, cv=5, scoring="neg_mean_squared_error"
)

print(f"Nested CV MSE: {-outer_scores.mean():.4f} +/- {outer_scores.std():.4f}")
```这会很昂贵（5个外部折叠 × 5个内部折叠 × 27个网格点 = 675个模型拟合），但它能给你一个可信的性能估计。在论文中报告最终结果，或者当决策的风险很高时，使用它。

### 实用技巧

**从学习率开始。** 对于基于梯度的方法，它始终是最重要的超参数。一个差的学习率会使其他一切变得无关紧要。将其他超参数固定在默认值，并首先进行学习率的扫描。

**对学习率和正则化使用对数均匀分布。** 0.001和0.01之间的差异与0.1和1.0之间的差异一样重要。线性搜索会在较大的数值端浪费预算。

**使用早停而不是调整n_estimators。** 对于提升方法和神经网络，将n_estimators或epochs设为高值，让早停决定何时停止。这可以将一个超参数从搜索中移除。

**预算分配。** 将你调整预算的60%用在最重要的两个超参数上。将剩余的40%用于其他所有内容。最重要的两个超参数占性能变化的大部分。

**尺度很重要。** 永远不要在对数尺度上搜索批量大小（16、32、64是合适的）。始终在对数尺度上搜索学习率。将搜索分布与超参数如何影响模型相匹配。

| 模型类型 | 顶级超参数 | 推荐搜索 | 预算 |
|------|---|----------|--------|
| 随机森林 | n_estimators, max_depth, min_samples_leaf | 随机搜索，50次试验 | 低（训练速度快） |
| 梯度提升 | learning_rate, n_estimators, max_depth | 贝叶斯，100次试验 + 早停 | 中等 |
| 神经网络 | learning_rate, weight_decay, batch_size | 贝叶斯或随机，100+次试验 | 高（训练速度慢） |
| SVM | C, gamma (RBF核) | 在对数尺度上进行网格搜索，25-50次试验 | 低（2个参数） |
| Lasso/Ridge | alpha | 在对数尺度上进行1D搜索，20次试验 | 非常低 |
| XGBoost | learning_rate, max_depth, subsample, colsample | 贝叶斯，100-200次试验 + 早停 | 中等 |

**当不确定时：** 进行随机搜索，试验次数是超参数数量的两倍（例如，6个超参数 = 至少12次试验）。你将惊讶于随机搜索进行50次试验时，常常能击败精心设计的网格搜索。```figure
k-fold-cv
```## 构建它

### 第一步：从零开始实现网格搜索

`code/tuning.py` 中的代码实现了从零开始的网格搜索、随机搜索和一个简单的贝叶斯优化器。```python
def grid_search(model_fn, param_grid, X_train, y_train, X_val, y_val):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    best_score = -float("inf")
    best_params = None
    n_evals = 0

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        model = model_fn(**params)
        model.fit(X_train, y_train)
        score = evaluate(model, X_val, y_val)
        n_evals += 1

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score, n_evals
```### 步骤 2：从零开始的随机搜索```python
def random_search(model_fn, param_distributions, X_train, y_train,
                  X_val, y_val, n_iter=50, seed=42):
    rng = np.random.RandomState(seed)
    best_score = -float("inf")
    best_params = None

    for _ in range(n_iter):
        params = {k: sample(v, rng) for k, v in param_distributions.items()}
        model = model_fn(**params)
        model.fit(X_train, y_train)
        score = evaluate(model, X_val, y_val)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score, n_iter
```### 步骤 3：贝叶斯优化（简化版）

核心思想：将高斯过程拟合到观察到的（超参数，得分）对上，然后使用获取函数来决定下一步要在哪里查找。```python
class SimpleBayesianOptimizer:
    def __init__(self, search_space, n_initial=5):
        self.search_space = search_space
        self.n_initial = n_initial
        self.X_observed = []
        self.y_observed = []

    def _kernel(self, x1, x2, length_scale=1.0):
        dists = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * dists / length_scale ** 2)

    def _fit_gp(self, X_new):
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)
        y_mean = y_obs.mean()
        y_centered = y_obs - y_mean

        K = self._kernel(X_obs, X_obs) + 1e-4 * np.eye(len(X_obs))
        K_star = self._kernel(X_new, X_obs)

        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
        mu = K_star @ alpha + y_mean

        v = np.linalg.solve(L, K_star.T)
        var = 1.0 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-6)

        return mu, var

    def _expected_improvement(self, mu, var, best_y):
        sigma = np.sqrt(var)
        z = (mu - best_y) / (sigma + 1e-10)
        ei = sigma * (z * norm_cdf(z) + norm_pdf(z))
        return ei

    def suggest(self):
        if len(self.X_observed) < self.n_initial:
            return sample_random(self.search_space)

        candidates = [sample_random(self.search_space) for _ in range(500)]
        X_cand = np.array([to_vector(c) for c in candidates])
        mu, var = self._fit_gp(X_cand)
        ei = self._expected_improvement(mu, var, max(self.y_observed))
        return candidates[np.argmax(ei)]

    def observe(self, params, score):
        self.X_observed.append(to_vector(params))
        self.y_observed.append(score)
```GP 代理模型在每个候选点给出两个值：一个预测得分（mu）和一个不确定性（var）。期望改进（Expected Improvement）在两者之间进行权衡：它倾向于模型预测得分高或者不确定性高的点。早期阶段，大部分点的不确定性较高，因此优化器会进行探索。后期，它会集中在最有希望的区域。

### 第四步：比较所有方法

在相同的合成目标函数上运行所有三种方法并进行比较。这个比较使用了一个简化的包装器，它直接用目标函数调用每个优化器（没有模型训练），因此 API 与上面基于模型的实现不同：```python
def synthetic_objective(params):
    lr = params["learning_rate"]
    depth = params["max_depth"]
    return -(np.log10(lr) + 2) ** 2 - (depth - 4) ** 2 + 10

param_grid = {
    "learning_rate": [0.001, 0.01, 0.1, 1.0],
    "max_depth": [2, 3, 4, 5, 6, 7, 8],
}

grid_best = None
grid_score = -float("inf")
grid_history = []
for combo in itertools.product(*param_grid.values()):
    params = dict(zip(param_grid.keys(), combo))
    score = synthetic_objective(params)
    grid_history.append((params, score))
    if score > grid_score:
        grid_score = score
        grid_best = params

param_dist = {
    "learning_rate": ("log_float", 0.001, 1.0),
    "max_depth": ("int", 2, 8),
}

rand_best = None
rand_score = -float("inf")
rand_history = []
rng = np.random.RandomState(42)
for _ in range(28):
    params = {k: sample(v, rng) for k, v in param_dist.items()}
    score = synthetic_objective(params)
    rand_history.append((params, score))
    if score > rand_score:
        rand_score = score
        rand_best = params

optimizer = SimpleBayesianOptimizer(param_dist, n_initial=5)
bayes_history = []
for _ in range(28):
    params = optimizer.suggest()
    score = synthetic_objective(params)
    optimizer.observe(params, score)
    bayes_history.append((params, score))
bayes_score = max(s for _, s in bayes_history)

print(f"{'Method':<20} {'Best Score':>12} {'Evaluations':>12}")
print("-" * 50)
print(f"{'Grid Search':<20} {grid_score:>12.4f} {len(grid_history):>12}")
print(f"{'Random Search':<20} {rand_score:>12.4f} {len(rand_history):>12}")
print(f"{'Bayesian Opt':<20} {bayes_score:>12.4f} {len(bayes_history):>12}")
```在相同的预算下，贝叶斯优化通常能最快找到最佳得分，因为它不会在明显较差的区域浪费评估。随机搜索覆盖的范围比网格搜索更广。只有在超参数非常少且可以承受穷举时，网格搜索才会占优。

## 使用它

### 实践中的 Optuna

Optuna 是用于严肃超参数调优的推荐库。它内置支持剪枝、分布式搜索和可视化功能。```python
import optuna

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    n_est = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 2, 10)

    model = GradientBoostingRegressor(
        learning_rate=lr,
        n_estimators=n_est,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)
    return mean_squared_error(y_val, model.predict(X_val))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
print(f"Best MSE: {study.best_value:.4f}")
```Optuna 的主要特性：
- 使用 `suggest_float(..., log=True)` 对以对数尺度最佳搜索的参数（如学习率、正则化）进行优化
- 使用 `suggest_int` 对整数参数进行优化
- 使用 `suggest_categorical` 对离散选项进行优化
- 内置 MedianPruner 用于提前停止表现不佳的试验
- 使用 `study.trials_dataframe()` 进行分析

### 使用 Pruning 的 Optuna

Pruning 会提前停止没有前景的试验，从而节省大量计算资源。以下是其使用模式：```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("lr", 1e-4, 0.5, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }

    model = GradientBoostingRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3,
                             scoring="neg_mean_squared_error")
    mean_score = -scores.mean()

    trial.report(mean_score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return mean_score

pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=200)
````MedianPruner` 如果其在相同步骤中的中间值比所有已完成试验的中位数更差，就会停止试验。剪枝需要调用 `trial.report()` 来报告中间指标，并调用 `trial.should_prune()` 来检查是否应停止试验。`n_startup_trials=10` 确保在开始剪枝之前至少有 10 个试验完全完成。这通常可以节省 40-60% 的总计算量。

### sklearn 内置的调参器

为了快速实验，sklearn 提供了 `GridSearchCV`、`RandomizedSearchCV` 和 `HalvingRandomSearchCV`：```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

param_dist = {
    "learning_rate": loguniform(1e-4, 0.5),
    "max_depth": randint(2, 10),
    "n_estimators": randint(50, 500),
}

search = RandomizedSearchCV(
    GradientBoostingRegressor(),
    param_dist,
    n_iter=100,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
print(f"Best CV MSE: {-search.best_score_:.4f}")
```使用 scipy 中的 `loguniform` 来设置学习率和正则化。使用 `randint` 来设置整数类型的超参数。使用 `n_jobs=-1` 标志可在所有 CPU 核心上进行并行处理。

### 超参数调优的常见错误

**通过预处理导致的数据泄露。** 如果在交叉验证之前对整个数据集拟合一个缩放器，那么验证集的信息会泄露到训练集中。始终将预处理操作放在 `Pipeline` 中，使其仅在训练集上进行拟合。

**对验证集过拟合。** 运行数千次试验实际上是在对验证集进行训练。使用嵌套交叉验证进行最终性能估计，或者保留一个从未在调优过程中接触过的独立测试集。

**搜索范围太窄。** 如果你的最佳值在搜索空间的边界上，那么你的搜索范围不够广。最佳值可能在你的范围之外。始终检查最佳参数是否在边缘。

**忽略交互效应。** 在提升算法中，学习率和估计器数量之间有强烈的交互作用。低学习率需要更多的估计器。独立调整它们的效果不如一起调整。

**未对迭代模型使用早停。** 对于梯度提升和神经网络，将 n_estimators 或 epochs 设置为一个较高的值，并使用早停。这比将迭代次数作为超参数进行调整要严格更好。

## 练习

1. 使用相同的总预算（例如，50 次评估）运行网格搜索和随机搜索。比较找到的最佳分数。使用不同的种子运行实验 10 次。随机搜索获胜的频率是多少？

2. 从头开始实现 Hyperband。从 81 个配置开始，每个配置训练 1 个 epoch。每轮保留前 1/3 的配置，并将它们的预算增加三倍。将总计算量（所有配置所有 epoch 的总和）与运行 81 个配置到完整预算的计算量进行比较。

3. 在第 11 课的梯度提升实现中添加一个学习率调度器（余弦退火）。与固定学习率相比，它是否有帮助？

4. 使用 Optuna 在真实数据集（如 sklearn 的乳腺癌数据集）上调整 RandomForestClassifier。使用 `optuna.visualization.plot_param_importances(study)` 查看哪些超参数最重要。它是否与本课的超参数重要性排序一致？

5. 实现一个简单的获取函数（预期改进），并演示探索与利用之间的权衡。绘制代理模型的均值和不确定性，并显示 EI 选择下次评估的位置。

## 重要术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------|
| 超参数 | "一个你自己选择的设置" | 在训练之前设置的值，控制学习过程，不从数据中学习 |
| 网格搜索 | "尝试所有组合" | 在指定的参数网格上进行穷举搜索。成本呈指数增长。 |
| 随机搜索 | "只是随机采样" | 从分布中采样超参数。比网格搜索更好地覆盖重要维度。 |
| 贝叶斯优化 | "智能搜索" | 使用目标函数的代理模型来决定下一步评估的位置，平衡探索和利用 |
| 代理模型 | "一个廉价的近似" | 一个模型（通常是高斯过程），根据观察到的评估近似昂贵的目标函数 |
| 获取函数 | "下一步要查看哪里" | 通过平衡预期改进与不确定性对候选点进行评分。EI 和 UCB 是常见选择。 |
| 早停 | "停止浪费时间" | 当验证性能停止提升时，提前终止训练 |
| Hyperband | "配置的锦标赛括号" | 自适应资源分配：从许多配置开始，使用小预算，保留最好的并增加它们的预算 |
| 学习率调度器 | "在训练过程中调整学习率" | 一个函数，随着训练过程的进行调整学习率以实现更好的收敛 |

## 进一步阅读

- [Bergstra & Bengio: 随机搜索用于超参数优化 (2012)](https://jmlr.org/papers/v13/bergstra12a.html) -- 展示随机搜索优于网格搜索的论文
- [Snoek 等人，机器学习算法的实用贝叶斯优化 (2012)](https://arxiv.org/abs/1206.2944) -- 用于机器学习的贝叶斯优化
- [Li 等人，Hyperband：一种基于 Bandit 的新方法 (2018)](https://jmlr.org/papers/v18/16-558.html) -- Hyperband 的论文
- [Optuna：下一代超参数优化框架](https://arxiv.org/abs/1907.10902) -- Optuna 的论文
- [Probst 等人，可调性：超参数的重要性 (2019)](https://jmlr.org/papers/v20/18-444.html) -- 哪些超参数最重要
