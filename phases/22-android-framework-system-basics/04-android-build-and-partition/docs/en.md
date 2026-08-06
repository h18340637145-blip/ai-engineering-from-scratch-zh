# Android 构建与分区配置：模块不是只要能编译

> 模块是否安装到正确分区、使用正确证书和权限边界，与“构建成功”同样重要。

**Type:** Build
**Languages:** Python
**Prerequisites:** 03-zygote-and-system-server
**Time:** ~90 分钟

## 学习目标

- 执行 AOSP 环境初始化、lunch 与目标构建的基本流程
- 阅读 `Android.mk` 与 Android.bp 的核心字段
- 区分 system、system_ext、product 和 vendor 的安装归属
- 解释 `priv-app`、平台签名与权限白名单的组合关系
- 在产品策略变化时验证模块路径而非照抄示例

## 概念

`lunch` 选择产品与变体，产物通常在 `out/target/product/<product>/`。Android.mk 使用 Make 描述模块；Android.bp 是 Soong 的声明式文件。新模块应先参考同目录相近模块，不应随意混用分区属性。

```mermaid
flowchart LR
    A[模块需求] --> B[确定模块类型]
    B --> C[选择分区]
    C --> D[选择证书与 privileged 标记]
    D --> E[检查产品配置引用]
    E --> F[构建目标模块]
    F --> G[验证安装路径与权限]
```

放入 `priv-app` 并不会自动获得所有特权能力：平台签名、privapp allowlist 和 SELinux 等条件仍需满足。这里的可运行模型采用保守的教学分区策略，真实产品必须以当前分支的分区和权限策略为准。

## 构建它

`ModuleSpec` 验证名称、分区、证书和特权标记；`installation_path()` 输出预期的 app/priv-app 路径。

```bash
cd phases/22-android-framework-system-basics/04-android-build-and-partition/code
python3 main.py
python3 -m unittest discover tests -v
```

## 诊断练习

出现 artifact path requirement 错误时，先核对模块所属分区和产品配置；不要直接把路径加入允许列表来掩盖归属错误。

## 发布它

模块分区审查卡见 `outputs/skill-build-partition-review.md`。
