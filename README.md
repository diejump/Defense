# ABP节点分类实验项目

当前版本：2026-11-11，`author_original_20261111`。

2026-11-13修正：生成时必须指定`--rate`或`--vs_number`之一；`--rate 3`按注入前原训练节点数的3%计算选样预算。默认`--class-sampling random`；random和uneven都先选样，再过滤原标签等于target的位置，不补选，之后才训练生成器。预算与实际组数分别保存。见[数量及类别选样说明](paper/数量与类别选样_20261113.md)。

方法包括无防御、ABP、RIGBD、LoSplit、DShield、MAD和Prune。RIGBD、LoSplit、DShield、MAD直接调用固定版本作者模型和原生训练函数；作者入口的防御与预测片段保存在`src/author_methods`。Prune直接调用LoSplit的`prune_unrelated_edge`。RS、ABL已从本项目比较入口移除。

## 阅读顺序

- [运行命令和参数](paper/运行命令.md)：手动修改默认参数、执行顺序及服务器命令。
- [作者实现接入记录](paper/作者实现接入_20261111.md)：核查发现、原实现的运行条件及本轮Cora验证。
- [代码结构与实验协议](paper/代码结构与实验协议.md)：源码对应关系、标签条件、计时和指标。
- [RIGBD参数依据](paper/RIGBD_参数依据_20261111.md)：作者固定设置、公开候选网格及尚未公开的信息。
- [研究方案](paper/研究方案.md)：ABP问题、规则和理论依据。

## 环境

```text
uv sync --locked --extra cu118
```

CPU环境将`cu118`改为`cpu`，两个extra选一个。依赖由`pyproject.toml`和`uv.lock`管理。原源码导入需要的`deeprobust`和`scikit-learn-extra`已写入依赖。

## Cora上运行

```text
uv run --locked --extra cu118 python run_attack.py --dataset Cora --attack UGBA --rate 3 --output-root outputs/cora_train_rate3_random
uv run --locked --extra cu118 python run_defense.py --dataset Cora --attack UGBA --model GCN --methods none abp rigbd dshield mad prune --output-root outputs/cora_train_rate3_random --device cuda:0
uv run --locked --extra cu118 python run_defense.py --dataset Cora --attack UGBA --model GCN --methods losplit --output-root outputs/cora_train_rate3_losplit_cpu --bundle outputs/cora_train_rate3_random/data/Cora/UGBA/seed10 --device cpu
```

LoSplit固定版本在划分节点时直接把GPU标签转成NumPy，会报错；当前使用作者原代码的CPU流程。跨CPU/GPU的耗时不用于方法速度比较。MAD保留用户已确认的全原节点标签辅助训练条件，原入口的五次最终模型训练也保留；这些条件写入结果。

## 保存和复核

```text
uv run --locked --extra cu118 python check_project.py
uv run --locked --extra cu118 python verify_results.py --output-root outputs/cora_train_rate3_random
uv run --locked --extra cu118 python summarize.py --output-root outputs/cora_train_rate3_random
```

`result.json`记录实际作者参数、原指标定义、实际监督节点和内部模型数量；`model.pt`保存原生模型权重及评估模式、随机状态。每个作者独立进程运行，避免同名`models`包相互覆盖。改变参数时使用新输出目录；参数一致的已完成运行可加`--resume`。

作者副本保存在`upstream`；逐文件来源见`paper/author_source_manifest.json`。旧实现和旧说明已归档至`paper/archive`，历史数据及结果仍保存在原目录。
