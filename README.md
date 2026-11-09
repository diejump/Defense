# ABP独立实验项目

研究半监督节点分类中的图后门防御。代码包含ABP、RIGBD、LoSplit、DShield及无防御对照。

公共流程为：准备公开数据 → 生成固定样本 → 训练所选方法 → 保存模型 → 评估CA、ASR与耗时。每个阶段显示中文进度条。

- 阅读公共流程：run_study.py、src/workflow.py。
- 阅读骨架选择：src/models.py。
- 阅读ABP规则：src/abp.py。
- [代码与实验协议](paper/代码结构与实验协议.md)
- [运行命令及参数](paper/运行命令.md)
- [研究方案](paper/研究方案.md)
- [本地比较结果](paper/本地比较结果_20261109.md)

## 环境

```text
uv sync --locked --extra cu118
```

无GPU时使用cpu extra。两种extra选一个。Python固定3.10，依赖版本由pyproject.toml与uv.lock管理。

## 执行本地对应组合

```text
uv run --locked --extra cu118 python prepare_data.py --datasets Cora Citeseer
uv run --locked --extra cu118 python run_study.py --datasets Cora Citeseer --models GCN GAT GraphSAGE --attacks UGBA --seeds 10 --output-root outputs/my_run
```

这会生成两份固定样本，完成六个数据／骨架组合、每组合五种方法。各方法共用样本。重新启动相同命令时追加--resume继续已完成的运行；改变数学参数时选择新的输出目录。

## 骨架关系

| 方法 | 评分／准备模型 | 最终模型 |
|---|---|---|
| RIGBD | 所选GCN／GAT／GraphSAGE | 相同骨架 |
| LoSplit | 固定GCN | 所选GCN／GAT／GraphSAGE |
| DShield | 对应作者分类模型与内部arch适配 | 对应作者分类模型 |
| ABP | 先处理连接 | 所选普通GCN／GAT／GraphSAGE |

三个作者源码分别保存在upstream的同名目录；公共调度、样本保存与评估重新编写。训练与评估文件分开保存，所有方法使用同一份原始测试真值。
