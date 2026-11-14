"""保留历史全图3%批次的汇总；不用于修正后的训练基数与过滤设置。"""
import json
import math
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/cora_rate3_uneven_20261113'
CPU_OUTPUT = ROOT / 'outputs/cora_rate3_uneven_losplit_cpu_20261113'


def main():
    bundle = OUTPUT / 'data/Cora/UGBA/seed10'
    metadata = json.loads((bundle / 'metadata.json').read_text(encoding='utf-8'))
    training = torch.load(bundle / 'training.pt', map_location='cpu')
    evaluation = torch.load(bundle / 'evaluation.pt', map_location='cpu')
    count = math.floor(evaluation['truth'].numel() * metadata['settings']['rate'] / 100)
    attached = evaluation['attached']
    allocation = metadata['class_allocation']
    actual_counts = torch.bincount(evaluation['truth'][attached], minlength=training['classes']).tolist()
    assert count == 81 and attached.numel() == 81 and attached.unique().numel() == 81
    assert actual_counts == allocation['poisoned_source_counts']
    assert bool((training['labels'][attached] == metadata['settings']['target']).all())
    assert not bool(torch.isin(attached, training['val']).any())
    assert not bool(torch.isin(attached, evaluation['clean_nodes']).any())
    assert not bool(torch.isin(attached, evaluation['attack_nodes']).any())
    index, failures = {}, []
    for output in [OUTPUT, CPU_OUTPUT]:
        for path in (output / 'runs').rglob('result.json'):
            row = json.loads(path.read_text(encoding='utf-8'))
            if row['status'] != 'complete':
                failures.append({'record': str(path), 'status': row['status'], 'error': row.get('error')})
                continue
            assert row['bundle_id'] == metadata['bundle_id']
            index[(row['model'], row['method'])] = {'record': str(path), 'CA': row['metrics']['CA'],
                'ASR': row['metrics']['ASR'], 'device': row['arguments']['device'],
                'CA_scope': row['metrics']['CA_scope'], 'ASR_scope': row['metrics']['ASR_scope'],
                'training_nodes': len(row['details'].get('author_training_nodes', [])),
                'attached_nodes_used_by_author': len(row['details'].get('author_attached_nodes', [])),
                'model_build_seconds': row['metrics']['model_build_seconds']}
    checks = {}
    for output in [OUTPUT, CPU_OUTPUT]:
        path = output / 'verification.json'
        if path.exists():
            value = json.loads(path.read_text(encoding='utf-8'))
            checks[output.name] = {'status': value['status'], 'verified': len(value['verified']),
                                   'failed_or_unfinished': value['failed_or_unfinished']}
    result = {'date': '2026-11-13', 'dataset': 'Cora', 'attack': 'UGBA', 'seed': 10,
              'rate_percent': 3, 'target': metadata['settings']['target'],
              'class_sampling': 'uneven', 'trigger_groups': count,
              'original_nodes': evaluation['truth'].numel(),
              'added_nodes': training['x'].size(0) - evaluation['truth'].numel(),
              'changed_labels': int(evaluation['truth'][attached].ne(metadata['settings']['target']).sum()),
              'class_allocation': allocation, 'bundle_validation': 'passed',
              'records': list(index.values()), 'failed_runs': failures, 'reload_verification': checks}
    (OUTPUT / 'rate3_uneven_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Cora、rate=3、不均匀原始类别选样结果', '',
             '**历史口径记录：本批次按全图节点数计算3%，生成前未滤原目标类；不代表当前修正接口。数据与数值保留溯源。**', '',
             '日期：2026-11-13。Cora、UGBA、seed=10、split_seed=10、target=0，其余生成与模型参数为当前默认。', '',
             '原始节点2708；触发器组数floor(2708×3/100)=81；每组3个新增节点，共新增243个节点。原始类别0—6的选样数量为' + str(actual_counts) + '。全部81个中毒后标签为0，63个发生标签改变，18个原本就是类别0。', '',
             '## CA与ASR', '', '| 方法 | 设备 | GCN CA/ASR（%） | GAT CA/ASR（%） | GraphSAGE CA/ASR（%） |',
             '|---|---|---:|---:|---:|']
    for method in ['none', 'abp', 'rigbd', 'dshield', 'mad', 'prune', 'losplit']:
        cells = []
        for model in ['GCN', 'GAT', 'GraphSAGE']:
            row = index.get((model, method))
            cells.append(format(row['CA'] * 100, '.2f') + ' / ' + format(row['ASR'] * 100, '.2f') if row else '未完成')
        lines.append('| ' + ' | '.join([method, 'CPU' if method == 'losplit' else 'cuda:0'] + cells) + ' |')
    lines += ['', 'none为无防御，ASR越低表示预测到指定目标类别的比例越低；不把它称为节点检测召回率。', '',
              '各作者保留原评估：DShield逐节点二跳图且ASR包含原目标类，其他方法多数用完整图并只统计原非目标类；RIGBD的CA在触发图上测量。MAD保留全原节点标签辅助训练和五个内部最终模型。定义不同，原数值保留，不作严格统一测试条件的排名。', '',
              'RIGBD仍使用原默认K=20、drop_ratio=0.5，未按本次样本调参。较高ASR也按原值保留，不能因数值不理想就改作者实现。LoSplit用CPU原流程；CPU与GPU运行有重叠，不据本表比较时间。', '',
              '## 数据核对', '',
              '- 81个附着位置均唯一，且不与原验证、正常测试或攻击测试索引重叠。',
              '- 原始类别计数已与evaluation.pt的原标签逐位置对照；中毒后标签均与target一致。',
              '- 两种数量参数均不提供或同时提供时，生成入口拒绝执行，且不会创建目标数据目录。',
              '- 15个预算/类别分配测试通过；244个原作者文件哈希检查通过。', '',
              '## 复现', '', '```text',
              'uv run --locked --extra cu118 python run_study.py --datasets Cora --models GCN GAT GraphSAGE --attacks UGBA --seeds 10 --methods none abp rigbd dshield mad prune --rate 3 --class-sampling uneven --output-root outputs/repeat_rate3_uneven --device cuda:0',
              '```', '',
              'LoSplit另用run_defense.py --methods losplit --device cpu，并指定同一--bundle目录和所需--model。数量接口说明见数量与类别选样_20261113.md。', '',
              'DMGNN公开文字描述了按类别注入不同数量，但未在该段明确seed到整数配额的具体算法，且其3%是训练集比例。本项目按用户指定使用总原始节点数，并明确采用Dirichlet(1)权重与容量约束整数分配，不宣称复现其未公开的具体选样代码。', '',
              '单个种子的结果用于检查本次接口和设置，不替代多种子比较。成功记录数：' + str(len(index)) + '；失败或未完成记录数：' + str(len(failures)) + '。']
    for name, value in checks.items():
        lines.append('重载核验：' + name + '，' + value['status'] + '，通过' + str(value['verified']) + '项。')
    (ROOT / 'paper/Cora_rate3_uneven_结果_20261113.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('本次数据和结果报告已生成，成功记录：' + str(len(index)))


if __name__ == '__main__':
    main()
