"""用小图检查三种生成器的训练、保存图和测试触发接口。"""
import torch
from tqdm.auto import tqdm
from src.config import parser, ROOT
from src.common import setup, save_json
from src.attack import make_generator, labels_for_generator, keep_positive_edges


def main():
    args = parser("生成接口检查").parse_args([])
    args.attack_epochs = 2
    args.attack_inner = 1
    args.rec_epochs = 2
    args.detector_updates = 1
    args.outer_size = 32
    reports = []
    for name in tqdm(["GTA", "UGBA", "DPGBA"], desc="检查三种生成接口", unit="种"):
        args.attack = name
        device = setup(10, "cpu")
        x = torch.rand(512, 16)
        x = x / x.sum(1, keepdim=True)
        nodes = torch.arange(512)
        edges = torch.stack([nodes, (nodes + 1) % 512])
        edges = torch.cat([edges, edges.flip(0)], dim=1)
        train = torch.arange(128)
        attached = torch.tensor([128, 129, 130])
        available = torch.arange(131, 512)
        data = {"labels": nodes % 2, "train": train, "classes": 2}
        labels = labels_for_generator(data, 0)
        generator, parameters = make_generator(args, device, available.numel())
        generator.fit(x, edges, torch.ones(edges.size(1)), labels.clone(), train, attached, available)
        poisoned_x, poisoned_edges, weights, poisoned_labels = generator.get_poisoned(
            x, edges, torch.ones(edges.size(1)), labels.clone(), attached)
        poisoned_x, poisoned_edges, weights = keep_positive_edges((poisoned_x, poisoned_edges, weights))
        changed = generator.inject_trigger(torch.tensor([200, 201]), poisoned_x, poisoned_edges, weights)
        changed_x, changed_edges, changed_weights = keep_positive_edges(changed)
        assert poisoned_x.size(0) == 521
        assert changed_x.size(0) == 527
        assert poisoned_labels.numel() == 512
        assert torch.isfinite(changed_x).all()
        assert torch.all(changed_weights > 0)
        reports.append({"attack": name, "status": "passed", "input_nodes": 512,
            "features": 16, "training_nodes": 128, "training_attached": 3,
            "poisoned_nodes": poisoned_x.size(0), "evaluation_attached": 2,
            "changed_nodes": changed_x.size(0), "actual_parameters": parameters})
    save_json(ROOT / "outputs/attack_interface_checks.json", {
        "status": "passed", "purpose": "算法接口检查，使用明确的小图用例，不计算或报告科研效果指标", "checks": reports})
    print("三种生成器接口全部通过。")


if __name__ == "__main__":
    main()
