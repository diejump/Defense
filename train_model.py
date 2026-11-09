"""单独在正常图或已保存污染图上训练普通分类模型。"""
import json
import torch
from src.config import parser
from src.common import setup, save_json, file_hash
from src.data import load_clean
from src.attack import bundle_directory, to_cpu
from src.workflow import run_method


def main():
    p = parser("训练普通分类模型")
    p.add_argument("--data", choices=["clean", "backdoor"], default="clean", help="正常图或已有污染图")
    args = p.parse_args()
    if args.data == "backdoor":
        folder = bundle_directory(args)
    else:
        args.attack = None
        folder = args.output_root / "clean_data" / args.dataset / ("split" + str(args.split_seed))
        if not (folder / "metadata.json").exists():
            device = setup(args.seed, args.device)
            data, info = load_clean(args, device)
            folder.mkdir(parents=True, exist_ok=False)
            torch.save(to_cpu(data), folder / "training.pt")
            edges = torch.cat([data["edges"], info["heldout"]], dim=1)
            evaluation = {"truth": info["truth"], "clean_nodes": info["clean_test"],
                          "attack_nodes": torch.empty(0, dtype=torch.long, device=device), "target": None,
                          "reference_x": data["x"], "reference_edges": edges,
                          "reference_weights": data["x"].new_ones(edges.size(1))}
            torch.save(to_cpu(evaluation), folder / "evaluation.pt")
            save_json(folder / "metadata.json", {"settings": {"dataset": args.dataset, "attack": None},
                "bundle_id": file_hash(folder / "training.pt") + ":" + file_hash(folder / "evaluation.pt"), "source": info["source"]})
    output = args.output_root / "ordinary_models" / args.dataset / str(args.attack) / ("seed" + str(args.seed)) / args.model
    run_method(folder, output, "none", args)


if __name__ == "__main__":
    main()
