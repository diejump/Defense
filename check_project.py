"""核对项目语法、锁定作者源码与提取片段；不训练模型。"""
import ast
import hashlib
import json
from pathlib import Path
from tqdm.auto import tqdm
from src.common import save_json, file_hash

ROOT = Path(__file__).resolve().parent


def check_sources():
    registry = json.loads((ROOT / "paper/author_source_manifest.json").read_text(encoding="utf-8"))
    for entry in tqdm(registry["files"], desc="核对原始源码", unit="文件"):
        path = ROOT / entry["file"]
        content = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if blob != entry["git_blob_sha1"]:
            raise ValueError("原作者文件发生变化：" + str(path))
    return len(registry["files"])


def check_adapters():
    first = json.loads((ROOT / "paper/rigbd_losplit_segments_20261111.json").read_text(encoding="utf-8"))
    for name, entry in first["files"].items():
        if file_hash(ROOT / name) != entry["generated_sha256"]:
            raise ValueError("作者提取片段发生变化：" + name)
        if file_hash(ROOT / entry["source"]) != entry["source_sha256"]:
            raise ValueError("提取来源变化：" + entry["source"])
    second = json.loads((ROOT / "paper/mad_dshield_segments_20261111.json").read_text(encoding="utf-8"))
    for entry in second["records"]:
        if file_hash(ROOT / entry["adapter"]) != entry["adapter_sha256"]:
            raise ValueError("作者提取片段发生变化：" + entry["adapter"])
        if file_hash(ROOT / entry["source"]) != entry["source_sha256"]:
            raise ValueError("提取来源变化：" + entry["source"])
    return len(first["files"]) + len(second["records"])


def check_prune_equivalence():
    # DPGBA调用根目录原剪边函数；两份原函数必须保持相同。
    sources = []
    for path in ["upstream/LoSplit/help_funcs.py", "upstream/LoSplit/DPGBA/help_funcs.py"]:
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
        functions = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in ["prune_unrelated_edge", "prune_unrelated_edge_isolated"]:
                functions[node.name] = ast.dump(node, include_attributes=False)
        sources.append(functions)
    if sources[0] != sources[1] or len(sources[0]) != 2:
        raise ValueError("LoSplit两分支原剪边函数不同，需重新检查导入来源。")
    return True


def main():
    paths = list(ROOT.glob("*.py")) + list((ROOT / "src").rglob("*.py")) + list((ROOT / "tools").glob("*.py"))
    for path in tqdm(paths, desc="核对项目语法", unit="文件"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    report = {"status": "passed", "syntax_files": len(paths), "source_files": check_sources(),
              "extracted_adapters": check_adapters(), "prune_functions_ast_identical": check_prune_equivalence(),
              "training_performed": False}
    save_json(ROOT / "outputs/source_verification_20261111.json", report)
    print("语法、原始源码和作者调用片段核对通过。")


if __name__ == "__main__":
    main()
