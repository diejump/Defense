"""将已下载的Git树与唯一作者副本逐文件比对，生成便携来源清单。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources, files = {}, []
    for method in ["RIGBD", "LoSplit", "DShield", "MAD"]:
        name = "mad" if method == "MAD" else method
        tree_path = ROOT / "outputs/author_source_20261111" / (name + "_tree.json")
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        sources[method] = {"commit": tree["sha"], "git_tree_url": tree["url"]}
        index = {}
        for entry in tree["tree"]:
            if entry["type"] == "blob":
                index[entry["path"]] = entry["sha"]
        folder = ROOT / "upstream" / method
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or ".git" in path.parts:
                continue
            relative = path.relative_to(folder).as_posix()
            data = path.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if relative not in index or blob != index[relative]:
                raise ValueError("作者源码与固定Git树不一致：" + str(path))
            files.append({"file": path.relative_to(ROOT).as_posix(), "git_blob_sha1": blob,
                          "sha256": hashlib.sha256(data).hexdigest()})
    result = {"date": "2026-11-11", "sources": sources, "files": files}
    target = ROOT / "paper/author_source_manifest.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("唯一作者副本通过Git树核对，文件数：" + str(len(files)))


if __name__ == "__main__":
    main()
