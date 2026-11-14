"""单个作者的独立入口，由主脚本自动调用，避免同名models包混用。"""
import json
import sys
from pathlib import Path
from src.author_execution import execute, reevaluate


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if request["operation"] == "evaluate":
        reevaluate(request)
    else:
        execute(request)


if __name__ == "__main__":
    main()
