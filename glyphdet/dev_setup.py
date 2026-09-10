"""一次性开发环境配置脚本（`pdm run setup` 调用）。

PDM 在 Windows + `cd glyphdet` 工作流下，不会自动把项目根的父目录加进 sys.path。
本脚本生成一个稳定的 .pth 文件（绝对路径）到 .venv/Lib/site-packages/，
使 `import glyphdet` 在任何 cwd 下都能解析到 glyphdet/__init__.py。
"""
import os
import sys
from pathlib import Path

# venv 的 site-packages 目录 = python.exe 上 1 级的 Lib/site-packages
sp = Path(sys.executable).parent.parent / "Lib" / "site-packages"
# 顶级包 glyphdet 的父目录 = glyphdet/ 的父目录 = Luonnotar/
dst = (Path.cwd().parent).resolve()
# 写 .pth（绝对路径，PDM 不会自动覆盖）
pth = sp / "glyphdet_dev.pth"
pth.write_text(str(dst) + os.sep, encoding="utf-8")
print(f"{pth.name} -> {dst}")
