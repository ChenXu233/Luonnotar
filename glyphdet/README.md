# glyphdet — 掩码条件化字形检测器（路线 A）

输入 = 目标码整串的标准 mask 渲染图（雅黑，不拆字符）+ 场景图 → 输出 = 场景中所有相似字形区域的 box。
**检测任务，非识别任务**：模型学字形形变不变性，永不"读"字。设计依据见 `../docs/AI-report/paddle-ocr-postmortem.md` 与 `docs/design.md`。

## 目录

- `glyphdet/core/` 稳定引擎（synth / dataset / model / train / eval / export_onnx），改动只增不破
- `experiments/<阶段>/config.yaml` 冻结超参；目录名是稳定标识符，禁止重命名
- `datasets/` 合成产物，git 不跟踪，`glyphdet/core/synth.py` 可复现重建
- `runs/<name>/` 训练产物，含 config 快照与 `code_version.txt`（git commit + dirty）
- `glyphdet/tools/` 一次性脚本（抽检可视化等）

## 包结构

`glyphdet/` 本身是 Python 顶级包（含 `__init__.py`），子包 `glyphdet.core` 与 `glyphdet.tools` 也是包。
入口统一以模块形式运行（IDE 与运行时都满意）：

```bash
cd glyphdet                 # pdm 项目根
python -m glyphdet.core.train --config experiments/.../config.yaml
python -m glyphdet.core.eval  --config ... --weights runs/<name>/last.pt
python -m glyphdet.tools.fppeek --config ... --weights runs/<name>/last.pt
```

也可使用 pdm 别名（`pyproject.toml [tool.pdm.scripts]` 已定义）：

```bash
pdm run train / eval / synth / export
pdm run fppeek / failpeek / peek
```

> 历史命令 `python core/train.py` 不再可用：脚本方式会绕过 PYTHONPATH，
> 导致 `from glyphdet.core.dataset import ...` 解析不到。统一用 `python -m`。

## 环境

- pdm 管理，虚拟环境在项目 `.venv/`。命令统一在 `glyphdet/` 下执行
- torch cu128（RTX 4070 Laptop 8GB）；**`.pdm.toml` 的 `use_uv=false` 别动**（uv 会把 torch 装成 CPU 版）
- 训练约束：用户电脑日常他用，batch ≤16、workers 4、显存目标 ≤3G

## 复现

```
pdm install           # 安装依赖（首次）
pdm run setup         # 一次性生成 .pth，让 import glyphdet 在 cwd=glyphdet 下可用
pdm run synth   --config experiments/mvp_baseline/config.yaml
pdm run train   --config experiments/mvp_baseline/config.yaml
pdm run eval    --config experiments/mvp_baseline/config.yaml
```

> `pdm run setup` 只在首次 clone / 重建 venv 后需要跑一次。背后写一个
> `glyphdet_dev.pth` 到 `.venv/Lib/site-packages/`，让 `import glyphdet` 解析到
> `glyphdet/__init__.py`（PDM 在 Windows + `cd glyphdet` 工作流下不会自动注入 sys.path）。

## IDE 提示

VSCode Pylance 解析 `from glyphdet.core.xxx import ...` 需把工作区根（`Luonnotar/`）
加入 `python.analysis.extraPaths`——本仓库已在 `.vscode/settings.json` 中配好，
IDE 与运行时使用同一套 import 路径，无误导。

## 已知坑（沿用 AI4gem 教训）

- cv2 读中文路径会挂：用 `cv2.imdecode(np.fromfile(path, dtype=np.uint8), ...)`
- Windows 显存溢出 = 静默大幅减速（WDDM 溢出到共享内存），step 时间异常先降 batch
- 杀后台任务后 python 可能残留成孤儿，重跑训练前 `tasklist | grep -i python` 确认清零
