# glyphdet — 掩码条件化字形检测器（路线 A）

输入 = 目标码整串的标准 mask 渲染图（雅黑，不拆字符）+ 场景图 → 输出 = 场景中所有相似字形区域的 box。
**检测任务，非识别任务**：模型学字形形变不变性，永不"读"字。设计依据见 `../docs/AI-report/paddle-ocr-postmortem.md` 与 `docs/design.md`。

## 目录

- `core/` 稳定引擎（synth / dataset / model / train / eval / export_onnx），改动只增不破
- `experiments/<阶段>/config.yaml` 冻结超参；目录名是稳定标识符，禁止重命名
- `datasets/` 合成产物，git 不跟踪，`core/synth.py` 可复现重建
- `runs/<name>/` 训练产物，含 config 快照与 `code_version.txt`（git commit + dirty）
- `tools/` 一次性脚本（抽检可视化等）

## 环境

- pdm 管理，虚拟环境在项目 `.venv/`。命令统一在 `glyphdet/` 下 `pdm run python ...`
- torch cu128（RTX 4070 Laptop 8GB）；**`.pdm.toml` 的 `use_uv=false` 别动**（uv 会把 torch 装成 CPU 版）
- 训练约束：用户电脑日常他用，batch ≤16、workers 4、显存目标 ≤3G

## 复现

```
pdm install
pdm run python core/synth.py  --config experiments/mvp_baseline/config.yaml
pdm run python core/train.py  --config experiments/mvp_baseline/config.yaml
pdm run python core/eval.py   --config experiments/mvp_baseline/config.yaml
```

## 已知坑（沿用 AI4gem 教训）

- cv2 读中文路径会挂：用 `cv2.imdecode(np.fromfile(path, dtype=np.uint8), ...)`
- Windows 显存溢出 = 静默大幅减速（WDDM 溢出到共享内存），step 时间异常先降 batch
- 杀后台任务后 python 可能残留成孤儿，重跑训练前 `tasklist | grep -i python` 确认清零
