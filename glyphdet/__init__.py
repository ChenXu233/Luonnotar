"""glyphdet —— 掩码条件化字形检测器（路线 A）。

子包：
- glyphdet.core   ：稳定引擎（synth / dataset / model / train / eval / export_onnx）
- glyphdet.tools  ：一次性诊断/可视化脚本

入口统一以模块形式运行（IDE/运行时都满意）：
    python -m glyphdet.core.train  --config experiments/.../config.yaml
    python -m glyphdet.core.eval   --config ... --weights ...
    python -m glyphdet.tools.fppeek --config ... --weights ...

也可使用 pdm scripts 别名（见 pyproject.toml [tool.pdm.scripts]）：
    pdm run train   /  pdm run eval   /  pdm run synth  /  pdm run export
    pdm run fppeek  /  pdm run failpeek  /  pdm run peek
"""
__version__ = "0.1.0"
