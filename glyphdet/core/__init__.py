"""glyphdet.core —— 稳定引擎层。

子模块（改动只增不破，向后兼容）：
- synth        ：全合成训练数据生成器
- dataset      ：PyTorch Dataset + FCOS 式正负分配
- model        ：掩码条件化字形检测网络（FCOS-style 3 级 head）
- decode       ：裸 head 输出 → 框（DFL + sigmoid + NMS）
- train        ：训练循环（focal + CIoU/DFL + centerness）
- eval         ：检出率 / 误检率 / 一位之差可分性 评估
- export_onnx  ：ONNX 导出 + 数值对拍
"""
