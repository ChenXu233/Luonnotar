"""glyphdet.tools —— 一次性诊断/可视化脚本。

子模块：
- peek     ：合成数据抽检可视化（mask + 场景带框）
- failpeek ：漏检案例检视（val 集中漏检样本）
- fppeek   ：误检案例检视（fp = 预测框与所有 target GT IoU<0.3；按分数降序出图）
"""
