"""导出 ONNX + 数值对拍（go/no-go ⓪）。

  pdm run python core/export_onnx.py --config ... --weights runs/mvp_baseline/last.pt

导出图 = 裸 head 输出（decode 在部署侧）。runtime 保持开放：
ONNX（onnxruntime-mobile / NNAPI / XNNPACK）或后续 onnx2ncnn / pnnx 转 ncnn。
对拍：onnxruntime 输出 vs PyTorch 输出，max abs diff 必须 < 1e-3。
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

from glyphdet.core.model import GlyphDet, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default=None, help="缺省用随机权重（链路验证模式）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    model = build_model(cfg).eval()
    if args.weights:
        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
    if hasattr(model, "reparam"):
        model.reparam()  # v2: RepVGG 训练分支融合回单 3×3 卷积（导出/部署形态）
    out_path = args.out or str(Path(args.weights or ".").with_name("glyphdet.onnx"))

    scene = torch.rand(1, 3, cfg["model"]["in_size"], cfg["model"]["in_size"])
    mask = torch.rand(1, 1, *cfg["model"]["mask_size"])
    strides = cfg["model"]["strides"]
    torch.onnx.export(
        model, (scene, mask), out_path,
        input_names=["scene", "mask"],
        output_names=[f"out{s}" for s in strides],
        opset_version=18,
    )
    size_mb = Path(out_path).stat().st_size / 1e6

    with torch.no_grad():
        ref = [o.numpy() for o in model(scene, mask)]
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"scene": scene.numpy(), "mask": mask.numpy()})
    diffs = [float(np.abs(r - g).max()) for r, g in zip(ref, got)]
    ok = all(d < 1e-3 for d in diffs)
    print(f"导出: {out_path}（{size_mb:.1f}MB fp32）")
    for s, d in zip(strides, diffs):
        print(f"  out{s} max|diff| = {d:.2e}")
    print("对拍", "通过" if ok else "失败（>1e-3，需检查算子）")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
