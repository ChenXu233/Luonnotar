import 'dart:io';

import 'package:flutter/services.dart';
import 'package:glyph_det/glyph_det.dart';
import 'package:path_provider/path_provider.dart';

import 'mask_render.dart';

/// 封装 glyph_det：模型初始化、查询 mask 设置、相机开关、结果轮询。
///
/// 与 OCR 路线的语义差异：不再"识别画面里所有文字再做文本匹配"，
/// 而是"给定取件码渲染成 mask，模型直接在画面里定位这串字"，
/// 因此目标变更时必须调用 [setTargetCode] 重新下发 mask。
class GlyphDetService {
  final GlyphDet _det = GlyphDet();

  bool _modelLoaded = false;
  bool get modelLoaded => _modelLoaded;

  static const _modelFiles = ['glyphdet.param', 'glyphdet.bin'];

  Future<String> _copyAssetToFile(String name) async {
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/models/$name');
    if (!await file.exists()) {
      await file.parent.create(recursive: true);
      final data = await rootBundle.load('assets/models/$name');
      await file.writeAsBytes(data.buffer.asUint8List());
    }
    return file.path;
  }

  /// 拷贝模型文件并加载。cpugpu: 1=Vulkan（默认，首次加载要编译着色器，
  /// 可能耗时数秒），0=CPU。2.4M 参数的小模型是否吃得起 Vulkan
  /// 派发开销，待真机实测后再定默认值。
  Future<void> loadModel({int cpugpu = 1}) async {
    final param = await _copyAssetToFile(_modelFiles[0]);
    final bin = await _copyAssetToFile(_modelFiles[1]);
    final ok = await _det.loadModel(param, bin, cpugpu: cpugpu);
    if (!ok) {
      throw StateError('glyph_det loadModel failed');
    }
    _modelLoaded = true;
  }

  /// 把取件码渲染成 64x384 查询 mask 并下发给原生侧。
  Future<bool> setTargetCode(String code) async {
    final gray = await renderQueryMask(code);
    return _det.setQueryMask(gray);
  }

  /// 打开相机。facing: 0=前置 1=后置（原生层约定），手机端默认后置。
  Future<bool> openCamera([int facing = 1]) => _det.openCamera(facing);

  Future<bool> closeCamera() => _det.closeCamera();

  Future<bool> toggleFlash() => _det.toggleFlash();

  /// 拉取最新一帧的检测快照（候选框 + 管线统计，帧坐标系）。
  Future<GlyphDetFrame> pollResults() => _det.pollResults();
}
