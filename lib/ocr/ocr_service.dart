import 'dart:io';

import 'package:fast_paddle_ocr/ocr.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

/// 封装 fast_paddle_ocr：模型初始化、相机开关、结果轮询。
class OcrService {
  final Ocr _ocr = Ocr();

  bool _modelLoaded = false;
  bool get modelLoaded => _modelLoaded;

  static const _modelFiles = [
    'PP_OCRv5_mobile_det.ncnn.param',
    'PP_OCRv5_mobile_det.ncnn.bin',
    'PP_OCRv5_mobile_rec.ncnn.param',
    'PP_OCRv5_mobile_rec.ncnn.bin',
  ];

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

  /// 拷贝模型文件、加载模型、设置取件码字符白名单。
  Future<void> loadModel() async {
    final paths = <String, String>{};
    for (final name in _modelFiles) {
      paths[name] = await _copyAssetToFile(name);
    }
    await _ocr.loadModel(
      detParam: paths['PP_OCRv5_mobile_det.ncnn.param']!,
      detModel: paths['PP_OCRv5_mobile_det.ncnn.bin']!,
      recParam: paths['PP_OCRv5_mobile_rec.ncnn.param']!,
      recModel: paths['PP_OCRv5_mobile_rec.ncnn.bin']!,
      sizeid: 0, // 320，预览分辨率下足够识别货架标签
      // CPU：实测 GPU(Vulkan) 对这套 mobile 小模型是净亏损——
      // rec 170ms/框（CPU 40ms），det 28ms（CPU 33ms）打平，
      // 且每次冷启动着色器编译 ~19s。小模型固定派发开销 > 计算本身。
      cpugpu: 0,
    );
    // 取件码只含数字与连字符：白名单既解决连字符被吞，也提升准确率
    await _ocr.setCharFilter('0123456789-');
    _modelLoaded = true;
  }

  /// 打开相机。facing: 0=前置 1=后置（原生层约定），手机端默认后置。
  /// 前置帧已在原生层做水平翻转，OCR 看到的文字不是镜像。
  Future<bool> openCamera([int facing = 1]) => _ocr.openCamera(facing);

  Future<bool> closeCamera() => _ocr.closeCamera();

  Future<bool> toggleFlash() => _ocr.toggleFlash();

  /// 拉取最新一帧的识别结果（文本 + 旋转框，帧坐标系）。
  Future<OcrFrame> pollResults() => _ocr.getOcrResults();
}
