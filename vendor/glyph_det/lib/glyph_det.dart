import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'glyph_det_result.dart';

export 'glyph_det_result.dart';

/// Mask-conditioned glyph detector: given a rendered query mask of a pickup
/// code, finds the matching label in the live camera frame.
///
/// Typical usage:
/// ```dart
/// final det = GlyphDet();
/// await det.loadModel(paramPath, binPath);        // ncnn param+bin on disk
/// await det.setQueryMask(maskGrayBytes);          // 64x384 white-bg black-text
/// await det.openCamera(1);                        // 1=back, 0=front
/// final frame = await det.pollResults();          // top box in frame.topBox
/// ```
class GlyphDet {
  /// Query mask width/height in pixels (model contract: (1,1,64,384)).
  static const int maskWidth = 384;
  static const int maskHeight = 64;
  static const int maskByteCount = maskWidth * maskHeight;

  @visibleForTesting
  static const MethodChannel methodChannel = MethodChannel('glyph_det');

  Future<String?> getPlatformVersion() async {
    return methodChannel.invokeMethod<String>('getPlatformVersion');
  }

  /// Load the ncnn model from file paths (param + bin).
  ///
  /// [cpugpu] - 1 = Vulkan GPU (default, first load compiles shaders and may
  /// take several seconds; runs on a background thread natively), 0 = CPU.
  Future<bool> loadModel(
    String paramPath,
    String binPath, {
    int cpugpu = 1,
  }) async {
    final result = await methodChannel.invokeMethod<bool>('loadModel', {
      'paramPath': paramPath,
      'binPath': binPath,
      'cpugpu': cpugpu,
    });
    return result ?? false;
  }

  /// Set the query mask: [maskByteCount] grayscale bytes (row-major,
  /// [maskHeight] x [maskWidth]), white background (255) with black text (0),
  /// as rendered by the app (see example/lib/mask_render.dart).
  Future<bool> setQueryMask(Uint8List gray) async {
    if (gray.length != maskByteCount) {
      throw ArgumentError.value(
        gray.length,
        'gray.length',
        'mask must be $maskHeight x $maskWidth = $maskByteCount bytes',
      );
    }
    final result = await methodChannel.invokeMethod<bool>('setQueryMask', {
      'mask': gray,
    });
    return result ?? false;
  }

  /// Open the camera. [facing]: 0 = front, 1 = back (native convention).
  Future<bool> openCamera(int facing) async {
    final result = await methodChannel.invokeMethod<bool>('openCamera', {
      'facing': facing,
    });
    return result ?? false;
  }

  Future<bool> closeCamera() async {
    final result = await methodChannel.invokeMethod<bool>('closeCamera');
    return result ?? false;
  }

  /// Toggle camera flash/torch on/off.
  Future<bool> toggleFlash() async {
    final result = await methodChannel.invokeMethod<bool>('toggleFlash');
    return result ?? false;
  }

  /// Pull the latest detection snapshot: camera frame size, pipeline stats
  /// and up to 20 boxes (score > 0.5, NMS IoU 0.4) sorted by score
  /// descending, in camera frame coordinates.
  Future<GlyphDetFrame> pollResults() async {
    final json = await methodChannel.invokeMethod<String>('pollResults');
    return GlyphDetFrame.parse(json);
  }
}
