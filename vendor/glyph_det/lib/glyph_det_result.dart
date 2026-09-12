import 'dart:convert';

/// A single detection: axis-aligned box of the queried pickup code in the
/// camera frame coordinate space (see [GlyphDetFrame.width] /
/// [GlyphDetFrame.height]).
class GlyphDetBox {
  const GlyphDetBox({
    required this.score,
    required this.cx,
    required this.cy,
    required this.w,
    required this.h,
    this.angle = 0,
  });

  /// Sigmoid confidence (0..1). The native side only reports boxes with
  /// score > 0.5.
  final double score;

  /// Box center and size in camera frame pixels.
  final double cx;
  final double cy;
  final double w;
  final double h;

  /// Rotation in degrees. Always 0 for the current axis-aligned head;
  /// kept in the wire format for forward compatibility.
  final double angle;

  /// Axis-aligned bounding rect edges, convenient for overlay painting.
  double get left => cx - w / 2;
  double get top => cy - h / 2;
  double get right => cx + w / 2;
  double get bottom => cy + h / 2;

  factory GlyphDetBox.fromJson(Map<String, dynamic> j) {
    double numOf(String key) => (j[key] as num).toDouble();
    return GlyphDetBox(
      score: numOf('score'),
      cx: numOf('cx'),
      cy: numOf('cy'),
      w: numOf('w'),
      h: numOf('h'),
      angle: (j['angle'] as num?)?.toDouble() ?? 0,
    );
  }
}

/// One snapshot of the detection pipeline: the camera frame size the result
/// coordinates refer to, plus all detected boxes sorted by score descending.
class GlyphDetFrame {
  const GlyphDetFrame({
    required this.width,
    required this.height,
    required this.results,
    this.fps = 0,
    this.inferMs = 0,
    this.nbox = 0,
  });

  final int width;
  final int height;

  /// Detected boxes (max 20), sorted by score descending. Coordinates are
  /// in the camera frame coordinate space ([width] x [height]).
  final List<GlyphDetBox> results;

  /// Smoothed render-loop fps reported by the native pipeline (0 if unknown).
  final double fps;

  /// EMA of one forward+decode pass in milliseconds.
  final double inferMs;

  /// Box count of the latest decode pass (== results.length).
  final int nbox;

  /// Highest-score box, or null when nothing is detected. The app only
  /// highlights this one.
  GlyphDetBox? get topBox => results.isEmpty ? null : results.first;

  static const empty = GlyphDetFrame(width: 0, height: 0, results: []);

  /// Parses the JSON string returned by the native `pollResults` call.
  static GlyphDetFrame parse(String? jsonStr) {
    if (jsonStr == null || jsonStr.isEmpty) return empty;
    try {
      final j = jsonDecode(jsonStr) as Map<String, dynamic>;
      final results = (j['results'] as List<dynamic>? ?? const [])
          .map((e) => GlyphDetBox.fromJson(e as Map<String, dynamic>))
          .toList();
      return GlyphDetFrame(
        width: (j['w'] as num?)?.toInt() ?? 0,
        height: (j['h'] as num?)?.toInt() ?? 0,
        results: results,
        fps: (j['fps'] as num?)?.toDouble() ?? 0,
        inferMs: (j['infer_ms'] as num?)?.toDouble() ?? 0,
        nbox: (j['nbox'] as num?)?.toInt() ?? results.length,
      );
    } on FormatException {
      return empty;
    }
  }
}
