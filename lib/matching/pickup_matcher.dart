import 'package:fast_paddle_ocr/ocr.dart';

/// 取件码（如 `23-5-1234`）匹配引擎：规范化、邻近框合并、多级匹配。
class PickupMatcher {
  PickupMatcher._();

  /// 规范化：去掉连字符与空白，便于双侧比对。
  static String normalize(String s) => s.replaceAll(RegExp(r'[-\s]'), '');

  /// 是否合法的取件码格式（数字段 + 连字符，如 23-5-1234）。
  static bool isValidCode(String s) =>
      RegExp(r'^\d+(-\d+)+$').hasMatch(s.trim());

  /// 把同一行、横向相邻的识别框拼接为一个候选。
  ///
  /// 取件码标签可能被识别成多个独立框（如 `23-5` 与 `1234` 分成两框），
  /// 合并规则：y 中心差 < 两者平均高度的 0.6 倍视为同行，水平间距
  /// < 平均高度的 1.5 倍则按 x 顺序拼接。
  static List<MergedCandidate> merge(List<OcrResultBox> boxes) {
    final sorted = [...boxes]..sort((a, b) => a.cy.compareTo(b.cy));
    final used = List.filled(sorted.length, false);
    final candidates = <MergedCandidate>[];

    for (var i = 0; i < sorted.length; i++) {
      if (used[i]) continue;
      used[i] = true;
      final group = [sorted[i]];

      for (var j = i + 1; j < sorted.length; j++) {
        if (used[j]) continue;
        final a = group.first;
        final b = sorted[j];
        final avgH = (a.h + b.h) / 2;
        if ((a.cy - b.cy).abs() < avgH * 0.6) {
          // 同行：检查与组内任意框水平相邻
          final gapOk = group.any((g) {
            final gap = (b.cx - g.cx).abs() - (g.w + b.w) / 2;
            return gap < avgH * 1.5;
          });
          if (gapOk) {
            used[j] = true;
            group.add(b);
          }
        }
      }

      group.sort((a, b) => a.cx.compareTo(b.cx));
      candidates.add(MergedCandidate.fromGroup(group));
    }
    return candidates;
  }

  /// 对合并候选做多级匹配，返回最佳命中（未命中返回 null）。
  static MatchResult? findBest(String target, List<OcrResultBox> boxes) {
    final normTarget = normalize(target.trim());
    if (normTarget.isEmpty) return null;

    MatchResult? best;
    for (final c in merge(boxes)) {
      final normText = normalize(c.text);
      if (normText.isEmpty) continue;

      final level = _matchLevel(normTarget, normText);
      if (level == null) continue;
      if (best == null || level.index < best.level.index) {
        best = MatchResult(candidate: c, level: level);
        if (level == MatchLevel.exact) break;
      }
    }
    return best;
  }

  static MatchLevel? _matchLevel(String normTarget, String normText) {
    if (normText == normTarget) return MatchLevel.exact;
    // 标签可能带额外文字（如“取件码”已被过滤，一般只剩数字），双向包含算部分匹配
    if (normText.contains(normTarget) || normTarget.contains(normText)) {
      return MatchLevel.partial;
    }
    if (_editDistanceAtMost(normTarget, normText, 1)) {
      return MatchLevel.fuzzy;
    }
    // 兜底：尾号 4 位一致（货架分组前缀相同时易误判，优先级最低）
    if (normTarget.length >= 4 &&
        normText.length >= 4 &&
        normTarget.substring(normTarget.length - 4) ==
            normText.substring(normText.length - 4)) {
      return MatchLevel.tailOnly;
    }
    return null;
  }

  /// 编辑距离是否 <= [limit]（提前退出的 Levenshtein）。
  static bool _editDistanceAtMost(String a, String b, int limit) {
    if ((a.length - b.length).abs() > limit) return false;
    var prev = List.generate(b.length + 1, (j) => j);
    for (var i = 1; i <= a.length; i++) {
      final cur = List.filled(b.length + 1, 0);
      cur[0] = i;
      var rowMin = cur[0];
      for (var j = 1; j <= b.length; j++) {
        final cost = a[i - 1] == b[j - 1] ? 0 : 1;
        cur[j] = [
          prev[j] + 1,
          cur[j - 1] + 1,
          prev[j - 1] + cost,
        ].reduce((x, y) => x < y ? x : y);
        if (cur[j] < rowMin) rowMin = cur[j];
      }
      if (rowMin > limit) return false;
      prev = cur;
    }
    return prev[b.length] <= limit;
  }
}

/// 一组相邻框拼接出的候选文本及其联合包围框。
class MergedCandidate {
  MergedCandidate({
    required this.text,
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  final String text;
  final double left;
  final double top;
  final double right;
  final double bottom;

  double get width => right - left;
  double get height => bottom - top;

  factory MergedCandidate.fromGroup(List<OcrResultBox> group) {
    final text = group.map((b) => b.text).join();
    return MergedCandidate(
      text: text,
      left: group.map((b) => b.left).reduce((a, b) => a < b ? a : b),
      top: group.map((b) => b.top).reduce((a, b) => a < b ? a : b),
      right: group.map((b) => b.right).reduce((a, b) => a > b ? a : b),
      bottom: group.map((b) => b.bottom).reduce((a, b) => a > b ? a : b),
    );
  }
}

enum MatchLevel { exact, partial, fuzzy, tailOnly }

class MatchResult {
  const MatchResult({required this.candidate, required this.level});

  final MergedCandidate candidate;
  final MatchLevel level;
}
