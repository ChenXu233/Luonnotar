import 'package:fast_paddle_ocr/ocr.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:luonnotar/matching/pickup_matcher.dart';

OcrResultBox box(String text, double cx, double cy,
        {double w = 60, double h = 20}) =>
    OcrResultBox(
        text: text, prob: 0.9, rprob: 0.9, cx: cx, cy: cy, w: w, h: h, angle: 0);

void main() {
  group('normalize', () {
    test('去连字符与空白', () {
      expect(PickupMatcher.normalize('23-5-1234'), '2351234');
      expect(PickupMatcher.normalize(' 23 - 5 '), '235');
    });

    test('isValidCode', () {
      expect(PickupMatcher.isValidCode('23-5-1234'), isTrue);
      expect(PickupMatcher.isValidCode('1-2'), isTrue);
      expect(PickupMatcher.isValidCode('2351234'), isFalse);
      expect(PickupMatcher.isValidCode('23-5-'), isFalse);
      expect(PickupMatcher.isValidCode('abc'), isFalse);
    });
  });

  group('merge', () {
    test('同行相邻框拼接', () {
      final boxes = [box('23-5-', 50, 100), box('1234', 130, 102)];
      final merged = PickupMatcher.merge(boxes);
      expect(merged, hasLength(1));
      expect(merged.single.text, '23-5-1234');
      expect(merged.single.left, 20);
      expect(merged.single.right, 160);
    });

    test('不同行不拼接', () {
      final boxes = [box('23-5-1234', 50, 100), box('6-1-0789', 60, 200)];
      final merged = PickupMatcher.merge(boxes);
      expect(merged, hasLength(2));
    });

    test('水平间距过大不拼接', () {
      final boxes = [box('23-5-', 50, 100), box('1234', 300, 100)];
      final merged = PickupMatcher.merge(boxes);
      expect(merged, hasLength(2));
    });
  });

  group('findBest', () {
    test('精确匹配单个框', () {
      final boxes = [
        box('23-5-1234', 50, 100),
        box('6-1-0789', 60, 200),
      ];
      final m = PickupMatcher.findBest('23-5-1234', boxes);
      expect(m, isNotNull);
      expect(m!.level, MatchLevel.exact);
    });

    test('跨框拼接后匹配', () {
      final boxes = [box('23-5-', 50, 100), box('1234', 130, 102)];
      final m = PickupMatcher.findBest('23-5-1234', boxes);
      expect(m, isNotNull);
      expect(m!.level, MatchLevel.exact);
    });

    test('OCR 错一位仍模糊命中', () {
      final boxes = [box('23-5-1284', 50, 100)];
      final m = PickupMatcher.findBest('23-5-1234', boxes);
      expect(m, isNotNull);
      expect(m!.level, MatchLevel.fuzzy);
    });

    test('尾号一致兜底命中', () {
      final boxes = [box('7-2-1234', 50, 100)];
      final m = PickupMatcher.findBest('23-5-1234', boxes);
      expect(m, isNotNull);
      expect(m!.level, MatchLevel.tailOnly);
    });

    test('完全不相关不命中', () {
      final boxes = [box('99-9-9999', 50, 100)];
      expect(PickupMatcher.findBest('23-5-1234', boxes), isNull);
    });

    test('空目标不命中', () {
      expect(PickupMatcher.findBest('', [box('23-5-1234', 50, 100)]), isNull);
    });
  });
}
