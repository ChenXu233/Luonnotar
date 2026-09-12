import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:glyph_det/glyph_det.dart';

/// Renders a pickup-code query mask with the same semantics as the training
/// side (`core/synth.py: render_mask`):
///
/// - render the whole string with a regular sans font at a large base size
/// - measure the tight ink bounding box (like PIL `font.getbbox`)
/// - scale proportionally so the ink height fills the mask height
/// - if the scaled width overflows, shrink proportionally (never squash)
/// - paste left-aligned, vertically centered on a white canvas
/// - black text (0) on white background (255)
///
/// Returns [GlyphDet.maskByteCount] grayscale bytes, row-major
/// [GlyphDet.maskHeight] x [GlyphDet.maskWidth].
Future<Uint8List> renderQueryMask(String text) async {
  const int W = GlyphDet.maskWidth; // 384
  const int H = GlyphDet.maskHeight; // 64

  final bytes = Uint8List(W * H);
  if (text.trim().isEmpty) {
    bytes.fillRange(0, bytes.length, 255);
    return bytes;
  }

  const double baseFontSize = 96;
  final builder = ui.ParagraphBuilder(
    ui.ParagraphStyle(
      fontSize: baseFontSize,
      // regular sans; the model is trained with held-out fonts, so the exact
      // family does not matter as long as it is a plain sans-serif
      fontFamily: 'sans-serif',
    ),
  )..addText(text);
  final paragraph = builder.build()
    ..layout(const ui.ParagraphConstraints(width: double.infinity));

  // tight ink bbox of the whole string (PIL getbbox equivalent)
  final boxes = paragraph.getBoxesForRange(0, text.length);
  if (boxes.isEmpty) {
    bytes.fillRange(0, bytes.length, 255);
    paragraph.dispose();
    return bytes;
  }
  double inkLeft = boxes.first.left;
  double inkTop = boxes.first.top;
  double inkRight = boxes.first.right;
  double inkBottom = boxes.first.bottom;
  for (final b in boxes) {
    inkLeft = inkLeft < b.left ? inkLeft : b.left;
    inkTop = inkTop < b.top ? inkTop : b.top;
    inkRight = inkRight > b.right ? inkRight : b.right;
    inkBottom = inkBottom > b.bottom ? inkBottom : b.bottom;
  }
  final inkW = (inkRight - inkLeft).clamp(1.0, double.infinity);
  final inkH = (inkBottom - inkTop).clamp(1.0, double.infinity);

  // proportional scale to mask height; shrink proportionally if too wide
  double scale = H / inkH;
  if (inkW * scale > W) {
    scale = W / inkW;
  }
  final nh = (inkH * scale).clamp(1.0, double.infinity);

  final recorder = ui.PictureRecorder();
  final canvas = Canvas(recorder);
  canvas.drawRect(
    Rect.fromLTWH(0, 0, W.toDouble(), H.toDouble()),
    Paint()..color = Colors.white,
  );
  canvas.save();
  // left-aligned, vertically centered
  canvas.translate(0, (H - nh) / 2);
  canvas.scale(scale);
  canvas.translate(-inkLeft, -inkTop);
  canvas.drawParagraph(paragraph, Offset.zero);
  canvas.restore();

  final picture = recorder.endRecording();
  final image = await picture.toImage(W, H);
  final rgba = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
  paragraph.dispose();
  picture.dispose();
  image.dispose();

  if (rgba == null) {
    bytes.fillRange(0, bytes.length, 255);
    return bytes;
  }

  // black text on white bg -> the R channel alone is the gray value
  final px = rgba.buffer.asUint8List();
  for (int i = 0; i < W * H; i++) {
    bytes[i] = px[i * 4];
  }
  return bytes;
}
