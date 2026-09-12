import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:glyph_det/glyph_det.dart';
import 'package:glyph_det/glyph_det_camera_view.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';

import 'mask_render.dart';

void main() {
  runApp(const GlyphDetExampleApp());
}

class GlyphDetExampleApp extends StatelessWidget {
  const GlyphDetExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const ScanPage(),
    );
  }
}

class ScanPage extends StatefulWidget {
  const ScanPage({super.key});

  @override
  State<ScanPage> createState() => _ScanPageState();
}

class _ScanPageState extends State<ScanPage> {
  final _det = GlyphDet();
  final _codeController = TextEditingController(text: '24-6-1234');

  bool _modelLoaded = false;
  bool _cameraOpen = false;
  bool _flashOn = false;
  bool _maskSet = false;
  String _status = 'init';

  GlyphDetFrame _frame = GlyphDetFrame.empty;
  Timer? _pollTimer;

  static const _modelFiles = ['glyphdet.param', 'glyphdet.bin'];

  @override
  void initState() {
    super.initState();
    _boot();
  }

  Future<void> _boot() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() => _status = 'camera permission denied');
      return;
    }
    await _loadModel();
    await _applyMask();
    await _openCamera();
    _pollTimer = Timer.periodic(const Duration(milliseconds: 66), (_) => _poll());
  }

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

  Future<void> _loadModel() async {
    setState(() => _status = 'loading model (Vulkan shader compile may take seconds)...');
    final param = await _copyAssetToFile(_modelFiles[0]);
    final bin = await _copyAssetToFile(_modelFiles[1]);
    // cpugpu: 1 = Vulkan (default), 0 = CPU
    final ok = await _det.loadModel(param, bin, cpugpu: 1);
    setState(() {
      _modelLoaded = ok;
      _status = ok ? 'model loaded' : 'model load failed';
    });
  }

  Future<void> _applyMask() async {
    final gray = await renderQueryMask(_codeController.text.trim());
    final ok = await _det.setQueryMask(gray);
    setState(() {
      _maskSet = ok;
      _status = ok
          ? 'query: ${_codeController.text.trim()} (mask set)'
          : 'setQueryMask failed (mask set: $_maskSet)';
    });
  }

  Future<void> _openCamera() async {
    final ok = await _det.openCamera(1); // 1 = back
    setState(() {
      _cameraOpen = ok;
      _status = ok ? 'camera open' : 'openCamera failed';
    });
  }

  Future<void> _poll() async {
    final frame = await _det.pollResults();
    if (mounted) setState(() => _frame = frame);
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _det.closeCamera();
    _codeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final top = _frame.topBox;
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          if (_cameraOpen) const GlyphDetCameraView(),
          if (_cameraOpen)
            CustomPaint(
              painter: _HighlightPainter(_frame),
            ),
          // stats
          Positioned(
            left: 12,
            top: MediaQuery.of(context).padding.top + 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                'fps ${_frame.fps.toStringAsFixed(1)}  '
                'infer ${_frame.inferMs.toStringAsFixed(1)}ms  '
                'nbox ${_frame.nbox}\n'
                '${top != null ? 'top ${top.score.toStringAsFixed(3)}' : 'no detection'}\n'
                '$_status',
                style: const TextStyle(fontSize: 12, color: Colors.white),
              ),
            ),
          ),
          // flash
          Positioned(
            right: 12,
            top: MediaQuery.of(context).padding.top + 8,
            child: IconButton(
              icon: Icon(_flashOn ? Icons.flash_on : Icons.flash_off),
              color: Colors.white,
              onPressed: () async {
                final ok = await _det.toggleFlash();
                if (ok) setState(() => _flashOn = !_flashOn);
              },
            ),
          ),
          // query input
          Positioned(
            left: 12,
            right: 12,
            bottom: MediaQuery.of(context).padding.bottom + 12,
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _codeController,
                    style: const TextStyle(fontSize: 18, letterSpacing: 1.5),
                    decoration: InputDecoration(
                      filled: true,
                      fillColor: Colors.black54,
                      hintText: 'pickup code, e.g. 24-6-1234',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                      isDense: true,
                    ),
                    onSubmitted: (_) => _applyMask(),
                  ),
                ),
                const SizedBox(width: 8),
                FilledButton(
                  onPressed: _modelLoaded ? _applyMask : null,
                  child: const Text('Set'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Draws the top-1 box (bright) and the remaining boxes (faint), mapping
/// camera-frame coordinates onto the preview widget with BoxFit.cover
/// semantics (the native roi crop already matches the window aspect).
class _HighlightPainter extends CustomPainter {
  _HighlightPainter(this.frame);

  final GlyphDetFrame frame;

  @override
  void paint(Canvas canvas, Size size) {
    if (frame.width <= 0 || frame.height <= 0 || frame.results.isEmpty) return;

    final scale = (size.width / frame.width) > (size.height / frame.height)
        ? size.width / frame.width
        : size.height / frame.height;
    final dx = (size.width - frame.width * scale) / 2;
    final dy = (size.height - frame.height * scale) / 2;

    Rect mapBox(GlyphDetBox b) => Rect.fromLTRB(
          b.left * scale + dx,
          b.top * scale + dy,
          b.right * scale + dx,
          b.bottom * scale + dy,
        );

    final faint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = Colors.yellowAccent.withValues(alpha: 0.35);
    for (int i = 1; i < frame.results.length; i++) {
      canvas.drawRect(mapBox(frame.results[i]), faint);
    }

    final top = frame.topBox!;
    final rect = mapBox(top);
    final border = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4
      ..color = Colors.greenAccent;
    canvas.drawRect(rect, border);

    final label = TextPainter(
      text: TextSpan(
        text: top.score.toStringAsFixed(3),
        style: const TextStyle(
          fontSize: 14,
          color: Colors.black,
          backgroundColor: Colors.greenAccent,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    label.paint(canvas, Offset(rect.left, rect.top - label.height));
  }

  @override
  bool shouldRepaint(_HighlightPainter oldDelegate) =>
      !identical(oldDelegate.frame, frame);
}
