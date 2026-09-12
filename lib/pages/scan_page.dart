import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:glyph_det/glyph_det.dart';
import 'package:glyph_det/glyph_det_camera_view.dart';
import 'package:permission_handler/permission_handler.dart';

import '../history/history_store.dart';
import '../ocr/glyph_det_service.dart';
import '../theme/frost_theme.dart';
import 'history_sheet.dart';

/// 扫描主页：全屏相机 + 目标高亮 overlay + 玻璃拟态输入栏/工具栏。
class ScanPage extends StatefulWidget {
  const ScanPage({super.key});

  @override
  State<ScanPage> createState() => _ScanPageState();
}

class _ScanPageState extends State<ScanPage> with TickerProviderStateMixin {
  final _det = GlyphDetService();
  final _history = HistoryStore();
  final _codeController = TextEditingController();

  Timer? _pollTimer;
  bool _polling = false;
  GlyphDetFrame _frame = GlyphDetFrame.empty;
  GlyphDetBox? _match;

  bool _permissionDenied = false;
  bool _loading = true;
  bool _cameraOpen = false;
  bool _flashOn = false;

  /// 当前相机朝向：0=前置 1=后置，默认后置
  int _facing = 1;
  bool _switchingCamera = false;

  String _targetCode = '';
  bool _hapticFired = false;

  late final AnimationController _pulseController = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1200),
  )..repeat(reverse: true);

  /// 取件码格式校验（数字段 + 连字符，如 23-5-1234）。
  static bool _isValidCode(String s) =>
      RegExp(r'^\d+(-\d+)+$').hasMatch(s.trim());

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() {
        _permissionDenied = true;
        _loading = false;
      });
      return;
    }
    try {
      await _det.loadModel();
      await _det.openCamera(_facing);
      if (!mounted) return;
      setState(() {
        _loading = false;
        _cameraOpen = true;
      });
      _startPolling();
    } catch (e) {
      debugPrint('glyph_det init failed: $e');
      if (mounted) setState(() => _loading = false);
    }
  }

  void _startPolling() {
    // 100ms 轮询：原生管线全速跑，UI 刷新不能成为新瓶颈。
    // _polling 防止上一次 channel 调用还没返回就叠加下一次。
    _pollTimer = Timer.periodic(const Duration(milliseconds: 100), (_) async {
      if (!mounted || !_cameraOpen || _polling) return;
      _polling = true;
      try {
        final frame = await _det.pollResults();
        if (!mounted) return;
        // glyph_det 语义：模型只找查询目标，top1 即匹配结果
        //（原生侧已按 score>0.5 + NMS 过滤）
        final match = _targetCode.isNotEmpty ? frame.topBox : null;
        setState(() {
          _frame = frame;
          _match = match;
        });
        if (match != null && !_hapticFired) {
          _hapticFired = true;
          HapticFeedback.mediumImpact();
          _history.markDone(_targetCode);
        } else if (match == null) {
          // 连续丢失后才重置，允许下次找到时再次震动
          _hapticFired = false;
        }
      } catch (e) {
        debugPrint('poll failed: $e');
      } finally {
        _polling = false;
      }
    });
  }

  Future<void> _startSearch() async {
    final code = _codeController.text.trim();
    if (!_isValidCode(code)) return;
    FocusScope.of(context).unfocus();
    // 目标变更要重新渲染 mask 下发给原生侧，之后的每帧检测都针对新目标
    final ok = await _det.setTargetCode(code);
    if (!mounted) return;
    if (!ok) {
      debugPrint('setTargetCode failed');
      return;
    }
    setState(() {
      _targetCode = code;
      _hapticFired = false;
      _match = null;
    });
    _history.add(code);
  }

  void _selectFromHistory(String code) {
    _codeController.text = code;
    _startSearch();
  }

  Future<void> _toggleFlash() async {
    final on = await _det.toggleFlash();
    if (mounted) setState(() => _flashOn = on);
  }

  /// 切换前后置相机：原生相机是全局单例，close 后换新朝向 reopen，
  /// SurfaceView 与输出窗口保持不变，不需要重建平台视图。
  Future<void> _switchCamera() async {
    if (!_cameraOpen || _switchingCamera) return;
    setState(() => _switchingCamera = true);
    final next = _facing == 1 ? 0 : 1;
    await _det.closeCamera();
    final ok = await _det.openCamera(next);
    if (!mounted) return;
    if (ok) {
      setState(() {
        _facing = next;
        _switchingCamera = false;
        _flashOn = false; // 新会话的闪光灯状态已重置
        _frame = GlyphDetFrame.empty;
        _match = null;
      });
    } else {
      // 打开失败回滚原相机
      await _det.openCamera(_facing);
      if (mounted) setState(() => _switchingCamera = false);
    }
  }

  Future<void> _openHistory() async {
    final code = await showHistorySheet(context, _history);
    if (code != null && mounted) _selectFromHistory(code);
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _pulseController.dispose();
    _codeController.dispose();
    _det.closeCamera();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // 键盘弹出时不要让相机预览和底部工具栏跟着上浮重排，
      // 全屏 Stack 不需要为输入法让位，避免整页 relayout 的性能浪费
      resizeToAvoidBottomInset: false,
      // Stack 在松散约束下会收缩到最高非定位子节点的高度，
      // 必须用 SizedBox.expand 强制撑满，否则相机预览会被压成一条
      body: SizedBox.expand(
        child: Stack(
        children: [
          // 相机预览（原生渲染）
          if (_cameraOpen)
            Positioned.fill(
              child: LayoutBuilder(
                builder: (context, constraints) {
                  final size = Size(constraints.maxWidth, constraints.maxHeight);
                  return Stack(
                    children: [
                      const Positioned.fill(child: GlyphDetCameraView()),
                      if (_match != null)
                        Positioned.fill(
                          child: AnimatedBuilder(
                            animation: _pulseController,
                            builder: (context, _) => CustomPaint(
                              painter: TargetHighlightPainter(
                                box: _match!,
                                label: _targetCode,
                                frame: _frame,
                                viewSize: size,
                                pulse: _pulseController.value,
                              ),
                            ),
                          ),
                        ),
                    ],
                  );
                },
              ),
            ),

          // 加载 / 权限占位
          if (_loading || _permissionDenied || !_cameraOpen)
            Positioned.fill(
              child: Container(
                color: FrostColors.background,
                child: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      SvgPicture.asset('assets/images/logo.svg', width: 160),
                      const SizedBox(height: 24),
                      if (_loading)
                        Text('模型加载中…',
                            style: Theme.of(context).textTheme.headlineMedium),
                      if (_permissionDenied) ...[
                        Text('需要相机权限才能找快递哦',
                            style: Theme.of(context).textTheme.headlineMedium),
                        const SizedBox(height: 16),
                        FilledButton(
                          onPressed: openAppSettings,
                          child: const Text('去开启权限'),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ),

          // 顶部：取件码输入胶囊
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  GlassContainer(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 20, vertical: 6),
                    child: Row(
                      children: [
                        Expanded(
                          child: TextField(
                            controller: _codeController,
                            keyboardType: TextInputType.text,
                            inputFormatters: [
                              FilteringTextInputFormatter.allow(
                                  RegExp(r'[0-9-]')),
                            ],
                            style: const TextStyle(
                              fontSize: 22,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 2,
                              color: FrostColors.primary,
                              fontFeatures: [FontFeature.tabularFigures()],
                            ),
                            decoration:
                                const InputDecoration(hintText: '输入取件码 如 23-5-1234'),
                            onSubmitted: (_) => _startSearch(),
                          ),
                        ),
                        FilledButton(
                          style: FilledButton.styleFrom(
                            backgroundColor: FrostColors.primary,
                            padding: const EdgeInsets.symmetric(
                                horizontal: 16, vertical: 10),
                            minimumSize: Size.zero,
                            tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                          ),
                          onPressed: _startSearch,
                          child: const Text(
                            '开始找',
                            style: TextStyle(
                                fontSize: 16, fontWeight: FontWeight.w700),
                          ),
                        ),
                      ],
                    ),
                  ),

                  // 找到提示条
                  AnimatedSwitcher(
                    duration: const Duration(milliseconds: 250),
                    child: _match != null
                        ? Padding(
                            key: const ValueKey('found'),
                            padding: const EdgeInsets.only(top: 10),
                            child: GlassContainer(
                              borderRadius: 18,
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 16, vertical: 10),
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  SvgPicture.asset(
                                    'assets/images/logo_head.svg',
                                    width: 32,
                                  ),
                                  const SizedBox(width: 10),
                                  Text(
                                    '找到啦！$_targetCode',
                                    style: const TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.w700,
                                      color: FrostColors.primary,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          )
                        : const SizedBox.shrink(key: ValueKey('none')),
                  ),
                ],
              ),
            ),
          ),

          // 底部：玻璃工具栏（识别状态 / 手电筒 / 历史）
          Positioned(
            left: 16,
            right: 16,
            bottom: 0,
            child: SafeArea(
              child: Padding(
                padding: const EdgeInsets.only(bottom: 16),
                child: GlassContainer(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Expanded(
                        child: Padding(
                          padding: const EdgeInsets.only(left: 8),
                          child: Row(
                            children: [
                              Flexible(
                                child: Text(
                                  _targetCode.isEmpty
                                      ? '输入取件码开始寻找'
                                      : (_match == null
                                          ? '正在寻找 $_targetCode …'
                                          : '已锁定目标'),
                                  overflow: TextOverflow.ellipsis,
                                  style: const TextStyle(
                                    color: FrostColors.strokeBlue,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ),
                              // 渲染帧率与推理耗时只在调试构建显示
                              if (kDebugMode)
                                Padding(
                                  padding: const EdgeInsets.only(left: 10),
                                  child: Text(
                                    '${_frame.fps.toStringAsFixed(1)}FPS '
                                    '${_frame.inferMs.toStringAsFixed(0)}ms',
                                    style: const TextStyle(
                                      color: FrostColors.strokeBlue,
                                      fontSize: 12,
                                      fontFeatures: [
                                        FontFeature.tabularFigures()
                                      ],
                                    ),
                                  ),
                                ),
                            ],
                          ),
                        ),
                      ),
                      Row(
                        children: [
                          IconButton(
                            icon: Icon(
                              Icons.cameraswitch_outlined,
                              color: _switchingCamera
                                  ? FrostColors.strokeBlue
                                      .withValues(alpha: 0.4)
                                  : FrostColors.primary,
                            ),
                            tooltip: _facing == 1 ? '切换到前置' : '切换到后置',
                            onPressed:
                                _cameraOpen && !_switchingCamera
                                    ? _switchCamera
                                    : null,
                          ),
                          IconButton(
                            icon: Icon(
                              _flashOn ? Icons.flash_on : Icons.flash_off,
                              color: _flashOn
                                  ? FrostColors.glowBlue
                                  : FrostColors.strokeBlue,
                            ),
                            onPressed: _cameraOpen ? _toggleFlash : null,
                          ),
                          IconButton(
                            icon: const Icon(Icons.history,
                                color: FrostColors.primary),
                            onPressed: _openHistory,
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
        ),
      ),
    );
  }
}

/// 目标高亮：霜月蓝圆角框 + 脉冲光晕 + 四角括号。
///
/// 坐标映射：原生侧已按输出窗口宽高比裁剪 ROI，这里按 BoxFit.cover
/// 语义做等比缩放 + 居中（与插件 example 的映射一致）。
class TargetHighlightPainter extends CustomPainter {
  TargetHighlightPainter({
    required this.box,
    required this.label,
    required this.frame,
    required this.viewSize,
    required this.pulse,
  });

  final GlyphDetBox box;
  final String label;
  final GlyphDetFrame frame;
  final Size viewSize;

  /// 0..1 呼吸值
  final double pulse;

  @override
  void paint(Canvas canvas, Size size) {
    if (frame.width <= 0 || frame.height <= 0) return;
    final scale = math.max(
        viewSize.width / frame.width, viewSize.height / frame.height);
    final dx = (viewSize.width - frame.width * scale) / 2;
    final dy = (viewSize.height - frame.height * scale) / 2;
    final pad = 8.0 + pulse * 6.0;
    final rect = Rect.fromLTRB(
      box.left * scale + dx - pad,
      box.top * scale + dy - pad,
      box.right * scale + dx + pad,
      box.bottom * scale + dy + pad,
    );
    final rrect = RRect.fromRectAndRadius(rect, const Radius.circular(12));

    // 脉冲光晕
    final glowPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 10 + pulse * 8
      ..color = FrostColors.glowCyan.withValues(alpha: 0.18 + pulse * 0.10)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 12);
    canvas.drawRRect(rrect, glowPaint);

    // 主框
    final borderPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = FrostColors.primary;
    canvas.drawRRect(rrect, borderPaint);

    // 四角括号（可爱感）
    final cornerPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 5
      ..strokeCap = StrokeCap.round
      ..color = FrostColors.glowBlue;
    const len = 18.0;
    final corners = [
      (rect.topLeft, const Offset(1, 0), const Offset(0, 1)),
      (rect.topRight, const Offset(-1, 0), const Offset(0, 1)),
      (rect.bottomLeft, const Offset(1, 0), const Offset(0, -1)),
      (rect.bottomRight, const Offset(-1, 0), const Offset(0, -1)),
    ];
    for (final (origin, dx, dy) in corners) {
      canvas.drawLine(
          origin, origin + Offset(dx.dx * len, dx.dy * len), cornerPaint);
      canvas.drawLine(
          origin, origin + Offset(dy.dx * len, dy.dy * len), cornerPaint);
    }

    // 目标文本标签
    final textPainter = TextPainter(
      text: TextSpan(
        text: label,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 16,
          fontWeight: FontWeight.w700,
          backgroundColor: FrostColors.primary,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    textPainter.paint(
        canvas, Offset(rect.left, rect.top - textPainter.height - 6));
  }

  @override
  bool shouldRepaint(TargetHighlightPainter old) =>
      old.pulse != pulse ||
      old.box != box ||
      old.label != label ||
      old.frame != frame;
}
