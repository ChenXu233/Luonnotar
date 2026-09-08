import 'dart:ui';

import 'package:flutter/material.dart';

/// 霜月主题色板（取自 docs/assert/logo.svg 的渐变与描边色）。
class FrostColors {
  FrostColors._();

  /// 霜月蓝·主色
  static const primary = Color(0xFF4369C8);

  /// 霜月白·底色
  static const background = Color(0xFFF5F9FF);

  /// 辉光白
  static const glowWhite = Color(0xFFEDFFFF);

  /// 描边蓝（细描边、次级文字）
  static const strokeBlue = Color(0xFF7A8DC8);

  /// 辉光点缀（找到目标的脉冲光效）
  /// 设计决策：界面不使用渐变填充，logo 的霜月渐变只以“外发光辉光”
  /// 的形式出现在高亮反馈中（scan_page 的脉冲光晕用这两个色）。
  static const glowBlue = Color(0xFF3366FF);
  static const glowCyan = Color(0xFF33CCFF);

  /// 玻璃面填充色（约 60% 霜月白）
  static const glassFill = Color(0x99F5F9FF);
  static const glassBorder = Color(0x66FFFFFF);
}

final frostTheme = ThemeData(
  useMaterial3: true,
  scaffoldBackgroundColor: FrostColors.background,
  colorScheme: ColorScheme.fromSeed(
    seedColor: FrostColors.primary,
    brightness: Brightness.light,
  ).copyWith(
    primary: FrostColors.primary,
    surface: FrostColors.background,
  ),
  textTheme: const TextTheme(
    headlineMedium: TextStyle(
      color: FrostColors.primary,
      fontWeight: FontWeight.w700,
      letterSpacing: 1.2,
    ),
  ),
  inputDecorationTheme: InputDecorationTheme(
    border: InputBorder.none,
    hintStyle: TextStyle(color: FrostColors.strokeBlue.withValues(alpha: 0.7)),
  ),
);

/// 轻玻璃拟态容器：高斯模糊 + 半透明白 + 大圆角 + 极浅描边。
class GlassContainer extends StatelessWidget {
  const GlassContainer({
    super.key,
    required this.child,
    this.borderRadius = 24,
    this.padding,
    this.margin,
    this.blur = 16,
    this.fill = FrostColors.glassFill,
  });

  final Widget child;
  final double borderRadius;
  final EdgeInsetsGeometry? padding;
  final EdgeInsetsGeometry? margin;
  final double blur;
  final Color fill;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: margin,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(borderRadius),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: blur, sigmaY: blur),
          child: Container(
            padding: padding,
            decoration: BoxDecoration(
              color: fill,
              borderRadius: BorderRadius.circular(borderRadius),
              border: Border.all(color: FrostColors.glassBorder),
            ),
            child: child,
          ),
        ),
      ),
    );
  }
}
