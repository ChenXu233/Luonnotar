import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'pages/scan_page.dart';
import 'theme/frost_theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // 锁定竖屏：保证原生预览坐标系到 overlay 的映射是简单缩放
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const LuonnotarApp());
}

class LuonnotarApp extends StatelessWidget {
  const LuonnotarApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Luonnotar',
      debugShowCheckedModeBanner: false,
      theme: frostTheme,
      home: const ScanPage(),
    );
  }
}
