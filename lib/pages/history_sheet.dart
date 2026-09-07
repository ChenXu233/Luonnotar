import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

import '../history/history_store.dart';
import '../theme/frost_theme.dart';

/// 历史取件码 BottomSheet：玻璃拟态列表，点选返回取件码，左滑删除。
Future<String?> showHistorySheet(BuildContext context, HistoryStore store) {
  return showModalBottomSheet<String>(
    context: context,
    backgroundColor: Colors.transparent,
    isScrollControlled: true,
    builder: (context) => _HistorySheet(store: store),
  );
}

class _HistorySheet extends StatefulWidget {
  const _HistorySheet({required this.store});

  final HistoryStore store;

  @override
  State<_HistorySheet> createState() => _HistorySheetState();
}

class _HistorySheetState extends State<_HistorySheet> {
  List<HistoryEntry> _entries = [];

  @override
  void initState() {
    super.initState();
    _reload();
  }

  Future<void> _reload() async {
    final entries = await widget.store.load();
    if (mounted) setState(() => _entries = entries);
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
      child: GlassContainer(
        borderRadius: 28,
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: FrostColors.strokeBlue.withValues(alpha: 0.4),
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const SizedBox(height: 14),
            Text('历史取件码', style: Theme.of(context).textTheme.headlineMedium),
            const SizedBox(height: 10),
            if (_entries.isEmpty)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 24),
                child: Center(
                  child: Column(
                    children: [
                      SvgPicture.asset('assets/images/logo.svg', width: 100),
                      const SizedBox(height: 12),
                      const Text(
                        '还没有记录，去扫一个吧',
                        style: TextStyle(color: FrostColors.strokeBlue),
                      ),
                    ],
                  ),
                ),
              )
            else
              ConstrainedBox(
                constraints: BoxConstraints(
                  maxHeight: MediaQuery.of(context).size.height * 0.4,
                ),
                child: ListView.builder(
                  shrinkWrap: true,
                  itemCount: _entries.length,
                  itemBuilder: (context, i) {
                    final e = _entries[i];
                    return Dismissible(
                      key: ValueKey(e.code),
                      direction: DismissDirection.endToStart,
                      background: Container(
                        alignment: Alignment.centerRight,
                        padding: const EdgeInsets.only(right: 16),
                        child: const Icon(Icons.delete_outline,
                            color: FrostColors.strokeBlue),
                      ),
                      onDismissed: (_) async {
                        await widget.store.remove(e.code);
                        _reload();
                      },
                      child: ListTile(
                        contentPadding: EdgeInsets.zero,
                        leading: Icon(
                          e.done
                              ? Icons.check_circle
                              : Icons.radio_button_unchecked,
                          color: e.done
                              ? FrostColors.glowCyan
                              : FrostColors.strokeBlue,
                        ),
                        title: Text(
                          e.code,
                          style: const TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 1.5,
                            color: FrostColors.primary,
                          ),
                        ),
                        subtitle: Text(
                          _formatTime(e.time),
                          style:
                              const TextStyle(color: FrostColors.strokeBlue),
                        ),
                        onTap: () => Navigator.of(context).pop(e.code),
                      ),
                    );
                  },
                ),
              ),
          ],
        ),
      ),
    );
  }

  static String _formatTime(DateTime t) {
    final now = DateTime.now();
    final isToday =
        t.year == now.year && t.month == now.month && t.day == now.day;
    final hm =
        '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}';
    if (isToday) return '今天 $hm';
    return '${t.month}月${t.day}日 $hm';
  }
}
