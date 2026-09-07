import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

/// 一条取件码历史记录。
class HistoryEntry {
  HistoryEntry({
    required this.code,
    required this.time,
    this.done = false,
  });

  final String code;
  final DateTime time;
  bool done;

  Map<String, dynamic> toJson() => {
        'code': code,
        'time': time.toIso8601String(),
        'done': done,
      };

  factory HistoryEntry.fromJson(Map<String, dynamic> j) => HistoryEntry(
        code: j['code'] as String,
        time: DateTime.parse(j['time'] as String),
        done: j['done'] as bool? ?? false,
      );
}

/// 基于 shared_preferences 的取件码历史存储（按时间倒序，去重，上限 50 条）。
class HistoryStore {
  static const _key = 'pickup_history';
  static const _maxEntries = 50;

  Future<List<HistoryEntry>> load() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_key);
    if (raw == null || raw.isEmpty) return [];
    try {
      final list = jsonDecode(raw) as List<dynamic>;
      return list
          .map((e) => HistoryEntry.fromJson(e as Map<String, dynamic>))
          .toList();
    } on FormatException {
      return [];
    }
  }

  Future<void> _save(List<HistoryEntry> entries) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      _key,
      jsonEncode(entries.map((e) => e.toJson()).toList()),
    );
  }

  /// 新增/置顶一条记录（同 code 去重并保留已取件标记）。
  Future<void> add(String code) async {
    final entries = await load();
    final existing = entries.where((e) => e.code == code).firstOrNull;
    entries.removeWhere((e) => e.code == code);
    entries.insert(
      0,
      HistoryEntry(
        code: code,
        time: DateTime.now(),
        done: existing?.done ?? false,
      ),
    );
    if (entries.length > _maxEntries) {
      entries.removeRange(_maxEntries, entries.length);
    }
    await _save(entries);
  }

  Future<void> markDone(String code) async {
    final entries = await load();
    for (final e in entries) {
      if (e.code == code) e.done = true;
    }
    await _save(entries);
  }

  Future<void> remove(String code) async {
    final entries = await load();
    entries.removeWhere((e) => e.code == code);
    await _save(entries);
  }
}
