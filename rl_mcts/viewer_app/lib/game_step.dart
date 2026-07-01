/// One snapshot of the match, matching the JSON shape produced by
/// GameRecorder.new_vis_steps() / save_visualizer() on the Python side:
/// { "select": {...}, "logs": [...], "current": {"turn", "players": [...]}, ... }
class GameStep {
  final int turn;
  final int result;
  final List<PlayerState> players;
  final List<LogEntry> logs;
  final String selectContext;

  GameStep({
    required this.turn,
    required this.result,
    required this.players,
    required this.logs,
    required this.selectContext,
  });

  factory GameStep.fromJson(Map<String, dynamic> json) {
    final current = json['current'] as Map<String, dynamic>? ?? {};
    final playersJson = current['players'] as List<dynamic>? ?? [];
    final select = json['select'] as Map<String, dynamic>?;
    final logsJson = json['logs'] as List<dynamic>? ?? [];
    return GameStep(
      turn: current['turn'] as int? ?? 0,
      result: current['result'] as int? ?? -1,
      players: playersJson
          .map((p) => PlayerState.fromJson(p as Map<String, dynamic>))
          .toList(),
      logs: logsJson
          .map((l) => LogEntry.fromJson(l as Map<String, dynamic>))
          .toList(),
      selectContext: select == null
          ? ''
          : '${select['type'] ?? ''} ${select['context'] ?? ''}'.trim(),
    );
  }
}

class PlayerState {
  final GameCard? active;
  final List<GameCard> bench;
  final int handCount;
  final int deckCount;
  final int prizeCount;
  final int discardCount;

  PlayerState({
    required this.active,
    required this.bench,
    required this.handCount,
    required this.deckCount,
    required this.prizeCount,
    required this.discardCount,
  });

  factory PlayerState.fromJson(Map<String, dynamic> json) {
    final activeList = json['active'] as List<dynamic>? ?? [];
    final benchList = json['bench'] as List<dynamic>? ?? [];
    final prizeList = json['prize'] as List<dynamic>? ?? [];
    final discardList = json['discard'] as List<dynamic>? ?? [];
    return PlayerState(
      active: activeList.isEmpty
          ? null
          : GameCard.fromJson(activeList.first as Map<String, dynamic>),
      bench: benchList
          .map((c) => GameCard.fromJson(c as Map<String, dynamic>))
          .toList(),
      handCount: json['handCount'] as int? ?? 0,
      deckCount: json['deckCount'] as int? ?? 0,
      prizeCount: prizeList.length,
      discardCount: discardList.length,
    );
  }
}

class GameCard {
  final String name;
  final int? hp;
  final int? maxHp;

  GameCard({required this.name, this.hp, this.maxHp});

  factory GameCard.fromJson(Map<String, dynamic> json) {
    return GameCard(
      name: json['name'] as String? ?? '?',
      hp: json['hp'] as int?,
      maxHp: json['maxHp'] as int?,
    );
  }
}

class LogEntry {
  final String type;
  final int? playerIndex;
  final int? cardId;

  LogEntry({required this.type, this.playerIndex, this.cardId});

  factory LogEntry.fromJson(Map<String, dynamic> json) {
    return LogEntry(
      type: json['type'] as String? ?? '?',
      playerIndex: json['playerIndex'] as int?,
      cardId: json['cardId'] as int?,
    );
  }

  @override
  String toString() {
    final who = playerIndex == null ? '' : 'P$playerIndex ';
    final card = cardId == null ? '' : ' (card $cardId)';
    return '$who$type$card';
  }
}
