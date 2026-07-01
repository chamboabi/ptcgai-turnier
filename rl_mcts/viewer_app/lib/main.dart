import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import 'game_step.dart';

void main() => runApp(const ViewerApp());

class ViewerApp extends StatelessWidget {
  const ViewerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PTCG Live Viewer',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: const ViewerPage(),
    );
  }
}

class ViewerPage extends StatefulWidget {
  const ViewerPage({super.key});

  @override
  State<ViewerPage> createState() => _ViewerPageState();
}

class _ViewerPageState extends State<ViewerPage> {
  static const _url = 'ws://localhost:8765';

  WebSocketChannel? _channel;
  StreamSubscription? _sub;
  Timer? _reconnectTimer;
  String _status = 'connecting';

  String? _gameId;
  final List<GameStep> _steps = [];
  bool _done = false;
  int? _result;
  int _currentIndex = 0;
  bool _following = true;

  @override
  void initState() {
    super.initState();
    _connect();
  }

  @override
  void dispose() {
    _reconnectTimer?.cancel();
    _sub?.cancel();
    _channel?.sink.close();
    super.dispose();
  }

  void _connect() {
    _reconnectTimer?.cancel();
    setState(() => _status = 'connecting');
    try {
      final channel = WebSocketChannel.connect(Uri.parse(_url));
      _channel = channel;
      channel.sink.add(jsonEncode({'type': 'hello', 'role': 'viewer'}));
      _sub = channel.stream.listen(
        _onMessage,
        onError: (_) => _scheduleReconnect(),
        onDone: _scheduleReconnect,
        cancelOnError: true,
      );
      setState(() => _status = 'connected');
    } catch (_) {
      _scheduleReconnect();
    }
  }

  void _scheduleReconnect() {
    if (!mounted) return;
    setState(() => _status = 'disconnected');
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(const Duration(seconds: 2), _connect);
  }

  void _onMessage(dynamic raw) {
    final msg = jsonDecode(raw as String) as Map<String, dynamic>;
    switch (msg['type']) {
      case 'snapshot':
        final rawSteps = msg['steps'] as List<dynamic>? ?? [];
        setState(() {
          _gameId = msg['game_id'] as String?;
          _steps
            ..clear()
            ..addAll(rawSteps.map((s) => GameStep.fromJson(s as Map<String, dynamic>)));
          _done = msg['done'] as bool? ?? false;
          _result = msg['result'] as int?;
          _currentIndex = _steps.isEmpty ? 0 : _steps.length - 1;
        });
        break;
      case 'start':
        setState(() {
          _gameId = msg['game_id'] as String?;
          _steps.clear();
          _done = false;
          _result = null;
          _currentIndex = 0;
          _following = true;
        });
        break;
      case 'step':
        final data = msg['data'] as List<dynamic>? ?? [];
        setState(() {
          _steps.addAll(data.map((s) => GameStep.fromJson(s as Map<String, dynamic>)));
          if (_following) _currentIndex = _steps.length - 1;
        });
        break;
      case 'done':
        setState(() {
          _done = true;
          _result = msg['result'] as int?;
        });
        break;
    }
  }

  String _resultText(int? result) {
    switch (result) {
      case 0:
        return 'P0 wins';
      case 1:
        return 'P1 wins';
      case 2:
        return 'Draw';
      default:
        return '?';
    }
  }

  @override
  Widget build(BuildContext context) {
    final current = _steps.isEmpty ? null : _steps[_currentIndex];
    return Scaffold(
      appBar: AppBar(
        title: const Text('PTCG Live Viewer'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Center(child: Text('relay: $_status')),
          ),
        ],
      ),
      body: current == null
          ? const Center(child: Text('Waiting for a game...'))
          : Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Padding(
                  padding: const EdgeInsets.all(8),
                  child: Text(
                    'Game: ${_gameId ?? '-'}   Turn ${current.turn}   '
                    'Step ${_currentIndex + 1}/${_steps.length}   '
                    '${_done ? 'Result: ${_resultText(_result)}' : 'In progress (${current.selectContext})'}',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ),
                Expanded(
                  child: Row(
                    children: [
                      Expanded(
                        child: PlayerPanel(
                          label: 'P0',
                          player: current.players.isNotEmpty ? current.players[0] : null,
                        ),
                      ),
                      Expanded(
                        child: PlayerPanel(
                          label: 'P1',
                          player: current.players.length > 1 ? current.players[1] : null,
                        ),
                      ),
                    ],
                  ),
                ),
                SizedBox(
                  height: 120,
                  child: Card(
                    margin: const EdgeInsets.all(8),
                    child: ListView(
                      padding: const EdgeInsets.all(8),
                      children: current.logs.map((l) => Text(l.toString())).toList(),
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  child: Row(
                    children: [
                      IconButton(
                        icon: Icon(_following ? Icons.play_circle_fill : Icons.play_circle_outline),
                        tooltip: 'Jump to live',
                        onPressed: () => setState(() {
                          _following = true;
                          _currentIndex = _steps.isEmpty ? 0 : _steps.length - 1;
                        }),
                      ),
                      Expanded(
                        child: Slider(
                          value: _currentIndex.toDouble(),
                          min: 0,
                          max: (_steps.length - 1).clamp(0, double.infinity).toDouble(),
                          divisions: _steps.length > 1 ? _steps.length - 1 : null,
                          label: '${_currentIndex + 1}',
                          onChanged: (v) => setState(() {
                            _currentIndex = v.round();
                            _following = _currentIndex == _steps.length - 1;
                          }),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
    );
  }
}

class PlayerPanel extends StatelessWidget {
  final String label;
  final PlayerState? player;

  const PlayerPanel({super.key, required this.label, required this.player});

  @override
  Widget build(BuildContext context) {
    final p = player;
    return Card(
      margin: const EdgeInsets.all(8),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(label, style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 8),
            if (p == null)
              const Text('-')
            else ...[
              Text('Active: ${_cardText(p.active)}'),
              const SizedBox(height: 4),
              Text('Bench: ${p.bench.isEmpty ? '-' : p.bench.map(_cardText).join(', ')}'),
              const SizedBox(height: 8),
              Text('Hand: ${p.handCount}   Deck: ${p.deckCount}   '
                  'Prize: ${p.prizeCount}   Discard: ${p.discardCount}'),
            ],
          ],
        ),
      ),
    );
  }

  String _cardText(GameCard? c) {
    if (c == null) return '-';
    if (c.hp != null && c.maxHp != null) return '${c.name} (${c.hp}/${c.maxHp} HP)';
    return c.name;
  }
}
