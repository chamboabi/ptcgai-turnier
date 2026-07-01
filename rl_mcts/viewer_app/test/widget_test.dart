import 'package:flutter_test/flutter_test.dart';

import 'package:viewer_app/main.dart';

void main() {
  testWidgets('shows waiting state before any relay connection', (WidgetTester tester) async {
    await tester.pumpWidget(const ViewerApp());
    await tester.pump();

    expect(find.text('Waiting for a game...'), findsOneWidget);
  });
}
